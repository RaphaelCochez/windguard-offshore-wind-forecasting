import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from app.feature_engineering import (
    build_era5_features,
    V1_FEATURE_COLS,
    V2_FEATURE_COLS,
    attach_horizon_nwp_features_bulk,
)
from app.models_loader import load_lightgbm_models

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "df26.csv"
V1_DIR = BASE_DIR / "models" / "wind_forecast_v1"
V2_DIR = BASE_DIR / "models" / "wind_forecast_v2"


def make_asymmetric_residual_objective(y_current: np.ndarray, high_wind_threshold: float = 12.0, under_weight: float = 2.5):
    """
    Custom LightGBM objective for predicting wind speed residuals (delta = y_{t+h} - y_t).
    Reconstructs y_pred_abs = y_current + delta_pred so high-wind under-predictions
    (y_true_abs > threshold and y_pred_abs < y_true_abs) receive asymmetric penalty,
    while keeping predictions naturally centered around persistence (y_current).
    """
    def objective(delta_pred: np.ndarray, dataset: lgb.Dataset):
        delta_true = dataset.get_label()
        y_true_abs = y_current + delta_true
        y_pred_abs = y_current + delta_pred

        residual = delta_pred - delta_true
        weights = np.ones_like(delta_pred)

        # Penalize under-prediction during high-wind regimes (> 12 m/s)
        under_mask = (y_true_abs > high_wind_threshold) & (y_pred_abs < y_true_abs)
        weights[under_mask] = under_weight

        grad = weights * residual
        hess = weights
        return grad, hess

    return objective


def make_asymmetric_residual_feval(y_current: np.ndarray, high_wind_threshold: float = 12.0, under_weight: float = 2.5):
    """
    Validation metric strictly aligned with make_asymmetric_residual_objective.
    Prevents the premature early-stopping collapse that occurred when pairing
    an asymmetric training objective with standard symmetric RMSE.
    """
    def feval(delta_pred: np.ndarray, dataset: lgb.Dataset):
        delta_true = dataset.get_label()
        y_true_abs = y_current + delta_true
        y_pred_abs = y_current + delta_pred

        residual = delta_pred - delta_true
        weights = np.ones_like(delta_pred)
        under_mask = (y_true_abs > high_wind_threshold) & (y_pred_abs < y_true_abs)
        weights[under_mask] = under_weight

        weighted_rmse = float(np.sqrt(np.sum(weights * (residual ** 2)) / np.sum(weights)))
        return "asym_wrmse", weighted_rmse, False

    return feval


def get_horizon_features(df_feat: pd.DataFrame, h: int) -> pd.DataFrame:
    """Adds target-horizon cyclic hour features and synoptic NWP guidance features (t + h)."""
    return attach_horizon_nwp_features_bulk(df_feat, h)


def train_and_evaluate():
    print(f"Loading dataset from {DATA_PATH}...")
    df_raw = pd.read_csv(DATA_PATH)
    time_col = "valid_time" if "valid_time" in df_raw.columns else "time"
    df_raw[time_col] = pd.to_datetime(df_raw[time_col])
    df_raw = df_raw.set_index(time_col).sort_index()

    print("Building enhanced V2 meteorological features...")
    df_feat = build_era5_features(df_raw)

    # Load existing v1 models for comparison and ensemble calibration
    v1_models = load_lightgbm_models(V1_DIR)

    # Chronological splits:
    # Train: Full Year 2025 (8,760h across all seasons)
    # Val:   Jan 1, 2026 - Feb 14, 2026 (1,080h winter wind regime)
    # Test:  Feb 15, 2026 - Jun 30, 2026 (3,240h completely held-out test window)
    V2_DIR.mkdir(parents=True, exist_ok=True)

    target_df = pd.DataFrame(
        {f"target_h{h}": df_feat["wind_speed_100m"].shift(-h) for h in range(1, 25)},
        index=df_feat.index,
    )
    # Ensure all 24 future hours are contiguous hourly steps (no shifting across archive date gaps)
    contiguous_24h = (df_feat.index.to_series().shift(-24) - df_feat.index) == pd.Timedelta(hours=24)
    valid_rows = target_df.notna().all(axis=1) & contiguous_24h
    df_all = df_feat.loc[valid_rows].copy()
    targets_all = target_df.loc[valid_rows].copy()

    train_mask = df_all.index < "2026-01-01"
    val_mask = (df_all.index >= "2026-01-01") & (df_all.index < "2026-02-15")
    test_mask = df_all.index >= "2026-02-15"

    y_curr_train = df_all.loc[train_mask, "wind_speed_100m"].values
    y_curr_val = df_all.loc[val_mask, "wind_speed_100m"].values
    y_curr_test = df_all.loc[test_mask, "wind_speed_100m"].values

    X_v1_val = df_all.loc[val_mask, V1_FEATURE_COLS]
    X_v1_test = df_all.loc[test_mask, V1_FEATURE_COLS]

    params = {
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "metric": "None",  # Use custom aligned feval exclusively
        "verbosity": -1,
        "seed": 42,
    }

    metrics_summary = {"horizons": {}, "ensemble_weights": {}}

    print("Training 24 residual LightGBM models (wind_forecast_v2) with aligned asymmetric loss...")
    for h in range(1, 25):
        X_h = get_horizon_features(df_all, h)
        X_train, X_val, X_test = X_h.loc[train_mask], X_h.loc[val_mask], X_h.loc[test_mask]

        y_train_abs = targets_all.loc[train_mask, f"target_h{h}"].values
        y_val_abs = targets_all.loc[val_mask, f"target_h{h}"].values
        y_test_abs = targets_all.loc[test_mask, f"target_h{h}"].values

        # Physical baseline blend: transitions from persistence y_t to synoptic_nwp_ws_h
        alpha_h = min(0.92, float(h) / (float(h) + 1.2))
        y_base_train = (1.0 - alpha_h) * y_curr_train + alpha_h * X_train["synoptic_nwp_ws_h"].values
        y_base_val = (1.0 - alpha_h) * y_curr_val + alpha_h * X_val["synoptic_nwp_ws_h"].values
        y_base_test = (1.0 - alpha_h) * y_curr_test + alpha_h * X_test["synoptic_nwp_ws_h"].values

        # Residual targets relative to y_base_h (prevents tree leaf clipping during >30 m/s storms)
        delta_train = y_train_abs - y_base_train
        delta_val = y_val_abs - y_base_val

        dtrain = lgb.Dataset(X_train, label=delta_train)
        dval = lgb.Dataset(X_val, label=delta_val, reference=dtrain)

        fobj = make_asymmetric_residual_objective(y_base_train, high_wind_threshold=18.0, under_weight=1.45)
        feval = make_asymmetric_residual_feval(y_base_val, high_wind_threshold=18.0, under_weight=1.45)

        h_params = dict(params)
        h_params["objective"] = fobj

        booster = lgb.train(
            h_params,
            dtrain,
            num_boost_round=800,
            valid_sets=[dval],
            valid_names=["val"],
            feval=feval,
            callbacks=[lgb.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=False)],
        )

        model_path = V2_DIR / f"lgb_model_h_{h}.txt"
        booster.save_model(str(model_path))

        # Validation & Test predictions for v2 (reconstructed from physical baseline: y_base + delta_hat)
        val_pred_v2 = np.clip(y_base_val + booster.predict(X_val), 0.0, None)
        test_pred_v2 = np.clip(y_base_test + booster.predict(X_test), 0.0, None)

        # Compare with v1 if available and learn optimal validation blend weight w_v2 in [0, 1]
        if f"h_{h}" in v1_models:
            val_pred_v1 = np.clip(v1_models[f"h_{h}"].predict(X_v1_val), 0.0, None)
            test_pred_v1 = np.clip(v1_models[f"h_{h}"].predict(X_v1_test), 0.0, None)

            # Grid-search optimal blend weight w on validation set (optimizing overall + high-wind RMSE)
            best_w = 0.9
            best_loss = float("inf")
            for w in np.linspace(0.5, 1.0, 21):
                blend_val = w * val_pred_v2 + (1.0 - w) * val_pred_v1
                err = blend_val - y_val_abs
                hw_mask = (y_val_abs > 15.0) & (blend_val < y_val_abs)
                sample_w = np.where(hw_mask, 1.5, 1.0)
                loss = float(np.sqrt(np.mean(sample_w * (err ** 2))))
                if loss < best_loss:
                    best_loss = loss
                    best_w = float(round(w, 3))
        else:
            test_pred_v1 = y_curr_test.copy()
            best_w = 1.0

        test_pred_ens = best_w * test_pred_v2 + (1.0 - best_w) * test_pred_v1

        # Persistence baseline on test set
        rmse_pers = float(np.sqrt(np.mean((y_curr_test - y_test_abs) ** 2)))
        rmse_v1 = float(np.sqrt(np.mean((test_pred_v1 - y_test_abs) ** 2)))
        rmse_v2 = float(np.sqrt(np.mean((test_pred_v2 - y_test_abs) ** 2)))
        rmse_ens = float(np.sqrt(np.mean((test_pred_ens - y_test_abs) ** 2)))

        mae_v1 = float(np.mean(np.abs(test_pred_v1 - y_test_abs)))
        mae_v2 = float(np.mean(np.abs(test_pred_v2 - y_test_abs)))
        mae_ens = float(np.mean(np.abs(test_pred_ens - y_test_abs)))

        # High-wind (> 12 m/s) bias & RMSE on test set
        hw_test = y_test_abs > 12.0
        if hw_test.sum() > 0:
            rmse_hw_v1 = float(np.sqrt(np.mean((test_pred_v1[hw_test] - y_test_abs[hw_test]) ** 2)))
            rmse_hw_v2 = float(np.sqrt(np.mean((test_pred_v2[hw_test] - y_test_abs[hw_test]) ** 2)))
            rmse_hw_ens = float(np.sqrt(np.mean((test_pred_ens[hw_test] - y_test_abs[hw_test]) ** 2)))
            bias_hw_v1 = float(np.mean(test_pred_v1[hw_test] - y_test_abs[hw_test]))
            bias_hw_v2 = float(np.mean(test_pred_v2[hw_test] - y_test_abs[hw_test]))
            bias_hw_ens = float(np.mean(test_pred_ens[hw_test] - y_test_abs[hw_test]))
        else:
            rmse_hw_v1 = rmse_hw_v2 = rmse_hw_ens = 0.0
            bias_hw_v1 = bias_hw_v2 = bias_hw_ens = 0.0

        metrics_summary["ensemble_weights"][f"h_{h}"] = best_w
        metrics_summary["horizons"][f"h_{h}"] = {
            "horizon": f"+{h}h",
            "v2_trees": booster.num_trees(),
            "v2_weight": best_w,
            "rmse_persistence": round(rmse_pers, 3),
            "rmse_v1": round(rmse_v1, 3),
            "rmse_v2": round(rmse_v2, 3),
            "rmse_ensemble": round(rmse_ens, 3),
            "mae_v1": round(mae_v1, 3),
            "mae_v2": round(mae_v2, 3),
            "mae_ensemble": round(mae_ens, 3),
            "high_wind_rmse_v1": round(rmse_hw_v1, 3),
            "high_wind_rmse_v2": round(rmse_hw_v2, 3),
            "high_wind_rmse_ensemble": round(rmse_hw_ens, 3),
            "high_wind_bias_v1": round(bias_hw_v1, 3),
            "high_wind_bias_v2": round(bias_hw_v2, 3),
            "high_wind_bias_ensemble": round(bias_hw_ens, 3),
        }
        print(
            f"  +h={h:02d}h | trees={booster.num_trees():3d} | "
            f"RMSE: v1={rmse_v1:.3f}, v2={rmse_v2:.3f}, ens={rmse_ens:.3f} | "
            f"HighWind Bias: v1={bias_hw_v1:+.2f}, v2={bias_hw_v2:+.2f}, ens={bias_hw_ens:+.2f}"
        )

    summary_path = V2_DIR / "metrics_summary.json"
    summary_path.write_text(json.dumps(metrics_summary, indent=2))
    print(f"Saved 24 v2 models and metrics summary to {V2_DIR}")


if __name__ == "__main__":
    train_and_evaluate()
