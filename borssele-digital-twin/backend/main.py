from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import json
import time
import asyncio
from datetime import datetime, timezone

from app.config import (
    SERVICE_TITLE,
    OPERATIONAL_QUESTION,
    CAUTION_SPEED_MS,
    SUPPORTED_HORIZONS,
    DEFAULT_HORIZON_HOURS,
    MAX_HORIZON_HOURS,
    CONSENSUS_THRESHOLD_MS,
    MIN_CONSENSUS_AGREEMENT_PCT,
    BACKGROUND_SYNC_INTERVAL_SECONDS,
)
from app.feature_engineering import (
    engineer_features_for_inference,
    attach_horizon_nwp_features_single,
    V1_FEATURE_COLS,
    V2_FEATURE_COLS,
)
from app.physics_engine import evaluate_turbine_operability
from app.era5_updater import (
    OFFSHORE_AREAS,
    fetch_updated_era5_for_area,
    fetch_historical_era5_window,
)

app = FastAPI(title=SERVICE_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_V1: Dict[int, lgb.Booster] = {}
MODELS_V2: Dict[int, lgb.Booster] = {}
ENSEMBLE_WEIGHTS: Dict[int, float] = {}
METRICS_SUMMARY: Dict[str, Any] = {}
HISTORICAL_DF: Optional[pd.DataFrame] = None
DATA_DIR = "data"
BACKGROUND_SYNC_STATE: Dict[str, Any] = {
    "status": "INITIALIZING",
    "last_sync_at": None,
    "interval_seconds": BACKGROUND_SYNC_INTERVAL_SECONDS,
    "cycles_completed": 0,
}


async def _background_era5_sync_loop():
    """Continuously fetches updated ERA5 / multi-NWP telemetry for all corridors in the background."""
    while True:
        try:
            for area_key in OFFSHORE_AREAS.keys():
                await asyncio.to_thread(
                    fetch_updated_era5_for_area,
                    area_key,
                    DATA_DIR,
                    True,
                    None,
                    HISTORICAL_DF,
                    False,
                )
            BACKGROUND_SYNC_STATE["status"] = "ACTIVE (Auto-Syncing in Background)"
            BACKGROUND_SYNC_STATE["last_sync_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            BACKGROUND_SYNC_STATE["cycles_completed"] += 1
        except Exception as exc:
            BACKGROUND_SYNC_STATE["status"] = f"FALLBACK_STANDBY ({exc})"
        await asyncio.sleep(BACKGROUND_SYNC_STATE["interval_seconds"])


# Three Historical North Sea Storm & Shut-Down Events
HISTORICAL_EVENTS = [
    {
        "id": "storm_eunice_shutdown",
        "title": "Historical Event 1: Storm Eunice (2022-02-18 • 31.3 m/s Cut-Out Shut-Down)",
        "timestamp": "2022-02-18 06:00:00",
        "area_id": "borssele",
        "description": "KNMI Code Red & TenneT/Elia documented North Sea wind park protective cut-out shut-down (10h >= 25.0 m/s from 12:00–21:00 UTC, peaking at 31.31 m/s).",
    },
    {
        "id": "storm_ciaran_shutdown",
        "title": "Historical Event 2: Storm Ciarán (2023-11-02 • 32.6 m/s Cut-Out Shut-Down)",
        "timestamp": "2023-11-02 00:00:00",
        "area_id": "borssele",
        "description": "961.6 hPa Bomb Cyclone & documented Southern North Sea cut-out shut-down (5h >= 25.0 m/s from 09:00–13:00 UTC, peaking at 32.56 m/s).",
    },
    {
        "id": "event_3_new_year_gale",
        "title": "Historical Event 3: New Year Gale (2025-01-01 • 24.5 m/s High Wind Caution)",
        "timestamp": "2025-01-01 06:00:00",
        "area_id": "borssele",
        "description": "Near-threshold severe gale peaking at 24.52 m/s (0.48 m/s below the 25.0 m/s cut-out limit) — verifies High Wind Caution without false shut-down.",
    },
]


class ForecastRequest(BaseModel):
    timestamp: str = "LIVE"
    area_id: str = "borssele"
    mode: Optional[str] = "ensemble"  # "ensemble" | "v2" | "v1" | "persistence" | "fallback_nwp"
    horizon_hours: Optional[int] = DEFAULT_HORIZON_HOURS  # 1 | 6 | 12 | 24 (max: 24)
    system_mode: Optional[str] = "normal"  # "normal" | "degraded_fallback"
    simulate_primary_outage: Optional[bool] = False


class Era5SyncRequest(BaseModel):
    area_id: str = "borssele"
    cds_api_key: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    global HISTORICAL_DF, METRICS_SUMMARY, ENSEMBLE_WEIGHTS, DATA_DIR

    data_paths = [
        "data/df26.csv",
        "borssele-digital-twin/backend/data/df26.csv",
    ]
    for p in data_paths:
        if os.path.exists(p):
            DATA_DIR = os.path.dirname(p) or "data"
            print(f"Loading collected ERA5 telemetry dataset from {p}...")
            HISTORICAL_DF = pd.read_csv(p)
            time_col = "valid_time" if "valid_time" in HISTORICAL_DF.columns else "time"
            HISTORICAL_DF["valid_time"] = pd.to_datetime(HISTORICAL_DF[time_col])
            HISTORICAL_DF = (
                HISTORICAL_DF.sort_values("valid_time")
                .drop_duplicates("valid_time")
                .reset_index(drop=True)
            )
            break

    v1_dirs = [
        "models/wind_forecast_v1",
        "borssele-digital-twin/backend/models/wind_forecast_v1",
    ]
    for d in v1_dirs:
        if os.path.exists(d):
            for h in range(1, MAX_HORIZON_HOURS + 1):
                mpath = os.path.join(d, f"lgb_model_h_{h}.txt")
                if os.path.exists(mpath):
                    MODELS_V1[h] = lgb.Booster(model_file=mpath)
            break

    v2_dirs = [
        "models/wind_forecast_v2",
        "borssele-digital-twin/backend/models/wind_forecast_v2",
    ]
    for d in v2_dirs:
        if os.path.exists(d):
            for h in range(1, MAX_HORIZON_HOURS + 1):
                mpath = os.path.join(d, f"lgb_model_h_{h}.txt")
                if os.path.exists(mpath):
                    MODELS_V2[h] = lgb.Booster(model_file=mpath)
            summary_path = os.path.join(d, "metrics_summary.json")
            if os.path.exists(summary_path):
                with open(summary_path, "r") as f:
                    METRICS_SUMMARY = json.load(f)
                    for k, v in METRICS_SUMMARY.get("horizons", {}).items():
                        h_idx = int(k.replace("h_", ""))
                        ENSEMBLE_WEIGHTS[h_idx] = float(v.get("v2_weight", v.get("ensemble_w_v2", 1.0)))
            break

    print(
        f"Loaded {len(MODELS_V1)} v1 models and {len(MODELS_V2)} v2 residual models."
    )
    asyncio.create_task(_background_era5_sync_loop())


@app.get("/api/health")
def get_health():
    min_t = (
        HISTORICAL_DF["valid_time"].iloc[48].strftime("%Y-%m-%d %H:%M:%S")
        if HISTORICAL_DF is not None
        else None
    )
    max_t = (
        HISTORICAL_DF["valid_time"].iloc[-25].strftime("%Y-%m-%d %H:%M:%S")
        if HISTORICAL_DF is not None
        else None
    )
    return {
        "status": "ok",
        "service_title": SERVICE_TITLE,
        "operational_question": OPERATIONAL_QUESTION,
        "v1_models_loaded": len(MODELS_V1),
        "v2_models_loaded": len(MODELS_V2),
        "supported_horizons": SUPPORTED_HORIZONS,
        "default_horizon_hours": DEFAULT_HORIZON_HOURS,
        "areas": list(OFFSHORE_AREAS.keys()),
        "historical_events": HISTORICAL_EVENTS,
        "dataset_range": {"min": min_t, "max": max_t},
    }


@app.get("/api/areas")
def get_areas():
    return {
        "default_area": "borssele",
        "areas": OFFSHORE_AREAS,
        "historical_events": HISTORICAL_EVENTS,
    }


@app.post("/api/era5/sync")
def sync_era5_data(req: Era5SyncRequest):
    area_id = req.area_id if req.area_id in OFFSHORE_AREAS else "borssele"
    try:
        df_synced, now_str, sync_meta = fetch_updated_era5_for_area(
            area_id=area_id,
            data_dir=DATA_DIR,
            force_refresh=True,
            cds_api_key=req.cds_api_key,
            fallback_df=HISTORICAL_DF,
        )
        return {
            "status": "synced",
            "area": OFFSHORE_AREAS[area_id],
            "sync_meta": sync_meta,
            "latest_timestamp": now_str,
            "rows": len(df_synced),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/metrics")
def get_metrics():
    if not METRICS_SUMMARY:
        raise HTTPException(status_code=404, detail="Metrics summary not found.")
    return METRICS_SUMMARY


@app.post("/api/predict")
def get_prediction(req: ForecastRequest):
    t_start = time.perf_counter()

    if HISTORICAL_DF is None:
        raise HTTPException(status_code=500, detail="Backend ERA5 dataset not loaded.")

    area_id = req.area_id if req.area_id in OFFSHORE_AREAS else "borssele"
    area_info = OFFSHORE_AREAS[area_id]

    # Strictly enforce max +24h horizon
    horizon_hours = int(req.horizon_hours or DEFAULT_HORIZON_HOURS)
    if horizon_hours not in SUPPORTED_HORIZONS:
        horizon_hours = min(MAX_HORIZON_HOURS, max(1, horizon_hours))

    requested_sys_mode = (req.system_mode or "normal").lower()
    requested_mode = (req.mode or "ensemble").lower()
    simulate_outage = bool(req.simulate_primary_outage)

    requested_ts = (req.timestamp or "LIVE").strip()
    is_live_mode = requested_ts.upper() in ("LIVE", "NOW", "TODAY", "")

    sync_meta: Dict[str, Any] = {}
    if is_live_mode:
        active_df, resolved_ts_str, sync_meta = fetch_updated_era5_for_area(
            area_id=area_id,
            data_dir=DATA_DIR,
            force_refresh=False,
            fallback_df=HISTORICAL_DF,
            simulate_primary_outage=simulate_outage,
        )
        target_dt = pd.to_datetime(resolved_ts_str)
    else:
        try:
            target_dt = pd.to_datetime(requested_ts)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid timestamp '{requested_ts}'. Use 'LIVE' or 'YYYY-MM-DD HH:MM:SS'.",
            )

        # Prevent selecting a future origin timestamp (> current UTC hour), capping horizon at +24h from now
        now_dt = pd.to_datetime(datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0, tzinfo=None))
        if target_dt > now_dt:
            is_live_mode = True

        if not is_live_mode:
            active_df, sync_meta = fetch_historical_era5_window(
                area_id=area_id,
                target_dt=target_dt,
                fallback_df=HISTORICAL_DF,
            )
        else:
            active_df, live_now_str, sync_meta = fetch_updated_era5_for_area(
                area_id=area_id,
                data_dir=DATA_DIR,
                force_refresh=False,
                fallback_df=HISTORICAL_DF,
                simulate_primary_outage=simulate_outage,
            )
            target_dt = pd.to_datetime(live_now_str)

    idx_list = active_df.index[active_df["valid_time"] == target_dt].tolist()
    if not idx_list:
        diffs = (active_df["valid_time"] - target_dt).abs()
        current_idx = int(diffs.idxmin())
        target_dt = active_df.loc[current_idx, "valid_time"]
    else:
        current_idx = idx_list[0]

    start_idx = max(0, current_idx - 48)
    df_slice = active_df.iloc[start_idx : current_idx + 1].copy()
    if len(df_slice) < 2:
        df_slice = pd.concat([df_slice.iloc[[0]], df_slice], ignore_index=True)

    try:
        X_v1 = engineer_features_for_inference(df_slice, version="v1")[V1_FEATURE_COLS]
        X_v2 = engineer_features_for_inference(df_slice, version="v2")[V2_FEATURE_COLS]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature engineering failed: {str(e)}")

    u100_now = float(df_slice["u100"].iloc[-1])
    v100_now = float(df_slice["v100"].iloc[-1])
    current_ws = float(np.sqrt(u100_now**2 + v100_now**2))
    current_dir_deg = float((270.0 - np.rad2deg(np.arctan2(v100_now, u100_now))) % 360.0)

    u100_prev1 = float(df_slice["u100"].iloc[-2])
    v100_prev1 = float(df_slice["v100"].iloc[-2])
    prev1_ws = float(np.sqrt(u100_prev1**2 + v100_prev1**2))
    ramp_1h = current_ws - prev1_ws

    has_secondary_col = "secondary_nwp_ws100" in active_df.columns
    raw_24h: List[Dict[str, Any]] = []

    for h in range(1, MAX_HORIZON_HOURS + 1):
        future_idx = current_idx + h
        sec_ext_ws: Optional[float] = None

        if future_idx < len(active_df):
            u_act = float(active_df.loc[future_idx, "u100"])
            v_act = float(active_df.loc[future_idx, "v100"])
            era5_or_primary_ws = float(np.sqrt(u_act**2 + v_act**2))
            future_time = active_df.loc[future_idx, "valid_time"].strftime("%Y-%m-%d %H:%M")
            if has_secondary_col:
                raw_sec = active_df.loc[future_idx, "secondary_nwp_ws100"]
                if pd.notna(raw_sec):
                    sec_ext_ws = float(raw_sec)
        else:
            era5_or_primary_ws = None
            future_time = (target_dt + pd.Timedelta(hours=h)).strftime("%Y-%m-%d %H:%M")

        pred_pers = max(0.0, current_ws)

        if h in MODELS_V1:
            pred_v1 = float(np.clip(MODELS_V1[h].predict(X_v1)[0], 0.0, 45.0))
        else:
            pred_v1 = pred_pers

        # Always compute horizon synoptic NWP features so secondary fallback never degrades to persistence
        X_v2_h = attach_horizon_nwp_features_single(
            X_v2, active_df, current_idx, h, is_live_mode=is_live_mode
        )
        nwp_synoptic_h = float(X_v2_h["synoptic_nwp_ws_h"].iloc[0])

        if h in MODELS_V2:
            alpha_h = min(0.92, float(h) / (float(h) + 1.2))
            y_base_h = (1.0 - alpha_h) * current_ws + alpha_h * nwp_synoptic_h
            pred_delta = float(MODELS_V2[h].predict(X_v2_h)[0])
            pred_v2 = float(np.clip(y_base_h + pred_delta, 0.0, 45.0))
        else:
            pred_v2 = pred_v1

        w_v2 = ENSEMBLE_WEIGHTS.get(h, 0.95)
        pred_ens = float(np.clip(w_v2 * pred_v2 + (1.0 - w_v2) * pred_v1, 0.0, 45.0))

        # Secondary NWP stream: prefer independent MET Norway / DWD ICON in live mode when available
        nwp_secondary = (
            sec_ext_ws
            if sec_ext_ws is not None
            else float(0.80 * nwp_synoptic_h + 0.20 * pred_pers)
        )

        # Conservative safety envelope whenever winds enter Caution (>=20 m/s) during degraded fallback
        pred_fallback = float(
            max(nwp_secondary, pred_ens)
            if max(nwp_secondary, pred_ens) >= CAUTION_SPEED_MS
            else nwp_secondary
        )

        # Reference curve for chart & accuracy: real independent secondary NWP in live mode (if fetched),
        # or historical ERA5 verification truth during archive replay
        ref_ws = (
            sec_ext_ws
            if (is_live_mode and sec_ext_ws is not None)
            else era5_or_primary_ws
        )

        source_spread = float(
            max(pred_v2, pred_v1, nwp_secondary) - min(pred_v2, pred_v1, nwp_secondary)
        )
        raw_24h.append({
            "step": h,
            "horizon": f"+{h}h",
            "timestamp": future_time,
            "pred_ens": pred_ens,
            "pred_v2": pred_v2,
            "pred_v1": pred_v1,
            "pred_fallback": pred_fallback,
            "pred_pers": pred_pers,
            "nwp_secondary": nwp_secondary,
            "ref_ws": ref_ws,
            "source_spread": source_spread,
        })

    # Automatic Safe Fallback Detection (per Box 9: Failure and Safe Fallback)
    missing_ml_models = (len(MODELS_V2) < MAX_HORIZON_HOURS) or (len(MODELS_V1) == 0)
    primary_stream_lost = is_live_mode and not bool(sync_meta.get("primary_sync_ok", True))

    prelim_window = raw_24h[:horizon_hours]
    prelim_scores = [
        max(0.0, 1.0 - (r["source_spread"] / max(r["pred_ens"], 6.0))) * 100.0
        for r in prelim_window
    ]
    prelim_consensus_pct = float(np.mean(prelim_scores))
    high_wind_present = any(
        max(r["pred_ens"], r["nwp_secondary"]) >= CAUTION_SPEED_MS for r in prelim_window
    )
    consensus_breach = high_wind_present and (prelim_consensus_pct < MIN_CONSENSUS_AGREEMENT_PCT)

    manual_fallback = (requested_sys_mode == "degraded_fallback") or (requested_mode == "fallback_nwp")

    if manual_fallback:
        is_fallback_active = True
        auto_fallback_triggered = False
        fallback_trigger_reason = "MANUAL_OPERATOR_SAFE_FALLBACK"
    elif primary_stream_lost:
        is_fallback_active = True
        auto_fallback_triggered = True
        fallback_trigger_reason = "AUTO_FALLBACK_PRIMARY_STREAM_LOSS"
    elif missing_ml_models:
        is_fallback_active = True
        auto_fallback_triggered = True
        fallback_trigger_reason = "AUTO_FALLBACK_ML_MODELS_UNAVAILABLE"
    elif consensus_breach:
        is_fallback_active = True
        auto_fallback_triggered = True
        fallback_trigger_reason = "AUTO_FALLBACK_HIGH_WIND_CONSENSUS_DIVERGENCE"
    else:
        is_fallback_active = False
        auto_fallback_triggered = False
        fallback_trigger_reason = "NONE (Nominal Multi-Source Consensus)"

    effective_system_mode = (
        "auto_degraded_fallback"
        if auto_fallback_triggered
        else ("degraded_fallback" if is_fallback_active else "normal")
    )

    full_24h: List[Dict[str, Any]] = []
    for r in raw_24h:
        if is_fallback_active:
            active_pred = r["pred_fallback"]
        elif requested_mode == "v1":
            active_pred = r["pred_v1"]
        elif requested_mode == "v2":
            active_pred = r["pred_v2"]
        elif requested_mode == "persistence":
            active_pred = r["pred_pers"]
        else:
            active_pred = r["pred_ens"]

        consensus_score_h = max(0.0, 1.0 - (r["source_spread"] / max(active_pred, 6.0)))
        consensus_met = bool(r["source_spread"] <= CONSENSUS_THRESHOLD_MS)

        op_pred = evaluate_turbine_operability(active_pred)
        op_act = evaluate_turbine_operability(r["ref_ws"]) if r["ref_ws"] is not None else None

        full_24h.append({
            "step": r["step"],
            "horizon": r["horizon"],
            "timestamp": r["timestamp"],
            "predicted": round(active_pred, 2),
            "predicted_ensemble": round(r["pred_ens"], 2),
            "predicted_v2": round(r["pred_v2"], 2),
            "predicted_v1": round(r["pred_v1"], 2),
            "predicted_nwp_fallback": round(r["pred_fallback"], 2),
            "secondary_nwp": round(r["nwp_secondary"], 2),
            "predicted_persistence": round(r["pred_pers"], 2),
            "actual": round(r["ref_ws"], 2) if r["ref_ws"] is not None else None,
            "delta_from_now": round(active_pred - current_ws, 2),
            "source_spread_ms": round(r["source_spread"], 2),
            "consensus_score_pct": round(consensus_score_h * 100.0, 1),
            "consensus_met": consensus_met,
            "canOperate": op_pred["can_operate"],
            "operationalState": op_pred["state_code"],
            "operationalLabel": op_pred["state_label"],
            "operationalReason": op_pred["reason"],
            "marginToCutOut": op_pred["margin_to_cutout_ms"],
            "marginAboveCutIn": op_pred["margin_above_cutin_ms"],
            "actualCanOperate": op_act["can_operate"] if op_act is not None else None,
            "actualOperationalState": op_act["state_code"] if op_act is not None else None,
        })

    forecast_results = full_24h[:horizon_hours]

    key_horizons = {}
    for kh in SUPPORTED_HORIZONS:
        if kh <= len(full_24h):
            key_horizons[f"+{kh}h"] = full_24h[kh - 1]

    valid_actuals = [r for r in forecast_results if r["actual"] is not None]
    consensus_rate = round(
        float(np.mean([r["consensus_score_pct"] for r in forecast_results])),
        1,
    )
    max_pred_ws = round(float(max(r["predicted"] for r in forecast_results)), 2)
    min_pred_ws = round(float(min(r["predicted"] for r in forecast_results)), 2)

    if valid_actuals:
        err_active = np.array([r["predicted"] - r["actual"] for r in valid_actuals])
        err_v1 = np.array([r["predicted_v1"] - r["actual"] for r in valid_actuals])
        err_v2 = np.array([r["predicted_v2"] - r["actual"] for r in valid_actuals])
        err_pers = np.array([r["predicted_persistence"] - r["actual"] for r in valid_actuals])
        rel_acc_points = [
            max(0.0, 1.0 - abs(r["predicted"] - r["actual"]) / max(r["actual"], r["predicted"], 4.0))
            for r in valid_actuals
        ]
        regime_matches = [
            1.0 if r["operationalState"] == r["actualOperationalState"] else 0.0
            for r in valid_actuals
        ]
        window_metrics = {
            "horizon_evaluated": f"+{horizon_hours}h",
            "rmse_active": round(float(np.sqrt(np.mean(err_active**2))), 2),
            "rmse_v2": round(float(np.sqrt(np.mean(err_v2**2))), 2),
            "rmse_v1": round(float(np.sqrt(np.mean(err_v1**2))), 2),
            "rmse_persistence": round(float(np.sqrt(np.mean(err_pers**2))), 2),
            "mae_active": round(float(np.mean(np.abs(err_active))), 2),
            "operability_accuracy_pct": round(float(np.mean(rel_acc_points) * 100.0), 1),
            "regime_match_pct": round(float(np.mean(regime_matches) * 100.0), 1),
            "consensus_agreement_pct": consensus_rate,
            "actualcutout_hours": int(sum(1 for r in valid_actuals if r["actualOperationalState"] == "CUT_OUT_SHUTDOWN")),
            "max_actual_ws_ms": round(float(max(r["actual"] for r in valid_actuals)), 2),
            "operable_hours": int(sum(1 for r in forecast_results if r["canOperate"])),
            "cutout_hours": int(sum(1 for r in forecast_results if r["operationalState"] == "CUT_OUT_SHUTDOWN")),
            "caution_hours": int(sum(1 for r in forecast_results if r["operationalState"] == "HIGH_WIND_CAUTION")),
            "below_cutin_hours": int(sum(1 for r in forecast_results if r["operationalState"] == "LOW_WIND_STANDBY")),
            "min_cutout_margin_ms": round(float(min(r["marginToCutOut"] for r in forecast_results)), 2),
            "max_predicted_ws_ms": max_pred_ws,
            "min_predicted_ws_ms": min_pred_ws,
        }
    else:
        window_metrics = {
            "horizon_evaluated": f"+{horizon_hours}h",
            "rmse_active": None,
            "operability_accuracy_pct": None,
            "regime_match_pct": None,
            "consensus_agreement_pct": consensus_rate,
            "actualcutout_hours": 0,
            "max_actual_ws_ms": None,
            "operable_hours": int(sum(1 for r in forecast_results if r["canOperate"])),
            "cutout_hours": int(sum(1 for r in forecast_results if r["operationalState"] == "CUT_OUT_SHUTDOWN")),
            "caution_hours": int(sum(1 for r in forecast_results if r["operationalState"] == "HIGH_WIND_CAUTION")),
            "below_cutin_hours": int(sum(1 for r in forecast_results if r["operationalState"] == "LOW_WIND_STANDBY")),
            "min_cutout_margin_ms": round(float(min(r["marginToCutOut"] for r in forecast_results)), 2),
            "max_predicted_ws_ms": max_pred_ws,
            "min_predicted_ws_ms": min_pred_ws,
        }

    # Grid Operator Decision Support & Safe Fallback Assessment
    ref_label = "Secondary NWP" if is_live_mode else "ERA5 Reference"
    if window_metrics["cutout_hours"] > 0:
        alert_level = "CRITICAL_SHUTDOWN_ALERT"
        grid_strategy = (
            f"VERIFIED CUT-OUT SHUT-DOWN: Forecast predicts {window_metrics['cutout_hours']}h >= 25.0 m/s (peak {max_pred_ws} m/s) "
            f"vs. {ref_label} {window_metrics['actualcutout_hours']}h >= 25.0 m/s (peak {window_metrics['max_actual_ws_ms']} m/s). "
            "Wind park closed via automatic 90° blade pitch feathering; activate HVDC spinning reserves."
        )
    elif window_metrics["caution_hours"] > 0:
        alert_level = "HIGH_WIND_CAUTION_ALERT"
        grid_strategy = (
            f"HIGH WIND CAUTION ({window_metrics['caution_hours']}h between 20.0–25.0 m/s, min margin {window_metrics['min_cutout_margin_ms']} m/s). "
            "Place secondary NWP consensus watch on standby and pre-allocate short-term balancing reserves."
        )
    elif window_metrics["below_cutin_hours"] > 0:
        alert_level = "LOW_WIND_STANDBY"
        grid_strategy = (
            f"LOW WIND (<3.0 m/s for {window_metrics['below_cutin_hours']}h within +{horizon_hours}h). "
            "Wind park can operate safely (no storm shut-down risk), but calm wind is below the 3.0 m/s turbine cut-in threshold."
        )
    else:
        alert_level = "GOOD_CONDITIONS"
        grid_strategy = (
            f"GOOD CONDITIONS: All {horizon_hours}h within 3.0–20.0 m/s operating range (peak {max_pred_ws} m/s). "
            "Maintain standard grid dispatch and routine 1h ERA5/NWP verification cycle."
        )

    latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

    sec_sources = sync_meta.get("secondary_sources") or ["Synoptic NWP Guidance"]
    system_mode_label = (
        f"AUTO SAFE FALLBACK ({fallback_trigger_reason})"
        if auto_fallback_triggered
        else (
            "MANUAL SAFE FALLBACK (Secondary NWP Active)"
            if is_fallback_active
            else "NORMAL (Primary ERA5/NWP + LightGBM Ensemble)"
        )
    )

    proposal_telemetry = {
        "system_mode": system_mode_label,
        "auto_fallback_triggered": auto_fallback_triggered,
        "fallback_trigger_reason": fallback_trigger_reason,
        "independent_nwp_sources": sec_sources,
        "alert_level": alert_level,
        "grid_operator_strategy": grid_strategy,
        "detection_to_alert_latency_ms": latency_ms,
        "update_interval": "1 hour (ERA5 / Multi-NWP Hourly Cycle)",
        "data_freshness": (
            "DEGRADED STREAM (Safe Fallback Cache + Secondary NWP)"
            if primary_stream_lost
            else ("LIVE SYNC (<1h lag)" if is_live_mode else "HISTORICAL EVENT REPLAY")
        ),
        "consensus_threshold_ms": CONSENSUS_THRESHOLD_MS,
        "consensus_agreement_pct": consensus_rate,
        "fallback_status": (
            f"ACTIVE ({' + '.join(sec_sources)})"
            if is_fallback_active
            else "STANDBY (Auto-Triggers on Stream Loss / Divergence)"
        ),
        "notification_channels": [
            {"channel": "Grid Control Room Dashboard", "status": "DELIVERED (Live UI)"},
            {"channel": "SCADA Alert Payload Bus", "status": f"READY ({alert_level})"},
            {"channel": "Multi-Source NWP Consensus Gate", "status": "PASS" if consensus_rate >= MIN_CONSENSUS_AGREEMENT_PCT else "REVIEW"},
        ],
    }

    current_op = evaluate_turbine_operability(current_ws)

    return {
        "timestamp": target_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "requested_timestamp": requested_ts,
        "is_live": is_live_mode,
        "area": area_info,
        "sync_meta": sync_meta,
        "background_sync": BACKGROUND_SYNC_STATE,
        "horizon_hours": horizon_hours,
        "mode": requested_mode,
        "system_mode": effective_system_mode,
        "current_actual": round(current_ws, 2),
        "current_operability": current_op,
        "current_wind_dir_deg": round(current_dir_deg, 1),
        "current_ramp_1h": round(ramp_1h, 2),
        "window_metrics": window_metrics,
        "proposal_telemetry": proposal_telemetry,
        "historical_events": HISTORICAL_EVENTS,
        "horizon_snapshots": key_horizons,
        "forecast": forecast_results,
        "forecast_24h": full_24h,
    }


static_dirs = ["static", "borssele-digital-twin/backend/static"]
for sdir in static_dirs:
    if os.path.exists(sdir) and os.path.isdir(sdir):
        app.mount("/", StaticFiles(directory=sdir, html=True), name="static")
        break
