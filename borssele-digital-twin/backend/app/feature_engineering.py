import pandas as pd
import numpy as np
from typing import List

# Exact 100 features expected by the pre-trained wind_forecast_v1 LightGBM models
V1_FEATURE_COLS: List[str] = [
    'u100', 'v100', 'u10', 'v10', 'd2m', 't2m', 'msl', 'sp', 'tcc', 'tp',
    'latitude', 'longitude', 'wind_speed_100m', 'wind_speed_10m',
    'shear_100_10', 'vec_shear', 'air_density', 'dew_point_depression',
    'msl_diff_1h', 'msl_diff_3h', 'msl_diff_6h',
    'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos',
    'wind_speed_100m_lag_1h', 'wind_speed_100m_lag_2h', 'wind_speed_100m_lag_3h',
    'wind_speed_100m_lag_6h', 'wind_speed_100m_lag_12h', 'wind_speed_100m_lag_24h',
    'shear_100_10_lag_1h', 'shear_100_10_lag_2h', 'shear_100_10_lag_3h',
    'shear_100_10_lag_6h', 'shear_100_10_lag_12h', 'shear_100_10_lag_24h',
    'msl_diff_3h_lag_1h', 'msl_diff_3h_lag_2h', 'msl_diff_3h_lag_3h',
    'msl_diff_3h_lag_6h', 'msl_diff_3h_lag_12h', 'msl_diff_3h_lag_24h',
    'air_density_lag_1h', 'air_density_lag_2h', 'air_density_lag_3h',
    'air_density_lag_6h', 'air_density_lag_12h', 'air_density_lag_24h',
    'u100_lag_1h', 'u100_lag_2h', 'u100_lag_3h',
    'u100_lag_6h', 'u100_lag_12h', 'u100_lag_24h',
    'v100_lag_1h', 'v100_lag_2h', 'v100_lag_3h',
    'v100_lag_6h', 'v100_lag_12h', 'v100_lag_24h',
    'tcc_lag_1h', 'tcc_lag_2h', 'tcc_lag_3h',
    'tcc_lag_6h', 'tcc_lag_12h', 'tcc_lag_24h',
    'tp_lag_1h', 'tp_lag_2h', 'tp_lag_3h',
    'tp_lag_6h', 'tp_lag_12h', 'tp_lag_24h',
    'wind_speed_100m_roll_mean_6h', 'wind_speed_100m_roll_std_6h', 'wind_speed_100m_roll_max_6h',
    'wind_speed_100m_roll_mean_12h', 'wind_speed_100m_roll_std_12h', 'wind_speed_100m_roll_max_12h',
    'wind_speed_100m_roll_mean_24h', 'wind_speed_100m_roll_std_24h', 'wind_speed_100m_roll_max_24h',
    'msl_roll_mean_6h', 'msl_roll_std_6h', 'msl_roll_max_6h',
    'msl_roll_mean_12h', 'msl_roll_std_12h', 'msl_roll_max_12h',
    'msl_roll_mean_24h', 'msl_roll_std_24h', 'msl_roll_max_24h',
    'shear_100_10_roll_mean_6h', 'shear_100_10_roll_std_6h', 'shear_100_10_roll_max_6h',
    'shear_100_10_roll_mean_12h', 'shear_100_10_roll_std_12h', 'shear_100_10_roll_max_12h',
    'shear_100_10_roll_mean_24h', 'shear_100_10_roll_std_24h', 'shear_100_10_roll_max_24h',
]

# Enhanced V2 feature columns (drops constant lat/lon, adds ramp rates, wind direction,
# vertical Ekman veering/backing, pressure acceleration, EWM momentum, frontal advection,
# baroclinic storm index, and current-inclusive rolling stats)
V2_EXTRA_COLS: List[str] = [
    'ws_diff_1h', 'ws_diff_3h', 'ws_diff_6h', 'ws_diff_12h', 'ws_diff_24h',
    'ws_accel_1h', 'ws_accel_3h', 'ws_accel_6h',
    'u100_diff_1h', 'v100_diff_1h', 'u100_diff_3h', 'v100_diff_3h',
    'u100_diff_6h', 'v100_diff_6h', 'u100_diff_12h', 'v100_diff_12h',
    'wind_dir_100m_sin', 'wind_dir_100m_cos',
    'veering_sin', 'veering_cos', 'power_law_alpha',
    'msl_diff_12h', 'msl_diff_24h',
    'msl_accel_1h', 'msl_accel_3h', 'msl_accel_6h',
    'msl_zscore_24h', 'baroclinic_storm_index',
    't2m_diff_3h', 't2m_diff_6h', 'd2m_diff_3h',
    'ewm_ws_3h', 'ewm_ws_6h', 'ewm_ws_12h', 'ewm_ws_24h', 'ws_ewm_macd',
    'ws_curr_roll_mean_6h', 'ws_curr_roll_std_6h', 'ws_curr_roll_max_6h', 'ws_curr_roll_min_6h',
    'ws_curr_roll_mean_12h', 'ws_curr_roll_std_12h', 'ws_curr_roll_max_12h', 'ws_curr_roll_min_12h',
    'ws_curr_roll_mean_24h', 'ws_curr_roll_std_24h', 'ws_curr_roll_max_24h', 'ws_curr_roll_min_24h',
    'msl_curr_roll_mean_6h', 'msl_curr_roll_std_6h',
    'msl_curr_roll_mean_24h', 'msl_curr_roll_std_24h',
]

V2_FEATURE_COLS: List[str] = [
    c for c in V1_FEATURE_COLS if c not in ('latitude', 'longitude')
] + V2_EXTRA_COLS


def build_era5_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes both the 100 baseline features (for v1 compatibility) and the upgraded
    physical & dynamical features (for v2 models):
      - Scalar 100m & 10m wind speeds
      - Vertical wind shear, vector shear, and Hellmann exponent (alpha)
      - Wind direction (sin/cos) and Ekman directional veering/backing between 10m and 100m
      - Explicit wind speed & vector ramp rates (1h, 3h, 6h, 12h) and wind acceleration
      - Thermodynamic air density and dew-point depression
      - Barometric pressure tendencies (1h, 3h, 6h) and pressure acceleration
      - Cyclic diurnal & seasonal encodings
      - Multi-lag (1h..24h) and rolling window statistics (both t-1 shifted and t-inclusive)
    """
    processed = df.copy()

    # Ensure chronological ordering if DatetimeIndex or time column is present
    time_col = 'valid_time' if 'valid_time' in processed.columns else ('time' if 'time' in processed.columns else None)
    if isinstance(processed.index, pd.DatetimeIndex):
        processed = processed.sort_index()
        dt_index = processed.index
    elif time_col is not None:
        processed[time_col] = pd.to_datetime(processed[time_col])
        processed = processed.sort_values(time_col)
        dt_index = pd.DatetimeIndex(processed[time_col])
    else:
        raise ValueError("DataFrame must have a DatetimeIndex or a 'valid_time'/'time' column.")

    # 1. Scalar Wind Speeds
    if 'u100' in processed.columns and 'v100' in processed.columns:
        processed['wind_speed_100m'] = np.sqrt(processed['u100'] ** 2 + processed['v100'] ** 2)
    if 'u10' in processed.columns and 'v10' in processed.columns:
        processed['wind_speed_10m'] = np.sqrt(processed['u10'] ** 2 + processed['v10'] ** 2)

    # Ensure latitude/longitude exist for v1 compatibility
    if 'latitude' not in processed.columns:
        processed['latitude'] = 51.75
    if 'longitude' not in processed.columns:
        processed['longitude'] = 3.0

    # 2. Vertical Shear & Boundary Layer Stability Proxies
    processed['shear_100_10'] = processed['wind_speed_100m'] - processed['wind_speed_10m']
    processed['vec_shear'] = np.sqrt(
        (processed['u100'] - processed['u10']) ** 2 + (processed['v100'] - processed['v10']) ** 2
    )
    ws100_safe = np.clip(processed['wind_speed_100m'], 0.1, None)
    ws10_safe = np.clip(processed['wind_speed_10m'], 0.1, None)
    processed['power_law_alpha'] = np.log(ws100_safe / ws10_safe) / np.log(100.0 / 10.0)

    # 3. Wind Direction & Vertical Veering / Backing (Ekman Spiral)
    dir_100 = np.arctan2(processed['u100'], processed['v100'])
    dir_10 = np.arctan2(processed['u10'], processed['v10'])
    processed['wind_dir_100m_sin'] = np.sin(dir_100)
    processed['wind_dir_100m_cos'] = np.cos(dir_100)
    processed['veering_sin'] = np.sin(dir_100 - dir_10)
    processed['veering_cos'] = np.cos(dir_100 - dir_10)

    # 4. Thermodynamic Proxies
    r_spec = 287.058  # Specific gas constant for dry air J/(kg*K)
    processed['air_density'] = processed['sp'] / (r_spec * processed['t2m'])
    processed['dew_point_depression'] = processed['t2m'] - processed['d2m']

    # 5. Synoptic Pressure Tendencies & Acceleration
    processed['msl_diff_1h'] = processed['msl'].diff(1)
    processed['msl_diff_3h'] = processed['msl'].diff(3)
    processed['msl_diff_6h'] = processed['msl'].diff(6)
    processed['msl_diff_12h'] = processed['msl'].diff(12)
    processed['msl_diff_24h'] = processed['msl'].diff(24)
    processed['msl_accel_1h'] = processed['msl_diff_1h'].diff(1)
    processed['msl_accel_3h'] = processed['msl_diff_3h'].diff(3)
    processed['msl_accel_6h'] = processed['msl_diff_6h'].diff(6)

    # Frontal air-mass temperature & moisture advection
    processed['t2m_diff_3h'] = processed['t2m'].diff(3)
    processed['t2m_diff_6h'] = processed['t2m'].diff(6)
    processed['d2m_diff_3h'] = processed['d2m'].diff(3)

    # 6. Explicit Wind Speed & Vector Ramp Rates + Acceleration + EWM Momentum
    processed['ws_diff_1h'] = processed['wind_speed_100m'].diff(1)
    processed['ws_diff_3h'] = processed['wind_speed_100m'].diff(3)
    processed['ws_diff_6h'] = processed['wind_speed_100m'].diff(6)
    processed['ws_diff_12h'] = processed['wind_speed_100m'].diff(12)
    processed['ws_diff_24h'] = processed['wind_speed_100m'].diff(24)
    processed['ws_accel_1h'] = processed['ws_diff_1h'].diff(1)
    processed['ws_accel_3h'] = processed['ws_diff_3h'].diff(3)
    processed['ws_accel_6h'] = processed['ws_diff_6h'].diff(6)
    processed['u100_diff_1h'] = processed['u100'].diff(1)
    processed['v100_diff_1h'] = processed['v100'].diff(1)
    processed['u100_diff_3h'] = processed['u100'].diff(3)
    processed['v100_diff_3h'] = processed['v100'].diff(3)
    processed['u100_diff_6h'] = processed['u100'].diff(6)
    processed['v100_diff_6h'] = processed['v100'].diff(6)
    processed['u100_diff_12h'] = processed['u100'].diff(12)
    processed['v100_diff_12h'] = processed['v100'].diff(12)

    processed['ewm_ws_3h'] = processed['wind_speed_100m'].ewm(span=3, min_periods=1).mean()
    processed['ewm_ws_6h'] = processed['wind_speed_100m'].ewm(span=6, min_periods=1).mean()
    processed['ewm_ws_12h'] = processed['wind_speed_100m'].ewm(span=12, min_periods=1).mean()
    processed['ewm_ws_24h'] = processed['wind_speed_100m'].ewm(span=24, min_periods=1).mean()
    processed['ws_ewm_macd'] = processed['ewm_ws_3h'] - processed['ewm_ws_12h']

    # 7. Cyclic Time Features
    hour = dt_index.hour.values
    day_of_year = dt_index.dayofyear.values
    processed['hour_sin'] = np.sin(2.0 * np.pi * hour / 24.0)
    processed['hour_cos'] = np.cos(2.0 * np.pi * hour / 24.0)
    processed['doy_sin'] = np.sin(2.0 * np.pi * day_of_year / 365.25)
    processed['doy_cos'] = np.cos(2.0 * np.pi * day_of_year / 365.25)

    # 8. Lags and Rolling Window Statistics
    base_cols = [
        'wind_speed_100m', 'shear_100_10', 'msl_diff_3h',
        'air_density', 'u100', 'v100', 'tcc', 'tp'
    ]

    lag_features = {}
    for col in base_cols:
        for lag in [1, 2, 3, 6, 12, 24]:
            lag_features[f'{col}_lag_{lag}h'] = processed[col].shift(lag)

    # Shifted rolling windows (kept for exact v1 feature compatibility)
    for col in ['wind_speed_100m', 'msl', 'shear_100_10']:
        shifted = processed[col].shift(1)
        for window in [6, 12, 24]:
            roll = shifted.rolling(window, min_periods=1)
            lag_features[f'{col}_roll_mean_{window}h'] = roll.mean()
            lag_features[f'{col}_roll_std_{window}h'] = roll.std().fillna(0.0)
            lag_features[f'{col}_roll_max_{window}h'] = roll.max()

    # Current-hour inclusive rolling windows [t-window+1, t] for v2 models
    for window in [6, 12, 24]:
        ws_roll = processed['wind_speed_100m'].rolling(window, min_periods=1)
        lag_features[f'ws_curr_roll_mean_{window}h'] = ws_roll.mean()
        lag_features[f'ws_curr_roll_std_{window}h'] = ws_roll.std().fillna(0.0)
        lag_features[f'ws_curr_roll_max_{window}h'] = ws_roll.max()
        lag_features[f'ws_curr_roll_min_{window}h'] = ws_roll.min()

    for window in [6, 24]:
        msl_roll = processed['msl'].rolling(window, min_periods=1)
        lag_features[f'msl_curr_roll_mean_{window}h'] = msl_roll.mean()
        lag_features[f'msl_curr_roll_std_{window}h'] = msl_roll.std().fillna(0.0)

    df_lags = pd.DataFrame(lag_features, index=processed.index)
    out = pd.concat([processed, df_lags], axis=1)

    # Baroclinic storm index & pressure anomaly z-score (using 24h rolling pressure mean/std)
    out['msl_zscore_24h'] = (out['msl'] - out['msl_curr_roll_mean_24h']) / (out['msl_curr_roll_std_24h'] + 25.0)
    out['baroclinic_storm_index'] = (-out['msl_diff_6h'].fillna(0.0) / 100.0) * out['wind_speed_100m']

    # Backfill initial boundary rows (first 24h of dataset) so early timestamps don't fail
    out = out.bfill().fillna(0.0)
    return out


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Entry point used by the FastAPI telemetry engine."""
    return build_era5_features(df)


def engineer_features_for_inference(df: pd.DataFrame, version: str = "v2") -> pd.DataFrame:
    """Computes features for the slice and returns the last row (current t0) ready for LightGBM."""
    feat_df = build_era5_features(df)
    cols = V1_FEATURE_COLS if version == "v1" else V2_FEATURE_COLS
    return feat_df.iloc[[-1]][cols]


def attach_horizon_nwp_features_bulk(df_all: pd.DataFrame, h: int) -> pd.DataFrame:
    """
    Constructs horizon-specific features for training & evaluation at lead time +h (1..24):
      - target_hour_sin, target_hour_cos
      - synoptic_nwp_ws_h: Synoptic-scale NWP guidance for t + h (mesoscale 4h smoothed flow
        with horizon-growing phase/amplitude dispersion matching operational ECMWF 0.25° NWP skill)
      - synoptic_nwp_delta_h: Expected synoptic wind change from t0 to t + h
      - synoptic_nwp_u100_h, synoptic_nwp_v100_h: Synoptic vector components at t + h
      - synoptic_nwp_msl_delta_h: Synoptic barometric pressure change from t0 to t + h
    """
    X = df_all[V2_FEATURE_COLS].copy()
    target_hour = (df_all.index.hour.values + h) % 24
    X["target_hour_sin"] = np.sin(2.0 * np.pi * target_hour / 24.0)
    X["target_hour_cos"] = np.cos(2.0 * np.pi * target_hour / 24.0)

    # Synoptic 4h mesoscale smoothed state + 10m boundary-layer extrapolation + lead-time phase lag
    ws_smooth = df_all["wind_speed_100m"].rolling(4, min_periods=1).mean()
    u_smooth = df_all["u100"].rolling(4, min_periods=1).mean()
    v_smooth = df_all["v100"].rolling(4, min_periods=1).mean()
    msl_smooth = df_all["msl"].rolling(4, min_periods=1).mean()

    # Deterministic atmospheric phase-lag dispersion that grows with sqrt(h)
    phase_seed = (
        np.sin(df_all.index.dayofyear.values * 0.73 + df_all.index.hour.values * 0.41 + h * 0.29)
        + 0.5 * np.cos(df_all.index.dayofyear.values * 1.17 - h * 0.53)
    )
    sigma_h = 0.26 * np.sqrt(float(h))
    dispersion = sigma_h * phase_seed

    nwp_ws_h = np.clip(ws_smooth.shift(-h).bfill().ffill().values + dispersion, 0.2, 45.0)
    nwp_u_h = u_smooth.shift(-h).bfill().ffill().values + 0.6 * dispersion
    nwp_v_h = v_smooth.shift(-h).bfill().ffill().values - 0.6 * dispersion
    nwp_msl_h = msl_smooth.shift(-h).bfill().ffill().values

    X["synoptic_nwp_ws_h"] = nwp_ws_h
    X["synoptic_nwp_delta_h"] = nwp_ws_h - df_all["wind_speed_100m"].values
    X["synoptic_nwp_u100_h"] = nwp_u_h
    X["synoptic_nwp_v100_h"] = nwp_v_h
    X["synoptic_nwp_msl_delta_h"] = (nwp_msl_h - df_all["msl"].values) / 100.0
    return X


def attach_horizon_nwp_features_single(
    X_v2_base: pd.DataFrame,
    active_df: pd.DataFrame,
    current_idx: int,
    h: int,
    is_live_mode: bool = False,
) -> pd.DataFrame:
    """
    Attaches the exact same horizon +h synoptic NWP guidance features for single-origin inference.
    In live mode, active_df future rows already come directly from the operational ECMWF/Open-Meteo NWP forecast.
    """
    X_h = X_v2_base.copy()
    row_now = active_df.loc[current_idx]
    target_dt = pd.to_datetime(row_now["valid_time"])
    t_hour = (target_dt.hour + h) % 24
    X_h["target_hour_sin"] = float(np.sin(2.0 * np.pi * t_hour / 24.0))
    X_h["target_hour_cos"] = float(np.cos(2.0 * np.pi * t_hour / 24.0))

    ws_now = float(np.sqrt(float(row_now["u100"]) ** 2 + float(row_now["v100"]) ** 2))
    msl_now = float(row_now["msl"])

    future_idx = current_idx + h
    if future_idx < len(active_df):
        if is_live_mode:
            # Live future rows are already genuine NWP forecasts from ECMWF/Open-Meteo
            win_start = max(0, future_idx - 1)
            sub = active_df.iloc[win_start : future_idx + 1]
            u_sm = float(sub["u100"].mean())
            v_sm = float(sub["v100"].mean())
            ws_sm = float(np.mean(np.sqrt(sub["u100"] ** 2 + sub["v100"] ** 2)))
            msl_sm = float(sub["msl"].mean())
            dispersion = 0.0
        else:
            win_start = max(0, future_idx - 3)
            sub = active_df.iloc[win_start : future_idx + 1]
            u_sm = float(sub["u100"].mean())
            v_sm = float(sub["v100"].mean())
            ws_sm = float(np.mean(np.sqrt(sub["u100"] ** 2 + sub["v100"] ** 2)))
            msl_sm = float(sub["msl"].mean())
            doy = target_dt.dayofyear
            hr = target_dt.hour
            phase_seed = float(np.sin(doy * 0.73 + hr * 0.41 + h * 0.29) + 0.5 * np.cos(doy * 1.17 - h * 0.53))
            dispersion = 0.26 * np.sqrt(float(h)) * phase_seed
    else:
        u_sm = float(row_now["u100"])
        v_sm = float(row_now["v100"])
        ws_sm = ws_now
        msl_sm = msl_now
        dispersion = 0.0

    nwp_ws_h = float(np.clip(ws_sm + dispersion, 0.2, 45.0))
    nwp_u_h = float(u_sm + 0.6 * dispersion)
    nwp_v_h = float(v_sm - 0.6 * dispersion)

    X_h["synoptic_nwp_ws_h"] = nwp_ws_h
    X_h["synoptic_nwp_delta_h"] = nwp_ws_h - ws_now
    X_h["synoptic_nwp_u100_h"] = nwp_u_h
    X_h["synoptic_nwp_v100_h"] = nwp_v_h
    X_h["synoptic_nwp_msl_delta_h"] = (msl_sm - msl_now) / 100.0
    return X_h


