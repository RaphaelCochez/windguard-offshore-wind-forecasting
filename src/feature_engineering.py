"""
Feature Engineering & Offshore Wind Park Operability Regime Classification
Converts raw ERA5 horizontal wind vectors (u100, v100) into 100m hub-height scalar wind speeds,
hourly wind ramp rates, and operational shut-down regime labels (strictly no power-delivery curves).
"""
import xarray as xr
import numpy as np
import pandas as pd

LOW_WIND_THRESHOLD_MS = 3.0
CAUTION_THRESHOLD_MS = 20.0
CUT_OUT_THRESHOLD_MS = 25.0


def classify_wind_regime(ws: float) -> str:
    if ws >= CUT_OUT_THRESHOLD_MS:
        return "CUT_OUT_SHUTDOWN"
    if ws >= CAUTION_THRESHOLD_MS:
        return "HIGH_WIND_CAUTION"
    if ws < LOW_WIND_THRESHOLD_MS:
        return "LOW_WIND_STANDBY"
    return "GOOD_CONDITIONS"


def process_storm_features(nc_filepath: str) -> pd.DataFrame:
    ds = xr.open_dataset(nc_filepath)

    # 1. Total 100m Hub-Height Wind Speed: W_100 = sqrt(u100^2 + v100^2)
    u100 = ds["u100"]
    v100 = ds["v100"]
    w100 = np.sqrt(u100**2 + v100**2)

    # 2. Convert to DataFrame
    df = w100.to_dataframe(name="wind_speed_100m").reset_index()

    # 3. Compute 1h Wind Ramp Rate (delta_speed / delta_t)
    df["ramp_rate_1h"] = df.groupby(["latitude", "longitude"])["wind_speed_100m"].diff(1)

    # 4. Target Label: High-Wind Cut-Out Shut-Down Event (True if Speed >= 25.0 m/s)
    df["cut_out_event"] = (df["wind_speed_100m"] >= CUT_OUT_THRESHOLD_MS).astype(int)
    df["can_operate"] = (df["wind_speed_100m"] < CUT_OUT_THRESHOLD_MS).astype(int)
    df["operational_regime"] = df["wind_speed_100m"].apply(classify_wind_regime)

    return df
