import os
import time
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd

from app.config import POWER_LAW_OFFSHORE_ALPHA

# Supported Offshore Wind Areas with exact ERA5 0.25° x 0.25° Grid Cells & Polygons
OFFSHORE_AREAS: Dict[str, Dict[str, Any]] = {
    "borssele": {
        "id": "borssele",
        "name": "Borssele Offshore Wind Zone (NL)",
        "region": "Southern North Sea • Netherlands",
        "lat": 51.716667,
        "lon": 3.000000,
        "grid_center": [51.750, 3.000],
        "era5_box": {
            "north": 51.875,
            "south": 51.625,
            "west": 2.875,
            "east": 3.125,
            "area_km2": 478,
        },
        "zone_area_km2": 344,
        "default_zoom": 9,
        "map_center": [51.68, 3.18],
        "zone_polygon": [
            [51.792, 2.915],
            [51.795, 3.085],
            [51.710, 3.112],
            [51.632, 3.058],
            [51.628, 2.942],
            [51.695, 2.888],
        ],
        "landmarks": [
            {"name": "Westkapelle (NL Coast)", "lat": 51.528, "lon": 3.440},
            {"name": "Vlissingen Port", "lat": 51.442, "lon": 3.573},
            {"name": "Burgh-Haamstede", "lat": 51.696, "lon": 3.712},
            {"name": "Zeebrugge (BE)", "lat": 51.331, "lon": 3.207},
        ],
    },
    "tahkoluoto": {
        "id": "tahkoluoto",
        "name": "Tahkoluoto Offshore Wind Farm (FI)",
        "region": "Bothnian Sea • Pori, Finland",
        "lat": 61.626111,
        "lon": 21.358056,
        "grid_center": [61.625, 21.375],
        "era5_box": {
            "north": 61.750,
            "south": 61.500,
            "west": 21.250,
            "east": 21.500,
            "area_km2": 368,
        },
        "zone_area_km2": 145,
        "default_zoom": 9,
        "map_center": [61.62, 21.45],
        "zone_polygon": [
            [61.668, 21.295],
            [61.672, 21.410],
            [61.615, 21.430],
            [61.582, 21.390],
            [61.585, 21.300],
        ],
        "landmarks": [
            {"name": "Tahkoluoto Port", "lat": 61.632, "lon": 21.396},
            {"name": "Reposaari Island", "lat": 61.608, "lon": 21.445},
            {"name": "Mäntyluoto (Pori)", "lat": 61.592, "lon": 21.486},
            {"name": "Kallo Lighthouse", "lat": 61.594, "lon": 21.462},
        ],
    },
    "hollandse_kust": {
        "id": "hollandse_kust",
        "name": "Hollandse Kust Zuid & Noord (NL)",
        "region": "Central Dutch Coast • North Sea",
        "lat": 52.300000,
        "lon": 4.100000,
        "grid_center": [52.250, 4.000],
        "era5_box": {
            "north": 52.375,
            "south": 52.125,
            "west": 3.875,
            "east": 4.125,
            "area_km2": 472,
        },
        "zone_area_km2": 356,
        "default_zoom": 9,
        "map_center": [52.26, 4.22],
        "zone_polygon": [
            [52.362, 3.940],
            [52.365, 4.110],
            [52.255, 4.118],
            [52.155, 4.045],
            [52.160, 3.915],
        ],
        "landmarks": [
            {"name": "IJmuiden Offshore Port", "lat": 52.461, "lon": 4.555},
            {"name": "Zandvoort Coast", "lat": 52.371, "lon": 4.527},
            {"name": "Noordwijk", "lat": 52.245, "lon": 4.429},
            {"name": "Scheveningen Port", "lat": 52.102, "lon": 4.265},
        ],
    },
    "dogger_bank": {
        "id": "dogger_bank",
        "name": "Dogger Bank Offshore Wind Zone (UK)",
        "region": "Central North Sea • UK Sector",
        "lat": 54.750000,
        "lon": 2.250000,
        "grid_center": [54.750, 2.250],
        "era5_box": {
            "north": 54.875,
            "south": 54.625,
            "west": 2.125,
            "east": 2.375,
            "area_km2": 446,
        },
        "zone_area_km2": 515,
        "default_zoom": 8,
        "map_center": [54.75, 2.25],
        "zone_polygon": [
            [54.845, 2.150],
            [54.850, 2.355],
            [54.725, 2.365],
            [54.645, 2.310],
            [54.648, 2.145],
        ],
        "landmarks": [
            {"name": "Dogger Bank Alpha Platform", "lat": 54.780, "lon": 2.210},
            {"name": "Dogger Bank Beta Sector", "lat": 54.710, "lon": 2.290},
        ],
    },
    "hornsea": {
        "id": "hornsea",
        "name": "Hornsea Offshore Wind Zone (UK)",
        "region": "Yorkshire Coast • UK North Sea",
        "lat": 53.880000,
        "lon": 1.800000,
        "grid_center": [53.875, 1.750],
        "era5_box": {
            "north": 54.000,
            "south": 53.750,
            "west": 1.625,
            "east": 1.875,
            "area_km2": 455,
        },
        "zone_area_km2": 407,
        "default_zoom": 8,
        "map_center": [53.85, 1.45],
        "zone_polygon": [
            [53.965, 1.660],
            [53.970, 1.860],
            [53.835, 1.868],
            [53.775, 1.810],
            [53.780, 1.655],
        ],
        "landmarks": [
            {"name": "Flamborough Head (UK)", "lat": 54.116, "lon": -0.082},
            {"name": "Spurn Point", "lat": 53.578, "lon": 0.111},
            {"name": "Grimsby O&M Hub", "lat": 53.575, "lon": -0.075},
        ],
    },
}

# In-memory caches of fetched live and historical ERA5 dataframes
_LIVE_CACHE: Dict[str, Dict[str, Any]] = {}
_HIST_WINDOW_CACHE: Dict[str, pd.DataFrame] = {}
CACHE_TTL_SECONDS = 300


def _wind_speed_dir_to_uv(speed_ms: np.ndarray, direction_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert meteorological wind speed & direction (degrees FROM) into ERA5 u, v components (m/s)."""
    rad = np.deg2rad(direction_deg)
    u = -speed_ms * np.sin(rad)
    v = -speed_ms * np.cos(rad)
    return u, v


def _try_copernicus_cds_sync(area: Dict[str, Any], target_nc_path: str, cds_api_key: Optional[str] = None) -> bool:
    """
    Optional direct Copernicus Climate Data Store (CDS) fetch using ecmwf-datastores-client
    when CDS credentials are configured.
    """
    rc_path = os.path.expanduser("~/.ecmwfdatastoresrc")
    api_key = cds_api_key or os.environ.get("CDS_API_KEY")
    if not api_key and not os.path.exists(rc_path):
        return False

    try:
        from ecmwf.datastores import Client  # type: ignore

        client = (
            Client(url="https://cds.climate.copernicus.eu/api", key=api_key)
            if api_key
            else Client()
        )
        end_dt = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=5)
        start_dt = end_dt - timedelta(days=14)
        request = {
            "variable": [
                "100m_u_component_of_wind",
                "100m_v_component_of_wind",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "mean_sea_level_pressure",
                "surface_pressure",
                "2m_temperature",
                "2m_dewpoint_temperature",
                "instantaneous_10m_wind_gust",
                "total_cloud_cover",
                "total_precipitation",
            ],
            "date": [start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")],
            "location": {"longitude": float(area["lon"]), "latitude": float(area["lat"])},
            "data_format": "netcdf",
        }
        client.retrieve("reanalysis-era5-single-levels-timeseries", request, target=target_nc_path)
        return True
    except Exception as exc:
        print(f"[ERA5-Updater] Copernicus CDS direct fetch skipped/fallback: {exc}")
        return False


def _fetch_open_meteo_primary(area: Dict[str, Any], utc_now_hour: datetime) -> pd.DataFrame:
    """Fetches primary Open-Meteo ECMWF/ERA5-Seamless hourly telemetry aligned strictly by UTC timestamp."""
    hourly_vars = ",".join([
        "wind_speed_100m",
        "wind_direction_100m",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
        "temperature_2m",
        "dew_point_2m",
        "pressure_msl",
        "surface_pressure",
        "cloud_cover",
        "precipitation",
    ])
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={area['lat']}&longitude={area['lon']}"
        f"&hourly={hourly_vars}"
        f"&wind_speed_unit=ms"
        f"&past_days=5&forecast_days=2"
        f"&timezone=UTC"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "WindGuard-ERA5-Updater/2.0"})
    with urllib.request.urlopen(req, timeout=6) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    hourly = payload.get("hourly", {})
    raw_times = pd.to_datetime(hourly.get("time", []))
    if len(raw_times) < 72:
        raise ValueError(f"Expected >=72 hourly rows from Open-Meteo API, got {len(raw_times)}")

    ws100 = np.asarray(hourly["wind_speed_100m"], dtype=float)
    wd100 = np.asarray(hourly["wind_direction_100m"], dtype=float)
    ws10 = np.asarray(hourly["wind_speed_10m"], dtype=float)
    wd10 = np.asarray(hourly["wind_direction_10m"], dtype=float)
    gust10 = np.asarray(hourly["wind_gusts_10m"], dtype=float)
    temp_c = np.asarray(hourly["temperature_2m"], dtype=float)
    dew_c = np.asarray(hourly["dew_point_2m"], dtype=float)
    msl_hpa = np.asarray(hourly["pressure_msl"], dtype=float)
    sp_hpa = np.asarray(hourly["surface_pressure"], dtype=float)
    cloud_pct = np.asarray(hourly["cloud_cover"], dtype=float)
    precip_mm = np.asarray(hourly["precipitation"], dtype=float)

    u100, v100 = _wind_speed_dir_to_uv(ws100, wd100)
    u10, v10 = _wind_speed_dir_to_uv(ws10, wd10)

    df = pd.DataFrame({
        "valid_time": raw_times,
        "u100": u100,
        "v100": v100,
        "u10": u10,
        "v10": v10,
        "d2m": dew_c + 273.15,
        "t2m": temp_c + 273.15,
        "msl": msl_hpa * 100.0,
        "sp": sp_hpa * 100.0,
        "tcc": np.clip(cloud_pct / 100.0, 0.0, 1.0),
        "tp": precip_mm / 1000.0,
        "i10fg": gust10,
        "latitude": float(area["grid_center"][0]),
        "longitude": float(area["grid_center"][1]),
    })
    df = df.ffill().bfill()

    # Slice [utc_now_hour - 96h, utc_now_hour + 24h] strictly in UTC
    min_dt = pd.Timestamp(utc_now_hour) - pd.Timedelta(hours=96)
    max_dt = pd.Timestamp(utc_now_hour) + pd.Timedelta(hours=24)
    df = df[(df["valid_time"] >= min_dt) & (df["valid_time"] <= max_dt)].reset_index(drop=True)
    if len(df) < 72:
        raise ValueError(f"Insufficient UTC-aligned primary rows ({len(df)}) around {utc_now_hour}")
    return df


def _fetch_met_norway_stream(lat: float, lon: float) -> Dict[pd.Timestamp, Dict[str, float]]:
    """Fetches independent NWP forecast from the Norwegian Meteorological Institute / Yr.no API."""
    url = f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={lat:.4f}&lon={lon:.4f}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "WindGuard-OffshoreForecast/2.0 (github.com/windguard-offshore-wind-forecasting)"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    scale_10m_to_100m = 10.0 ** POWER_LAW_OFFSHORE_ALPHA  # (100/10)^0.11 ~= 1.288
    out: Dict[pd.Timestamp, Dict[str, float]] = {}
    for entry in payload.get("properties", {}).get("timeseries", []):
        t_utc = pd.to_datetime(entry["time"]).tz_localize(None)
        details = entry.get("data", {}).get("instant", {}).get("details", {})
        ws10 = details.get("wind_speed")
        if ws10 is None:
            continue
        out[t_utc] = {
            "ws100": float(ws10) * scale_10m_to_100m,
            "wd100": float(details.get("wind_from_direction", 240.0)),
            "msl_pa": float(details.get("air_pressure_at_sea_level", 1013.25)) * 100.0,
        }
    return out


def _fetch_dwd_icon_stream(lat: float, lon: float) -> Dict[pd.Timestamp, Dict[str, float]]:
    """Fetches independent DWD ICON-Seamless 100m NWP forecast via Open-Meteo."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=wind_speed_100m,wind_direction_100m,pressure_msl"
        f"&models=icon_seamless&wind_speed_unit=ms&past_days=1&forecast_days=2&timezone=UTC"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "WindGuard-ERA5-Updater/2.0"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    hourly = payload.get("hourly", {})
    times = pd.to_datetime(hourly.get("time", []))
    ws_arr = hourly.get("wind_speed_100m", [])
    wd_arr = hourly.get("wind_direction_100m", [])
    msl_arr = hourly.get("pressure_msl", [])
    out: Dict[pd.Timestamp, Dict[str, float]] = {}
    for idx, t_utc in enumerate(times):
        ws_val = ws_arr[idx] if idx < len(ws_arr) else None
        if ws_val is None:
            continue
        wd_val = wd_arr[idx] if idx < len(wd_arr) and wd_arr[idx] is not None else 240.0
        msl_val = msl_arr[idx] if idx < len(msl_arr) and msl_arr[idx] is not None else 1013.25
        out[t_utc] = {
            "ws100": float(ws_val),
            "wd100": float(wd_val),
            "msl_pa": float(msl_val) * 100.0,
        }
    return out


def _fetch_secondary_nwp_streams(
    area: Dict[str, Any],
    skip_open_meteo: bool = False,
) -> Tuple[Dict[pd.Timestamp, Dict[str, float]], List[str]]:
    """
    Concurrently polls genuinely independent secondary NWP sources (MET Norway / Yr.no and DWD ICON-Seamless)
    and merges their 100m wind speed & sea-level pressure maps by exact UTC timestamp.
    """
    lat, lon = float(area["lat"]), float(area["lon"])
    metno_map: Dict[pd.Timestamp, Dict[str, float]] = {}
    icon_map: Dict[pd.Timestamp, Dict[str, float]] = {}
    sources: List[str] = []

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_mn = pool.submit(_fetch_met_norway_stream, lat, lon)
        fut_ic = None if skip_open_meteo else pool.submit(_fetch_dwd_icon_stream, lat, lon)

        try:
            metno_map = fut_mn.result(timeout=6)
            if metno_map:
                sources.append("MET Norway / Yr.no")
        except Exception as exc:
            print(f"[ERA5-Updater] MET Norway stream skipped: {exc}")

        if fut_ic is not None:
            try:
                icon_map = fut_ic.result(timeout=6)
                if icon_map:
                    sources.append("DWD ICON-Seamless")
            except Exception as exc:
                print(f"[ERA5-Updater] DWD ICON stream skipped: {exc}")

    merged: Dict[pd.Timestamp, Dict[str, float]] = {}
    all_ts = sorted(set(metno_map.keys()) | set(icon_map.keys()))
    for ts in all_ts:
        ws_vals, wd_vals, msl_vals = [], [], []
        if ts in icon_map:
            ws_vals.append(icon_map[ts]["ws100"])
            wd_vals.append(icon_map[ts]["wd100"])
            msl_vals.append(icon_map[ts]["msl_pa"])
        if ts in metno_map:
            ws_vals.append(metno_map[ts]["ws100"])
            wd_vals.append(metno_map[ts]["wd100"])
            msl_vals.append(metno_map[ts]["msl_pa"])
        if ws_vals:
            merged[ts] = {
                "ws100": float(np.mean(ws_vals)),
                "wd100": float(wd_vals[0]),
                "msl_pa": float(np.mean(msl_vals)),
            }
    return merged, sources


def _attach_secondary_columns(
    df: pd.DataFrame,
    secondary_map: Dict[pd.Timestamp, Dict[str, float]],
) -> pd.DataFrame:
    """Attaches independent secondary NWP 100m wind speed aligned by UTC timestamp."""
    if not secondary_map:
        df["secondary_nwp_ws100"] = np.nan
        return df
    df["secondary_nwp_ws100"] = df["valid_time"].map(
        lambda ts: secondary_map.get(ts, {}).get("ws100", np.nan)
    )
    return df


def fetch_updated_era5_for_area(
    area_id: str = "borssele",
    data_dir: str = "data",
    force_refresh: bool = False,
    cds_api_key: Optional[str] = None,
    fallback_df: Optional[pd.DataFrame] = None,
    simulate_primary_outage: bool = False,
) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """
    Fetches UTC-aligned hourly ECMWF/ERA5-Seamless telemetry plus independent secondary NWP streams
    (MET Norway / Yr.no & DWD ICON) for the specified offshore wind corridor, persisting updates to
    `data/era5_<area_id>.csv`. Automatically falls back to live secondary NWP + local/archive caches
    when the primary meteorological stream is unavailable.
    """
    area = OFFSHORE_AREAS.get(area_id, OFFSHORE_AREAS["borssele"])
    area_key = area["id"]

    now_ts = time.time()
    cached = _LIVE_CACHE.get(area_key)
    if (
        not simulate_primary_outage
        and not force_refresh
        and cached
        and (now_ts - cached["fetched_at"] < CACHE_TTL_SECONDS)
    ):
        return cached["df"].copy(), cached["now_str"], cached["meta"]

    utc_now_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    now_str = utc_now_hour.strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, f"era5_{area_key}.csv")

    cds_used = False
    if (cds_api_key or os.environ.get("CDS_API_KEY")) and not simulate_primary_outage:
        nc_path = os.path.join(data_dir, f"cds_{area_key}.nc")
        cds_used = _try_copernicus_cds_sync(area, nc_path, cds_api_key=cds_api_key)

    secondary_map, secondary_sources = _fetch_secondary_nwp_streams(
        area, skip_open_meteo=simulate_primary_outage
    )

    primary_exc: Optional[Exception] = None
    if not simulate_primary_outage:
        try:
            df = _fetch_open_meteo_primary(area, utc_now_hour)
            df = _attach_secondary_columns(df, secondary_map)
            df.to_csv(csv_path, index=False)

            source_label = f"Open-Meteo ECMWF/ERA5-Seamless ({area['lat']:.4f}°N, {area['lon']:.4f}°E)"
            if secondary_sources:
                source_label += " + " + " & ".join(secondary_sources)
            if cds_used:
                source_label += " + Copernicus CDS"

            meta = {
                "area_id": area_key,
                "area_name": area["name"],
                "synced_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "now_timestamp": now_str,
                "rows_synced": len(df),
                "csv_path": csv_path,
                "source": source_label,
                "primary_sync_ok": True,
                "secondary_nwp_ok": bool(secondary_sources),
                "secondary_sources": secondary_sources,
                "live_sync_ok": True,
                "fallback_reason": None,
            }
            _LIVE_CACHE[area_key] = {
                "fetched_at": now_ts,
                "df": df,
                "now_str": now_str,
                "meta": meta,
            }
            return df.copy(), now_str, meta
        except Exception as exc:
            primary_exc = exc
            print(f"[ERA5-Updater] Primary API stream failure for {area_key}: {exc}")
    else:
        primary_exc = RuntimeError("Simulated primary API outage")

    # Degraded Fallback Path 1: Load disk-cached CSV or archive fallback aligned to utc_now_hour
    df_fb: Optional[pd.DataFrame] = None
    cache_origin = "Local ERA5 Reanalysis Archive"
    if os.path.exists(csv_path):
        df_disk = pd.read_csv(csv_path)
        df_disk["valid_time"] = pd.to_datetime(df_disk["valid_time"])
        n_rows = len(df_disk)
        anchor_idx = max(48, n_rows - 25)
        df_disk["valid_time"] = [
            utc_now_hour + timedelta(hours=(i - anchor_idx))
            for i in range(n_rows)
        ]
        df_fb = df_disk
        cache_origin = f"Disk Cache ({area['name']})"
    elif fallback_df is not None and len(fallback_df) >= 120:
        df_tail = fallback_df.iloc[-121:].copy().reset_index(drop=True)
        n_rows = len(df_tail)
        anchor_idx = n_rows - 25
        df_tail["valid_time"] = [
            utc_now_hour + timedelta(hours=(i - anchor_idx))
            for i in range(n_rows)
        ]
        df_fb = df_tail
        cache_origin = f"Collected Archive ({area['name']})"

    if df_fb is not None:
        # Overlay live independent MET Norway / secondary NWP onto future horizon if available
        if secondary_map:
            for idx, row in df_fb.iterrows():
                ts = row["valid_time"]
                if ts in secondary_map and ts >= pd.Timestamp(utc_now_hour):
                    sec_ws = secondary_map[ts]["ws100"]
                    sec_wd = secondary_map[ts]["wd100"]
                    u_s, v_s = _wind_speed_dir_to_uv(np.array([sec_ws]), np.array([sec_wd]))
                    df_fb.at[idx, "u100"] = float(u_s[0])
                    df_fb.at[idx, "v100"] = float(v_s[0])
                    df_fb.at[idx, "msl"] = float(secondary_map[ts]["msl_pa"])
            df_fb = _attach_secondary_columns(df_fb, secondary_map)

        sec_desc = " + " + " & ".join(secondary_sources) if secondary_sources else " (Offline Cache)"
        meta = {
            "area_id": area_key,
            "area_name": area["name"],
            "synced_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "now_timestamp": now_str,
            "rows_synced": len(df_fb),
            "csv_path": csv_path,
            "source": f"Safe Fallback: {cache_origin}{sec_desc}",
            "primary_sync_ok": False,
            "secondary_nwp_ok": bool(secondary_sources),
            "secondary_sources": secondary_sources,
            "live_sync_ok": False,
            "fallback_reason": f"Primary stream unavailable ({primary_exc})",
        }
        return df_fb, now_str, meta

    raise RuntimeError(f"Could not fetch or load ERA5 data for area '{area_key}': {primary_exc}")


def fetch_historical_era5_window(
    area_id: str,
    target_dt: pd.Timestamp,
    fallback_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Resolves a 6-day ERA5 reanalysis window around any user-selected historical date & hour
    (Year, Month, Day, Hour) for any of the 5 offshore wind corridors:
      1. If area_id == 'borssele' and target_dt is present in fallback_df (df26.csv) with
         contiguous 24h lookahead, uses fallback_df directly.
      2. Otherwise fetches [target_dt - 3d, target_dt + 2d] from the ERA5 Archive API
         with in-memory caching.
    """
    area_key = area_id if area_id in OFFSHORE_AREAS else "borssele"
    area = OFFSHORE_AREAS[area_key]

    if area_key == "borssele" and fallback_df is not None:
        matches = fallback_df.index[fallback_df["valid_time"] == target_dt].tolist()
        if matches:
            idx = matches[0]
            if idx + 24 < len(fallback_df):
                dt_plus_24 = fallback_df.loc[idx + 24, "valid_time"]
                if (dt_plus_24 - target_dt) == pd.Timedelta(hours=24):
                    return fallback_df, {
                        "area_id": area_key,
                        "area_name": area["name"],
                        "synced_at": "Copernicus C3S ERA5 Archive (df26.csv)",
                        "now_timestamp": target_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "rows_synced": len(fallback_df),
                        "source": f"Copernicus C3S ERA5 Reanalysis ({area['lat']:.4f}°N, {area['lon']:.4f}°E)",
                        "primary_sync_ok": True,
                        "secondary_nwp_ok": False,
                        "secondary_sources": ["ERA5 Reanalysis Verification Truth"],
                        "live_sync_ok": True,
                        "fallback_reason": None,
                    }

    start_date_str = (target_dt - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    end_date_str = (target_dt + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    cache_key = f"{area_key}:{start_date_str}:{end_date_str}"

    if cache_key in _HIST_WINDOW_CACHE:
        df_cached = _HIST_WINDOW_CACHE[cache_key]
        return df_cached, {
            "area_id": area_key,
            "area_name": area["name"],
            "synced_at": "Copernicus C3S ERA5 On-Demand Archive Cache",
            "now_timestamp": target_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "rows_synced": len(df_cached),
            "source": f"Copernicus C3S ERA5 Archive API ({area['lat']:.4f}°N, {area['lon']:.4f}°E)",
            "primary_sync_ok": True,
            "secondary_nwp_ok": False,
            "secondary_sources": ["ERA5 Archive Cache"],
            "live_sync_ok": True,
            "fallback_reason": None,
        }

    try:
        hourly_vars = ",".join([
            "wind_speed_100m",
            "wind_direction_100m",
            "wind_speed_10m",
            "wind_direction_10m",
            "pressure_msl",
            "surface_pressure",
            "temperature_2m",
            "dew_point_2m",
            "cloud_cover",
            "precipitation",
        ])
        url = (
            f"https://archive-api.open-meteo.com/v1/era5"
            f"?latitude={area['lat']}&longitude={area['lon']}"
            f"&start_date={start_date_str}&end_date={end_date_str}"
            f"&hourly={hourly_vars}"
            f"&wind_speed_unit=ms&timezone=UTC"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "WindGuard-DigitalTwin/2.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        hourly = payload.get("hourly", {})
        times = hourly.get("time", [])
        if times and any(x is not None for x in hourly.get("wind_speed_100m", [])):
            n_rows = len(times)
            ws100 = pd.Series(hourly.get("wind_speed_100m", [8.0] * n_rows), dtype=float).ffill().bfill().values
            wd100 = pd.Series(hourly.get("wind_direction_100m", [240.0] * n_rows), dtype=float).ffill().bfill().values
            ws10 = pd.Series(hourly.get("wind_speed_10m", ws100 * 0.78), dtype=float).ffill().bfill().values
            wd10 = pd.Series(hourly.get("wind_direction_10m", wd100), dtype=float).ffill().bfill().values
            msl_hpa = pd.Series(hourly.get("pressure_msl", [1013.25] * n_rows), dtype=float).ffill().bfill().values
            sp_hpa = pd.Series(hourly.get("surface_pressure", msl_hpa), dtype=float).ffill().bfill().values
            temp_c = pd.Series(hourly.get("temperature_2m", [12.0] * n_rows), dtype=float).ffill().bfill().values
            dew_c = pd.Series(hourly.get("dew_point_2m", [8.0] * n_rows), dtype=float).ffill().bfill().values
            cloud_pct = pd.Series(hourly.get("cloud_cover", [50.0] * n_rows), dtype=float).ffill().bfill().values
            precip_mm = pd.Series(hourly.get("precipitation", [0.0] * n_rows), dtype=float).ffill().bfill().values

            u100, v100 = _wind_speed_dir_to_uv(ws100, wd100)
            u10, v10 = _wind_speed_dir_to_uv(ws10, wd10)

            df_win = pd.DataFrame({
                "valid_time": pd.to_datetime([t.replace("T", " ") + ":00" for t in times]),
                "u100": u100,
                "v100": v100,
                "u10": u10,
                "v10": v10,
                "d2m": dew_c + 273.15,
                "t2m": temp_c + 273.15,
                "msl": msl_hpa * 100.0,
                "sp": sp_hpa * 100.0,
                "tcc": np.clip(cloud_pct / 100.0, 0.0, 1.0),
                "tp": precip_mm / 1000.0,
                "latitude": float(area["grid_center"][0]),
                "longitude": float(area["grid_center"][1]),
            })
            _HIST_WINDOW_CACHE[cache_key] = df_win
            return df_win, {
                "area_id": area_key,
                "area_name": area["name"],
                "synced_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "now_timestamp": target_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "rows_synced": len(df_win),
                "source": f"Copernicus C3S ERA5 Archive API ({area['lat']:.4f}°N, {area['lon']:.4f}°E)",
                "primary_sync_ok": True,
                "secondary_nwp_ok": False,
                "secondary_sources": ["ERA5 Archive Verification"],
                "live_sync_ok": True,
                "fallback_reason": None,
            }
    except Exception as exc:
        print(f"[ERA5-Updater] Historical window fetch fallback for {area_key} at {target_dt}: {exc}")

    if fallback_df is not None:
        return fallback_df, {
            "area_id": area_key,
            "area_name": area["name"],
            "synced_at": "Local ERA5 Reanalysis Archive",
            "now_timestamp": target_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "rows_synced": len(fallback_df),
            "source": f"Local ERA5 Archive ({area['name']})",
            "primary_sync_ok": False,
            "secondary_nwp_ok": False,
            "secondary_sources": ["Local ERA5 Archive"],
            "live_sync_ok": False,
            "fallback_reason": "Historical Archive API unavailable; using local ERA5 dataset",
        }
    raise RuntimeError(f"Unable to load historical ERA5 data for {area_key} at {target_dt}")
