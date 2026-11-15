import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.config import (
    CAUTION_SPEED_MS,
    CONSENSUS_THRESHOLD_MS,
    MIN_CONSENSUS_AGREEMENT_PCT,
)


def compute_secondary_fallback_and_consensus(
    pred_ens: float,
    pred_v2: float,
    pred_v1: float,
    pred_pers: float,
    nwp_synoptic_h: float,
    sec_ext_ws: Optional[float],
    step: int = 1,
) -> Dict[str, float]:
    """Computes independent secondary NWP trajectory (with time-decaying persistence fallback), conservative envelope, and inter-source spread."""
    beta_h = 0.20 * math.exp(-float(max(1, step)) / 6.0)
    nwp_secondary = (
        sec_ext_ws
        if sec_ext_ws is not None
        else float((1.0 - beta_h) * nwp_synoptic_h + beta_h * pred_pers)
    )
    pred_fallback = float(
        max(nwp_secondary, pred_ens)
        if max(nwp_secondary, pred_ens) >= CAUTION_SPEED_MS
        else nwp_secondary
    )
    source_spread = float(
        max(pred_v2, pred_v1, nwp_secondary) - min(pred_v2, pred_v1, nwp_secondary)
    )
    consensus_score_h = float(max(0.0, 1.0 - (source_spread / max(pred_ens, 6.0))))
    return {
        "nwp_secondary": nwp_secondary,
        "pred_fallback": pred_fallback,
        "source_spread": source_spread,
        "consensus_score_h": consensus_score_h,
    }


def evaluate_fallback_state(
    raw_24h: List[Dict[str, Any]],
    horizon_hours: int,
    is_live_mode: bool,
    sync_meta: Dict[str, Any],
    missing_ml_models: bool,
    requested_sys_mode: str,
    requested_mode: str,
) -> Tuple[bool, bool, str, str, bool]:
    """
    Evaluates manual override and three autonomous safe-fallback triggers:
      - AUTO_FALLBACK_PRIMARY_STREAM_LOSS
      - AUTO_FALLBACK_ML_MODELS_UNAVAILABLE
      - AUTO_FALLBACK_HIGH_WIND_CONSENSUS_DIVERGENCE
    Returns (is_fallback_active, auto_fallback_triggered, fallback_trigger_reason, effective_system_mode, primary_stream_lost).
    """
    primary_stream_lost = is_live_mode and not bool(sync_meta.get("primary_sync_ok", True))

    prelim_window = raw_24h[:horizon_hours]
    prelim_consensus_pct = float(np.mean([r["consensus_score_h"] * 100.0 for r in prelim_window]))
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
    return (
        is_fallback_active,
        auto_fallback_triggered,
        fallback_trigger_reason,
        effective_system_mode,
        primary_stream_lost,
    )


def build_grid_strategy_and_telemetry(
    window_metrics: Dict[str, Any],
    horizon_hours: int,
    max_pred_ws: float,
    is_live_mode: bool,
    sync_meta: Dict[str, Any],
    primary_stream_lost: bool,
    is_fallback_active: bool,
    auto_fallback_triggered: bool,
    fallback_trigger_reason: str,
    consensus_rate: float,
    latency_ms: float,
) -> Dict[str, Any]:
    """Builds TSO operational strategy text and proposal telemetry block."""
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

    return {
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
