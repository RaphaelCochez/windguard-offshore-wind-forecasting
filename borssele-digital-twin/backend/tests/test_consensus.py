import math
import unittest
from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.consensus import (
    compute_secondary_fallback_and_consensus,
    evaluate_fallback_state,
    build_grid_strategy_and_telemetry,
)


class TestConsensusModule(unittest.TestCase):
    def test_decaying_persistence_fallback_and_upper_envelope(self):
        # At h=1 vs h=24 with no external secondary NWP, persistence influence decays exponentially
        r_h1 = compute_secondary_fallback_and_consensus(
            pred_ens=12.0,
            pred_v2=12.0,
            pred_v1=11.0,
            pred_pers=20.0,
            nwp_synoptic_h=10.0,
            sec_ext_ws=None,
            step=1,
        )
        r_h24 = compute_secondary_fallback_and_consensus(
            pred_ens=12.0,
            pred_v2=12.0,
            pred_v1=11.0,
            pred_pers=20.0,
            nwp_synoptic_h=10.0,
            sec_ext_ws=None,
            step=24,
        )
        beta_1 = 0.20 * math.exp(-1.0 / 6.0)
        beta_24 = 0.20 * math.exp(-24.0 / 6.0)
        self.assertAlmostEqual(r_h1["nwp_secondary"], (1.0 - beta_1) * 10.0 + beta_1 * 20.0, places=9)
        self.assertAlmostEqual(r_h24["nwp_secondary"], (1.0 - beta_24) * 10.0 + beta_24 * 20.0, places=9)
        self.assertGreater(r_h1["nwp_secondary"], r_h24["nwp_secondary"])

        # Above 20.0 m/s caution threshold, fallback adopts conservative upper envelope max(nwp_sec, pred_ens)
        r_caution = compute_secondary_fallback_and_consensus(
            pred_ens=23.5,
            pred_v2=23.5,
            pred_v1=18.0,
            pred_pers=19.0,
            nwp_synoptic_h=19.5,
            sec_ext_ws=20.2,
            step=6,
        )
        self.assertEqual(r_caution["pred_fallback"], 23.5)

    def test_fallback_trigger_priority(self):
        divergent_window = [{"consensus_score_h": 0.4, "pred_ens": 22.0, "nwp_secondary": 18.0}] * 6

        # 1. Manual override wins over everything
        is_fb, auto_fb, reason, mode, _ = evaluate_fallback_state(
            raw_24h=divergent_window,
            horizon_hours=6,
            is_live_mode=True,
            sync_meta={"primary_sync_ok": False},
            missing_ml_models=True,
            requested_sys_mode="degraded_fallback",
            requested_mode="ensemble",
        )
        self.assertTrue(is_fb)
        self.assertFalse(auto_fb)
        self.assertEqual(reason, "MANUAL_OPERATOR_SAFE_FALLBACK")
        self.assertEqual(mode, "degraded_fallback")

        # 2. Primary stream loss
        _, auto_fb, reason, mode, lost = evaluate_fallback_state(
            raw_24h=divergent_window,
            horizon_hours=6,
            is_live_mode=True,
            sync_meta={"primary_sync_ok": False},
            missing_ml_models=True,
            requested_sys_mode="normal",
            requested_mode="ensemble",
        )
        self.assertTrue(auto_fb)
        self.assertTrue(lost)
        self.assertEqual(reason, "AUTO_FALLBACK_PRIMARY_STREAM_LOSS")
        self.assertEqual(mode, "auto_degraded_fallback")

        # 3. Missing ML models
        _, auto_fb, reason, _, _ = evaluate_fallback_state(
            raw_24h=divergent_window,
            horizon_hours=6,
            is_live_mode=False,
            sync_meta={"primary_sync_ok": True},
            missing_ml_models=True,
            requested_sys_mode="normal",
            requested_mode="ensemble",
        )
        self.assertTrue(auto_fb)
        self.assertEqual(reason, "AUTO_FALLBACK_ML_MODELS_UNAVAILABLE")

        # 4. High-wind consensus divergence
        _, auto_fb, reason, _, _ = evaluate_fallback_state(
            raw_24h=divergent_window,
            horizon_hours=6,
            is_live_mode=False,
            sync_meta={"primary_sync_ok": True},
            missing_ml_models=False,
            requested_sys_mode="normal",
            requested_mode="ensemble",
        )
        self.assertTrue(auto_fb)
        self.assertEqual(reason, "AUTO_FALLBACK_HIGH_WIND_CONSENSUS_DIVERGENCE")

    def test_alert_levels_in_grid_telemetry(self):
        cases = [
            ({"cutout_hours": 2, "actualcutout_hours": 2, "max_actual_ws_ms": 27.0, "caution_hours": 3, "min_cutout_margin_ms": -2.0, "below_cutin_hours": 0}, "CRITICAL_SHUTDOWN_ALERT"),
            ({"cutout_hours": 0, "actualcutout_hours": 0, "max_actual_ws_ms": 22.0, "caution_hours": 4, "min_cutout_margin_ms": 3.0, "below_cutin_hours": 0}, "HIGH_WIND_CAUTION_ALERT"),
            ({"cutout_hours": 0, "actualcutout_hours": 0, "max_actual_ws_ms": 2.5, "caution_hours": 0, "min_cutout_margin_ms": 22.5, "below_cutin_hours": 5}, "LOW_WIND_STANDBY"),
            ({"cutout_hours": 0, "actualcutout_hours": 0, "max_actual_ws_ms": 12.0, "caution_hours": 0, "min_cutout_margin_ms": 13.0, "below_cutin_hours": 0}, "GOOD_CONDITIONS"),
        ]
        for wm, expected_alert in cases:
            pt = build_grid_strategy_and_telemetry(
                window_metrics=wm,
                horizon_hours=24,
                max_pred_ws=26.0,
                is_live_mode=False,
                sync_meta={"secondary_sources": ["ERA5"]},
                primary_stream_lost=False,
                is_fallback_active=False,
                auto_fallback_triggered=False,
                fallback_trigger_reason="NONE",
                consensus_rate=95.0,
                latency_ms=12.3,
            )
            self.assertEqual(pt["alert_level"], expected_alert)


if __name__ == "__main__":
    unittest.main(verbosity=2)
