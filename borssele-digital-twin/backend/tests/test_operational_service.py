import asyncio
import unittest
from pathlib import Path
import sys
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import main
from app.physics_engine import evaluate_turbine_operability
from app.config import CUT_IN_SPEED_MS, CAUTION_SPEED_MS, CUT_OUT_SPEED_MS, MAX_HORIZON_HOURS


class TestOperationalService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load dataset & trained boosters once cleanly without starting background infinite task
        data_path = BACKEND_DIR / "data" / "df26.csv"
        df = pd.read_csv(data_path)
        time_col = "valid_time" if "valid_time" in df.columns else "time"
        df["valid_time"] = pd.to_datetime(df[time_col])
        main.HISTORICAL_DF = df.sort_values("valid_time").drop_duplicates("valid_time").reset_index(drop=True)

        import json
        import lightgbm as lgb

        v1_dir = BACKEND_DIR / "models" / "wind_forecast_v1"
        v2_dir = BACKEND_DIR / "models" / "wind_forecast_v2"
        main.MODELS_V1.clear()
        main.MODELS_V2.clear()
        for h in range(1, MAX_HORIZON_HOURS + 1):
            p1 = v1_dir / f"lgb_model_h_{h}.txt"
            p2 = v2_dir / f"lgb_model_h_{h}.txt"
            if p1.exists():
                main.MODELS_V1[h] = lgb.Booster(model_file=str(p1))
            if p2.exists():
                main.MODELS_V2[h] = lgb.Booster(model_file=str(p2))

        summary_path = v2_dir / "metrics_summary.json"
        if summary_path.exists():
            main.METRICS_SUMMARY = json.loads(summary_path.read_text())
            for k, v in main.METRICS_SUMMARY.get("horizons", {}).items():
                main.ENSEMBLE_WEIGHTS[int(k.replace("h_", ""))] = float(v.get("v2_weight", 1.0))

    def test_four_wind_operability_regimes(self):
        low = evaluate_turbine_operability(CUT_IN_SPEED_MS - 0.5)
        self.assertTrue(low["can_operate"])
        self.assertEqual(low["state_code"], "LOW_WIND_STANDBY")

        good = evaluate_turbine_operability(12.0)
        self.assertTrue(good["can_operate"])
        self.assertEqual(good["state_code"], "GOOD_CONDITIONS")

        caution = evaluate_turbine_operability(CAUTION_SPEED_MS + 1.5)
        self.assertTrue(caution["can_operate"])
        self.assertEqual(caution["state_code"], "HIGH_WIND_CAUTION")

        shutdown = evaluate_turbine_operability(CUT_OUT_SPEED_MS)
        self.assertFalse(shutdown["can_operate"])
        self.assertEqual(shutdown["state_code"], "CUT_OUT_SHUTDOWN")

    def test_horizon_and_future_timestamp_clamped(self):
        req = main.ForecastRequest(
            timestamp="2099-01-01 00:00:00",
            area_id="borssele",
            horizon_hours=72,
            system_mode="normal",
        )
        res = main.get_prediction(req)
        self.assertEqual(res["horizon_hours"], MAX_HORIZON_HOURS)
        self.assertEqual(len(res["forecast"]), MAX_HORIZON_HOURS)
        self.assertTrue(res["is_live"])

    def test_three_historical_events_normal_and_degraded_modes(self):
        events = [
            ("2022-02-18 06:00:00", True),   # Storm Eunice: verified cut-out shut-down
            ("2023-11-02 00:00:00", True),   # Storm Ciaran: verified cut-out shut-down
            ("2025-01-01 06:00:00", False),  # New Year Gale: High Wind Caution, 0h shut-down
        ]
        for sys_mode in ("normal", "degraded_fallback"):
            for ts, expect_shutdown in events:
                res = main.get_prediction(
                    main.ForecastRequest(
                        timestamp=ts,
                        area_id="borssele",
                        horizon_hours=24,
                        system_mode=sys_mode,
                    )
                )
                wm = res["window_metrics"]
                if expect_shutdown:
                    self.assertGreaterEqual(wm["cutout_hours"], 5, f"{sys_mode} missed shutdown at {ts}")
                else:
                    self.assertEqual(wm["cutout_hours"], 0, f"{sys_mode} false shutdown at {ts}")
                self.assertEqual(len(res["forecast"]), 24)

    def test_auto_fallback_on_primary_stream_outage(self):
        res = main.get_prediction(
            main.ForecastRequest(
                timestamp="LIVE",
                area_id="borssele",
                horizon_hours=24,
                system_mode="normal",
                simulate_primary_outage=True,
            )
        )
        pt = res["proposal_telemetry"]
        self.assertEqual(res["system_mode"], "auto_degraded_fallback")
        self.assertTrue(pt["auto_fallback_triggered"])
        self.assertEqual(pt["fallback_trigger_reason"], "AUTO_FALLBACK_PRIMARY_STREAM_LOSS")
        self.assertFalse(res["sync_meta"]["primary_sync_ok"])
        self.assertEqual(len(res["forecast"]), 24)

    def test_auto_fallback_when_v2_models_unavailable(self):
        saved_v2 = dict(main.MODELS_V2)
        try:
            main.MODELS_V2.clear()
            res = main.get_prediction(
                main.ForecastRequest(
                    timestamp="2023-11-02 00:00:00",
                    area_id="borssele",
                    horizon_hours=24,
                    system_mode="normal",
                )
            )
            pt = res["proposal_telemetry"]
            self.assertEqual(res["system_mode"], "auto_degraded_fallback")
            self.assertTrue(pt["auto_fallback_triggered"])
            self.assertEqual(pt["fallback_trigger_reason"], "AUTO_FALLBACK_ML_MODELS_UNAVAILABLE")
            self.assertGreaterEqual(res["window_metrics"]["cutout_hours"], 5)
        finally:
            main.MODELS_V2.update(saved_v2)

    def test_live_multi_nwp_sync(self):
        res = main.get_prediction(
            main.ForecastRequest(
                timestamp="LIVE",
                area_id="tahkoluoto",
                horizon_hours=6,
                system_mode="normal",
            )
        )
        self.assertEqual(res["horizon_hours"], 6)
        self.assertEqual(len(res["forecast"]), 6)
        self.assertIn("independent_nwp_sources", res["proposal_telemetry"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
