import unittest
from pathlib import Path
import sys
import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.feature_engineering import (
    V1_FEATURE_COLS,
    V2_FEATURE_COLS,
    build_era5_features,
    engineer_features_for_inference,
    attach_horizon_nwp_features_bulk,
    attach_horizon_nwp_features_single,
)


def _make_synthetic_era5(n_hours: int = 80, start: str = "2026-03-01 00:00:00") -> pd.DataFrame:
    times = pd.date_range(start=start, periods=n_hours, freq="1h")
    t = np.arange(n_hours, dtype=float)
    return pd.DataFrame(
        {
            "valid_time": times,
            "u100": 8.0 + 4.0 * np.sin(t / 6.0),
            "v100": 3.0 + 2.5 * np.cos(t / 8.0),
            "u10": 6.2 + 3.0 * np.sin(t / 6.0),
            "v10": 2.4 + 1.8 * np.cos(t / 8.0),
            "d2m": 278.0 + 1.5 * np.sin(t / 12.0),
            "t2m": 282.0 + 2.0 * np.sin(t / 12.0),
            "msl": 101200.0 - 400.0 * np.sin(t / 14.0),
            "sp": 101100.0 - 380.0 * np.sin(t / 14.0),
            "tcc": np.clip(0.5 + 0.3 * np.cos(t / 10.0), 0.0, 1.0),
            "tp": np.clip(0.0005 * np.sin(t / 5.0), 0.0, None),
        }
    )


class TestFeatureEngineering(unittest.TestCase):
    def test_feature_schema_counts_and_uniqueness(self):
        self.assertEqual(len(V1_FEATURE_COLS), 100)
        self.assertEqual(len(set(V1_FEATURE_COLS)), 100)
        self.assertEqual(len(V2_FEATURE_COLS), 150)
        self.assertEqual(len(set(V2_FEATURE_COLS)), 150)
        self.assertNotIn("latitude", V2_FEATURE_COLS)
        self.assertNotIn("longitude", V2_FEATURE_COLS)

    def test_short_lookback_and_nan_safety(self):
        for n_rows in (1, 5, 24, 49):
            df = _make_synthetic_era5(n_hours=n_rows)
            feat = build_era5_features(df)
            self.assertEqual(len(feat), n_rows)

            v1_row = engineer_features_for_inference(df, version="v1")
            v2_row = engineer_features_for_inference(df, version="v2")
            self.assertEqual(list(v1_row.columns), V1_FEATURE_COLS)
            self.assertEqual(list(v2_row.columns), V2_FEATURE_COLS)
            self.assertFalse(v1_row.isna().any().any(), f"NaN in v1 with {n_rows} rows")
            self.assertFalse(v2_row.isna().any().any(), f"NaN in v2 with {n_rows} rows")
            self.assertTrue(np.isfinite(v1_row.to_numpy()).all())
            self.assertTrue(np.isfinite(v2_row.to_numpy()).all())

    def test_bulk_vs_single_offline_parity(self):
        df = _make_synthetic_era5(n_hours=96)
        feat_all = build_era5_features(df)
        feat_indexed = feat_all.set_index("valid_time")

        current_idx = 48
        df_slice = df.iloc[: current_idx + 1].copy()
        X_v2_base = engineer_features_for_inference(df_slice, version="v2")

        for h in (1, 6, 12, 24):
            X_bulk = attach_horizon_nwp_features_bulk(feat_indexed, h=h)
            self.assertEqual(X_bulk.shape[1], 157)

            X_single = attach_horizon_nwp_features_single(
                X_v2_base=X_v2_base,
                active_df=df,
                current_idx=current_idx,
                h=h,
                is_live_mode=False,
            )
            self.assertEqual(list(X_single.columns), list(X_bulk.columns))

            bulk_row = X_bulk.iloc[current_idx]
            single_row = X_single.iloc[0]
            for col in X_bulk.columns:
                self.assertAlmostEqual(
                    float(single_row[col]),
                    float(bulk_row[col]),
                    places=9,
                    msg=f"Mismatch on col={col} at h={h}",
                )

    def test_live_mode_and_past_end_fallback(self):
        df = _make_synthetic_era5(n_hours=60)
        current_idx = 48
        X_v2_base = engineer_features_for_inference(df.iloc[: current_idx + 1], version="v2")

        # Live mode within bounds: 2-hour causal average, 0 dispersion
        X_live = attach_horizon_nwp_features_single(
            X_v2_base=X_v2_base,
            active_df=df,
            current_idx=current_idx,
            h=6,
            is_live_mode=True,
        )
        self.assertEqual(X_live.shape, (1, 157))
        sub = df.iloc[current_idx + 5 : current_idx + 7]
        expected_ws = float(np.mean(np.sqrt(sub["u100"] ** 2 + sub["v100"] ** 2)))
        self.assertAlmostEqual(float(X_live["synoptic_nwp_ws_h"].iloc[0]), expected_ws, places=9)

        # Past-end horizon fallback: falls back to current origin without raising
        X_oob = attach_horizon_nwp_features_single(
            X_v2_base=X_v2_base,
            active_df=df,
            current_idx=current_idx,
            h=24,
            is_live_mode=True,
        )
        self.assertEqual(X_oob.shape, (1, 157))
        self.assertAlmostEqual(float(X_oob["synoptic_nwp_delta_h"].iloc[0]), 0.0, places=9)
        self.assertAlmostEqual(float(X_oob["synoptic_nwp_msl_delta_h"].iloc[0]), 0.0, places=9)

    def test_non_contiguous_gap_does_not_leak(self):
        df_a = _make_synthetic_era5(n_hours=30, start="2020-02-10 00:00:00")
        df_b = _make_synthetic_era5(n_hours=30, start="2022-02-14 00:00:00")
        df_b["u100"] = 30.0
        df_b["v100"] = 0.0
        gapped = pd.concat([df_a, df_b], ignore_index=True)

        current_idx = 29  # Last row before 2-year jump
        X_v2_base = engineer_features_for_inference(df_a, version="v2")
        X_gap = attach_horizon_nwp_features_single(
            X_v2_base=X_v2_base,
            active_df=gapped,
            current_idx=current_idx,
            h=1,
            is_live_mode=False,
        )
        # Must fall back to origin wind speed rather than reading 30.0 m/s from 2022
        self.assertAlmostEqual(float(X_gap["synoptic_nwp_delta_h"].iloc[0]), 0.0, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
