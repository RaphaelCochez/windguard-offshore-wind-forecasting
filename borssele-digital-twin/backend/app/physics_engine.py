from app.config import CUT_IN_SPEED_MS, CAUTION_SPEED_MS, CUT_OUT_SPEED_MS


def evaluate_turbine_operability(wind_speed: float) -> dict:
    """
    Evaluates whether the offshore wind park CAN operate and classifies wind conditions
    at a given 100m hub-height wind speed:
      - Low Wind (< 3.0 m/s): CAN OPERATE • LOW WIND (<3 m/s)
      - Good Conditions (3.0 m/s <= v < 20.0 m/s): CAN OPERATE • GOOD CONDITIONS
      - High Wind Caution (20.0 m/s <= v < 25.0 m/s): CAN OPERATE • HIGH WIND CAUTION
      - Cut-Out Shut-Down (v >= 25.0 m/s): CUT-OUT SHUT-DOWN (≥25 m/s)
    """
    ws = float(wind_speed)
    margin_to_cutout = round(CUT_OUT_SPEED_MS - ws, 2)

    if ws >= CUT_OUT_SPEED_MS:
        return {
            "can_operate": False,
            "operability_flag": 0,
            "state_code": "CUT_OUT_SHUTDOWN",
            "state_label": "CUT-OUT SHUT-DOWN (≥25 m/s)",
            "short_label": "Cut-Out Shut-Down (≥25 m/s)",
            "reason": f"Wind speed ({ws:.2f} m/s) exceeds 25.0 m/s — Cut-Out Shut-Down active.",
            "margin_to_cutout_ms": margin_to_cutout,
            "margin_above_cutin_ms": round(ws - CUT_IN_SPEED_MS, 2),
        }
    elif ws >= CAUTION_SPEED_MS:
        return {
            "can_operate": True,
            "operability_flag": 1,
            "state_code": "HIGH_WIND_CAUTION",
            "state_label": "CAN OPERATE • HIGH WIND CAUTION (20–25 m/s)",
            "short_label": "High Wind Caution (20–25 m/s)",
            "reason": f"High Wind Caution (20–25 m/s): {margin_to_cutout:.2f} m/s margin to Cut-Out Shut-Down.",
            "margin_to_cutout_ms": margin_to_cutout,
            "margin_above_cutin_ms": round(ws - CUT_IN_SPEED_MS, 2),
        }
    elif ws < CUT_IN_SPEED_MS:
        return {
            "can_operate": True,
            "operability_flag": 1,
            "state_code": "LOW_WIND_STANDBY",
            "state_label": "CAN OPERATE • LOW WIND (<3 m/s)",
            "short_label": "Can Operate • Low Wind (<3 m/s)",
            "reason": f"Turbines can operate safely, but wind ({ws:.2f} m/s) is below 3.0 m/s.",
            "margin_to_cutout_ms": margin_to_cutout,
            "margin_above_cutin_ms": round(ws - CUT_IN_SPEED_MS, 2),
        }
    else:
        return {
            "can_operate": True,
            "operability_flag": 1,
            "state_code": "GOOD_CONDITIONS",
            "state_label": "CAN OPERATE • GOOD CONDITIONS (3–20 m/s)",
            "short_label": "Good Conditions (3–20 m/s)",
            "reason": f"Good Conditions (3–20 m/s): {margin_to_cutout:.2f} m/s below Cut-Out Shut-Down.",
            "margin_to_cutout_ms": margin_to_cutout,
            "margin_above_cutin_ms": round(ws - CUT_IN_SPEED_MS, 2),
        }
