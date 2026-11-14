"""
Baseline Cut-Out Shut-Down Classifier (Reference Script)
Note: Production operational multi-horizon (+1h to +24h) LightGBM residual ensemble models
are implemented in borssele-digital-twin/backend/train_models.py.
"""
from sklearn.ensemble import RandomForestClassifier


def train_cut_out_predictor(df_train, forecast_horizon_hours: int = 6):
    if not (1 <= forecast_horizon_hours <= 24):
        raise ValueError("Forecast horizon must be between +1h and +24h.")

    df_work = df_train.copy()
    df_work["target_lead_event"] = (
        df_work.groupby(["latitude", "longitude"])["cut_out_event"].shift(-forecast_horizon_hours)
    )

    clean_data = df_work.dropna(subset=["wind_speed_100m", "ramp_rate_1h", "target_lead_event"])
    features = ["wind_speed_100m", "ramp_rate_1h"]
    X = clean_data[features]
    y = clean_data["target_lead_event"]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf
