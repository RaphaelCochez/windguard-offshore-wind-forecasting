"""
Centralized Operational Configuration & Threshold Constants
Service: Operational EO service to predict off-shore wind park shut-down events
Authors: Raphael Cochez, Pasan Merengnage, Mariia Melnikova
"""

SERVICE_TITLE = "Operational EO service to predict off-shore wind park shut-down events"
OPERATIONAL_QUESTION = (
    "Using collected Copernicus ERA5 satellite reanalysis and NWP data, can we make a "
    "sufficiently timely and accurate prediction of offshore wind park shut-down events "
    "(>= 25.0 m/s) for grid-operator decision-makers to make an informed operational decision?"
)

# Strictly defined 100m hub-height wind speed decision thresholds (m/s)
CUT_IN_SPEED_MS = 3.0
CAUTION_SPEED_MS = 20.0
CUT_OUT_SPEED_MS = 25.0

# Offshore boundary-layer power-law exponent used to scale 10m coastal NWP feeds to 100m hub height
POWER_LAW_OFFSHORE_ALPHA = 0.11

# Forecast horizon constraints
SUPPORTED_HORIZONS = [1, 6, 12, 24]
DEFAULT_HORIZON_HOURS = 24
MAX_HORIZON_HOURS = 24

# Multi-source NWP consensus & background sync parameters
CONSENSUS_THRESHOLD_MS = 2.0
MIN_CONSENSUS_AGREEMENT_PCT = 70.0
BACKGROUND_SYNC_INTERVAL_SECONDS = 300
