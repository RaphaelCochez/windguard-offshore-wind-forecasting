# WindGuard: Technical Methodology, Mathematical Formulation & Resilience Reference

This document provides the complete mathematical formulation, atmospheric physics derivations, LightGBM residual training objective, continuous evaluation metrics, and safe-fallback equations for the **WindGuard Offshore Wind Shut-Down Forecasting Service**.

For the project overview, architecture diagram, and Docker quick start, see [README.md](../README.md).

---

## 1. 10-Year Climatological Baseline (`2016–2025`)

An exploratory 10-year Copernicus ERA5 reanalysis study (`2016–2025`) was conducted for the Borssele Offshore Wind Farm (`51.716667°N, 3.000000°E`) in [10_years_Analysis_Borssele_offshore_wind_farm.ipynb](../10_years_Analysis_Borssele_offshore_wind_farm.ipynb):

- **Heavy-Tail Storm Sparsity:** Across the 10-year ERA5 sample, 100 m wind speed quantiles are `10.08 m/s` (50%), `13.73 m/s` (75%), `17.17 m/s` (90%), `19.64 m/s` (95%), and `23.60 m/s` (99%). Days with daily peak $W_{100} > 25.0\text{ m/s}$ occurred only **20 days across 2016–2025** (`2016: 3`, `2017: 2`, `2018: 1`, `2019: 0`, `2020: 5`, `2021: 3`, `2022: 2`, `2023: 1`, `2024: 3`, `2025: 0`).
- **Seasonal Regime Clustering & Lag Autocorrelation:** Extreme winds cluster in late autumn through early spring (Nov–Apr) and exhibit strong short-term autocorrelation (`1h–24h` lags) that decays at multi-day lead times—explaining why standard uniform MSE tree regression regresses storm peaks toward the seasonal mean.

---

## 2. Atmospheric Physics & Feature Engineering (`157` Features per Horizon in `v2`)

Implemented in [feature_engineering.py](../borssele-digital-twin/backend/app/feature_engineering.py), `build_era5_features` and `attach_horizon_nwp_features_*` construct `100` baseline features (`V1_FEATURE_COLS`) and `150` base features (`V2_FEATURE_COLS`: 98 from `v1` excluding `latitude`/`longitude` + 52 `V2_EXTRA_COLS`), plus `7` horizon-specific covariates for each lead time $h \in \{1,\dots,24\}$ (**`157` total features** per `v2` model):

1. **Hub-Height & Surface Scalar Wind Speeds:**

$$
W_{100} = \sqrt{u_{100}^2 + v_{100}^2}, \qquad W_{10} = \sqrt{u_{10}^2 + v_{10}^2}
$$

2. **Vertical Scalar Shear, Vector Shear & Hellmann Power-Law Exponent ($\alpha$):**

$$
\Delta W_{100-10} = W_{100} - W_{10}, \qquad \Delta\mathbf{V}_{100-10} = \sqrt{(u_{100}-u_{10})^2 + (v_{100}-v_{10})^2}, \qquad \alpha = \frac{\ln(\max(W_{100}, 0.1) / \max(W_{10}, 0.1))}{\ln(100 / 10)}
$$

3. **Ekman Spiral Directional Veering / Backing:**
   Encodes wind direction ($\theta_{100} = \text{atan2}(u_{100}, v_{100})$, $\theta_{10} = \text{atan2}(u_{10}, v_{10})$) and vertical directional shear via `wind_dir_100m_sin/cos`, `veering_sin` ($= \sin(\theta_{100} - \theta_{10})$), and `veering_cos` ($= \cos(\theta_{100} - \theta_{10})$).

4. **Thermodynamic Air Density & Moisture Proxies:**
   Dry-air equation-of-state density at surface pressure (`sp`) and `2 m` temperature (`t2m`) using $R_d = 287.058\text{ J/(kg}\cdot\text{K)}$, plus dew-point depression:

$$
\rho = \frac{\text{sp}}{R_d\,t_{2\text{m}}}, \qquad \Delta T_d = t_{2\text{m}} - d_{2\text{m}}
$$

5. **Barometric Pressure Tendencies, Z-Score & Baroclinic Storm Index ($I_{\text{baro}}$):**
   First-order pressure differences ($\Delta\text{MSL}_{1\text{h}, 3\text{h}, 6\text{h}, 12\text{h}, 24\text{h}}$), pressure accelerations (`msl_accel_1h/3h/6h`), 24h rolling pressure Z-score (`msl_zscore_24h` with `+25.0 Pa` stabilizer), frontal `t2m`/`d2m` differences, and the **Baroclinic Storm Index**:

$$
I_{\text{baro}}(t) = -\frac{\Delta\text{MSL}_{6\text{h}}(t)}{100.0} \cdot W_{100}(t)
$$

6. **Wind Ramp Rates, Autoregressive Lags, Rolling Windows & EWM Momentum:**
   Wind speed/vector differences and accelerations (`1h–24h`), 48 autoregressive lags at `{1, 2, 3, 6, 12, 24}h` across 8 variables (`wind_speed_100m`, `shear_100_10`, `msl_diff_3h`, `air_density`, `u100`, `v100`, `tcc`, `tp`), shifted (`v1`) and current-inclusive (`v2`) `6h / 12h / 24h` rolling statistics, and EWM wind momentum (`ewm_ws_3h/6h/12h/24h`, `ws_ewm_macd = ewm_ws_3h - ewm_ws_12h`).

7. **Horizon-Specific Synoptic NWP Guidance Features (`+7` cols → `157` total):**
   `target_hour_sin`, `target_hour_cos`, `synoptic_nwp_ws_h`, `synoptic_nwp_delta_h`, `synoptic_nwp_u100_h`, `synoptic_nwp_v100_h`, and `synoptic_nwp_msl_delta_h`.
   - *Live operational mode (`is_live_mode=True`):* Computed directly from the incoming Open-Meteo ECMWF/ERA5-Seamless forecast rows around `t + h` (2-hour causal smoothing, `dispersion = 0.0`).
   - *Offline historical training/backtesting (`is_live_mode=False`):* Synthesized as a 4-hour smoothed synoptic flow at `t + h` perturbed by a deterministic $\sqrt{h}$-growing phase/amplitude dispersion (`sigma_h = 0.26 * sqrt(h)`).

---

## 3. Physics-Informed Residual LightGBM (`wind_forecast_v2`)

Direct tree regression (`wind_forecast_v1`) trained solely on state features at $t_0$ suffers from regression-to-the-mean leaf clipping and objective/early-stopping mismatch (`-4.223 m/s` high-wind bias at `+24h`). [train_models.py](../borssele-digital-twin/backend/train_models.py) resolves both issues in three steps:

1. **Physical Persistence-to-Synoptic Baseline Blend:**
   For each horizon $h \in \{1,\dots,24\}$, a baseline trajectory $y_{\text{base}, h}(t)$ transitions smoothly from initial persistence $W_{100}(t)$ at `+1h` ($\alpha_1 \approx 0.455$) to synoptic NWP guidance $\text{synoptic\_nwp\_ws}_h$ at `+24h` ($\alpha_{24} = 0.920$):

$$
y_{\text{base}, h}(t) = (1 - \alpha_h)\,W_{100}(t) + \alpha_h\,\text{synoptic\_nwp\_ws}_h(t), \qquad \alpha_h = \min\!\left(0.92,\; \frac{h}{h + 1.2}\right)
$$

2. **Asymmetric Residual Boosting with Aligned Validation Metric:**
   Each LightGBM booster (`num_leaves=31`, `max_depth=6`, `lr=0.03`) predicts the residual $\delta_h(t) = W_{100}(t+h) - y_{\text{base}, h}(t)$ using a custom asymmetric objective paired with a matching validation metric (`asym_wrmse`) in early stopping:

$$
\mathcal{L}_{\text{asym}}(\hat{\delta}_h, \delta_h) = \sum_{i} w_i \left(\hat{\delta}_{h, i} - \delta_{h, i}\right)^2, \qquad w_i = \begin{cases} 1.45 & \text{if } W_{\text{true}, i} > 18.0\text{ m/s and } \widehat{W}_i < W_{\text{true}, i} \\ 1.00 & \text{otherwise} \end{cases}
$$

3. **Optimal Validation Ensemble Calibration:**
   Validation predictions $\widehat{W}_h^{(v2)} = \max(0, y_{\text{base}, h} + \hat{\delta}_h)$ are blended with $\widehat{W}_h^{(v1)}$ via grid search over $w_{v2, h} \in [0.50, 1.00]$, yielding calibrated weights $w_{v2, h} \in [0.900, 1.000]$ across the 24 horizons.

---

## 4. Continuous Window Evaluation Metrics

To avoid binary boolean accuracy inflation, [main.py](../borssele-digital-twin/backend/main.py) evaluates each queried horizon window $H \in \{1, 6, 12, 24\}$ using **Continuous Relative Wind Forecast Accuracy**:

$$
\text{Acc}_{\text{wind}}(H) = \frac{100}{H} \sum_{h=1}^{H} \max\!\left(0,\; 1 - \frac{\left|\widehat{W}_{100}(t+h) - W_{100}(t+h)\right|}{\max\!\left(W_{100}(t+h),\, \widehat{W}_{100}(t+h),\, 4.0\text{ m/s}\right)}\right)
$$

alongside RMSE, MAE, and exact 4-regime match rate (`regime_match_pct`).

---

## 5. Safe Fallback & Multi-Source Consensus Equations

1. **Independent Secondary NWP Safe Fallback (`degraded_fallback`):**
   [era5_updater.py](../borssele-digital-twin/backend/app/era5_updater.py) concurrently polls **MET Norway Locationforecast (`api.met.no`)** (extrapolated from `10 m` to `100 m` using `POWER_LAW_OFFSHORE_ALPHA = 0.11`) and **DWD ICON-Seamless (`icon_seamless`)** to attach `secondary_nwp_ws100`. [main.py](../borssele-digital-twin/backend/main.py) computes:

$$
\widehat{W}_{\text{NWP, sec}}(t+h) = 0.80\,\text{secondary\_nwp\_ws}_h + 0.20\,W_{100}(t)
$$

   When $\max(\widehat{W}_{\text{NWP, sec}}, \widehat{W}_{\text{ens}}) \ge 20.0\text{ m/s}$, `pred_fallback` adopts the conservative upper envelope $\max(\widehat{W}_{\text{NWP, sec}}, \widehat{W}_{\text{ens}})$ to protect turbine infrastructure and grid balance.

2. **Multi-Source Consensus Gate (`CONSENSUS_THRESHOLD_MS = 2.0 m/s`):**
   At each forecast step $h$, inter-source spread $S_h$ and consensus confidence $C_h$ across `pred_v2`, `pred_v1`, and `nwp_secondary` are:

$$
S_h = \max\!\left(\widehat{W}_{v2}, \widehat{W}_{v1}, \widehat{W}_{\text{NWP, sec}}\right) - \min\!\left(\widehat{W}_{v2}, \widehat{W}_{v1}, \widehat{W}_{\text{NWP, sec}}\right), \qquad C_h = \max\!\left(0,\; 1 - \frac{S_h}{\max(\widehat{W}_{\text{active}}, 6.0)}\right)
$$

   Each hour records `consensus_met = (S_h <= 2.0)`. Automatic fallback (`AUTO_FALLBACK_HIGH_WIND_CONSENSUS_DIVERGENCE`) triggers whenever forecast winds reach $\ge 20.0\text{ m/s}$ with window `consensus_agreement_pct < 70.0%`.
