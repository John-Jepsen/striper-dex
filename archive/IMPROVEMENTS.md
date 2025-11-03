# Fishing Forecast Improvement Roadmap

## Current State
- ✅ 30+ years of NOAA SST data (1993-present)
- ✅ Random Forest model with temperature-focused features
- ✅ 7-day ahead forecasting
- ✅ Species-specific scoring based on temperature preferences

## High-Impact Improvements

### 1. **Integrate Real-Time Weather Data** 🌤️
**Why:** Ocean temps respond to air temperature, wind, solar radiation, and upwelling events.

**What to add:**
- NOAA NDBC buoy meteorological data (wind speed/direction, air temp, pressure)
- NOAA NWS forecasts for Monterey Bay
- Upwelling indices from PFEL/NOAA (critical for local SST changes)

**Expected impact:** +15-25% forecast accuracy for short-term (1-3 day) predictions

**Implementation:**
```python
# Add NDBC Station 46042 (Monterey Bay) met data
# Add upwelling index from NOAA PFEL
# Features: wind_speed, wind_direction, air_temp, pressure_trend, upwelling_index
```

---

### 2. **Add Tidal & Current Data** 🌊
**Why:** Fish behavior correlates strongly with tidal cycles and current strength.

**What to add:**
- NOAA CO-OPS tide predictions for Monterey Harbor
- Current predictions (if available)
- Moon phase (affects tides and fish feeding)

**Expected impact:** Better "when to fish" recommendations beyond just temperature

**Implementation:**
```python
# Features: tide_height, tide_direction (flood/ebb), current_speed
# Score modifier: higher scores for incoming tide + dawn/dusk
```

---

### 3. **Satellite SST & Chlorophyll** 🛰️
**Why:** Capture offshore dynamics, fronts, and productivity zones not visible from single buoy.

**What to add:**
- NOAA CoastWatch VIIRS/MODIS SST (daily, 1km resolution)
- Chlorophyll-a concentration (food chain indicator)
- SST gradients (thermal fronts = fish highways)

**Expected impact:** Identify offshore hotspots, predict local upwelling events

**Data sources:**
- NOAA CoastWatch ERDDAP: https://coastwatch.pfeg.noaa.gov/erddap/
- NASA OBPG for chlorophyll

---

### 4. **Multi-Model Ensemble** 📊
**Why:** Different models capture different patterns. Ensembles reduce overfitting.

**Current:** Single Random Forest

**Upgrade to:**
- Random Forest (current)
- Gradient Boosting (XGBoost/LightGBM) - better for sequential patterns
- LSTM/GRU neural network - captures long-term temporal dependencies
- Simple persistence model (baseline: "tomorrow = today")

**Combine via:**
- Weighted average (tune weights on validation set)
- Stacking (meta-model learns how to blend predictions)

**Expected impact:** +10-15% accuracy, more robust to regime shifts

---

### 5. **Validation & Uncertainty Quantification** 📈
**Current:** No train/test split, no confidence intervals

**Add:**
- Time-series cross-validation (don't leak future into past)
- Prediction intervals (e.g., 90% confidence: 59-62°F)
- Track forecast skill scores (MAE, RMSE) by lead time
- Backtesting: "If we ran this forecast 1 year ago, how good was it?"

**Why:** Know when to trust the model vs when uncertainty is too high

**Implementation:**
```python
# Use scikit-learn TimeSeriesSplit
# Quantile regression forests for prediction intervals
# Log metrics to MLflow or simple CSV
```

---

### 6. **Species Behavior Models** 🐟
**Current:** Simple temperature range scoring

**Upgrade to:**
- Historical catch data (if you track it): actual success vs conditions
- CDFW recreational fishing reports (public data)
- Depth preferences by species + temperature
- Feeding windows (dawn/dusk amplifiers)
- Seasonal migration patterns (e.g., salmon runs)

**Expected impact:** Much more accurate "what to target" recommendations

**Data collection:**
```python
# Start logging: date, species, count, location, conditions
# Build species-specific models: P(catch | temp, tide, wind, ...)
```

---

### 7. **Interactive Dashboard** 💻
**Current:** Terminal text output

**Upgrade to:**
- Web app (Streamlit/Plotly Dash)
- Live charts: temperature history + forecast + uncertainty bands
- Species scoring heatmap (day × species)
- Map view: buoy location, satellite SST overlay
- Mobile-friendly for on-the-water access

**Tools:**
- Streamlit (fastest): `pip install streamlit`
- Plotly for interactive charts
- Folium for maps

---

### 8. **Nowcast (Real-Time Updates)** ⚡
**Current:** Static daily forecast

**Add:**
- Check for new NOAA data every 6 hours
- Re-run forecast automatically
- Alert system: "Temps dropped 2°F overnight - upwelling event!"
- Webhook/SMS notifications for optimal conditions

**Implementation:**
```python
# Cron job or GitHub Actions to run pull_noaa_history.py + forecast
# Compare forecast vs actual, trigger alerts
```

---

### 9. **Spatial Expansion** 🗺️
**Current:** Single buoy (Monterey Harbor)

**Add multi-station forecasting:**
- NOAA 9413450 (Monterey Harbor) - current
- NOAA 9414290 (Santa Cruz)
- NOAA 9414750 (Alameda)
- MBARI M1 buoy (offshore, deeper water)

**Why:** Compare inshore vs offshore, pick best fishing zone

**UI:** Dropdown to select station, map with color-coded conditions

---

### 10. **Explainability** 🔍
**Current:** Black-box predictions

**Add:**
- Feature importance plots (which features drove today's forecast?)
- SHAP values: "Temp is 60.7°F *because* 7-day rolling mean was 60.5°F and upwelling index is neutral"
- Scenario analysis: "If wind shifts to north, expect -1°F change"

**Why:** Build trust, learn what actually matters for your local waters

**Tools:**
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_forecast)
shap.force_plot(...)  # Show driver breakdown
```

---

## Quick Wins (Implement This Week)

1. **Add confidence intervals** (4 hours)
   - Use Random Forest's per-tree predictions to estimate uncertainty
   - Display as "60.7°F ± 0.8°F (90% CI)"

2. **Track forecast accuracy** (2 hours)
   - Save predictions to CSV
   - Compare tomorrow's prediction vs actual daily update
   - Plot running MAE over time

3. **Add tide data** (3 hours)
   - NOAA CO-OPS tide API: https://api.tidesandcurrents.noaa.gov/
   - Add to species scoring: bonus for incoming tide

4. **Species tuning with local knowledge** (1 hour)
   - Adjust temperature ranges based on your actual catch logs
   - Add "peak season" modifiers

5. **Generate feature importance plot** (1 hour)
   - Add matplotlib bar chart of `model.feature_importances_`
   - Save alongside forecast

---

## Data Quality Improvements

1. **Gap filling:**
   - Current: Missing data gets forward-filled
   - Better: Use satellite SST or nearby buoys to interpolate gaps

2. **Outlier detection:**
   - Flag suspicious readings (sensor failures)
   - Use IQR or isolation forest

3. **Multi-source fusion:**
   - Blend NOAA buoy + satellite + MBARI
   - Weight by recency and reliability

---

## Next Steps (Priority Order)

**Phase 1 (This Month):**
1. Add confidence intervals ✓
2. Track forecast skill ✓
3. Integrate tide data ✓
4. Feature importance visualization ✓

**Phase 2 (Next Month):**
1. NDBC meteorological data
2. Upwelling index
3. Multi-model ensemble (add XGBoost)
4. Backtesting framework

**Phase 3 (Q1 2026):**
1. Satellite SST integration
2. Streamlit dashboard
3. Multi-station support
4. SHAP explainability

**Phase 4 (Future):**
1. Catch logging app
2. Species behavior models from real data
3. Mobile app with push notifications
4. Community sharing (compare forecasts with other anglers)

---

## Questions to Guide Next Steps

1. **What's your primary goal?**
   - Maximize catch rate → Focus on species models + real catch data
   - Plan trips in advance → Improve 7-day forecast accuracy
   - Daily "should I go today?" → Add real-time nowcasting

2. **What data can you collect?**
   - If you log catches → Build custom species models
   - If you want satellite data → Set up ERDDAP pipeline
   - If you want weather → Integrate NDBC

3. **How do you want to interact with it?**
   - Terminal is fine → Stick with CLI, add more detail
   - Want visuals → Build Streamlit dashboard
   - On the boat → Mobile-optimized web app

---

**Bottom line:** The foundation is solid. Biggest gains will come from:
1. **Weather/upwelling data** (short-term accuracy)
2. **Validation & uncertainty** (know when to trust it)
3. **Real catch logs** (species-specific tuning)
4. **Dashboard** (usability)
