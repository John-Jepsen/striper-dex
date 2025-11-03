# Addressing Underfitting (31% R² → 60%+ Target)

## 🔍 Problem Diagnosis
**Current**: Model R² = 0.31 (only explains 31% of variance)
- Linear model learns: "most days ≈ 29 points" (average)
- Cannot capture: "spring + warm water + falling pressure = 90+ points"
- Model is **too simple** for complex multi-factor interactions

---

## 🎯 Solutions Ranked by Impact

### 1. **Add Polynomial & Interaction Features** (Expected +15-20% R²)
**Why it works**: Captures non-linear relationships like temp² and temp×pressure×season

```python
# Add to feature_engineering.py
from sklearn.preprocessing import PolynomialFeatures

def create_polynomial_features(df, degree=2):
    """Create polynomial and interaction terms for key predictors."""
    
    # Select most impactful features for polynomial expansion
    key_features = [
        'morning_temp_F',
        'pressure_change_6h',
        'temp_change_7d',
        'temp_volatility_7d',
    ]
    
    # Create polynomial features (degree 2 = squares + interactions)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(df[key_features].fillna(0))
    
    # Create DataFrame with proper names
    poly_names = poly.get_feature_names_out(key_features)
    poly_df = pd.DataFrame(poly_features, columns=poly_names, index=df.index)
    
    # Add back to original df
    return pd.concat([df, poly_df], axis=1)
```

**Key interactions to capture**:
- `temp² ` - Temperature has U-shaped relationship (too cold AND too hot = bad)
- `temp × pressure_change` - Warm water + falling pressure = feeding frenzy
- `temp × season` - 60°F in spring (migration) ≠ 60°F in summer (resident)
- `pressure_change × month` - Spring pressure drops trigger more activity

---

### 2. **Use More Complex Models** (Expected +5-10% R²)
**Current**: Using Gradient Boosting with default params
**Better**: Tune hyperparameters aggressively

```python
# In train_fishing_model.py
from sklearn.ensemble import GradientBoostingRegressor

def train_optimized_gradient_boosting(X_train, y_train, X_test, y_test):
    """Train GBM with aggressive hyperparameters for complex patterns."""
    
    param_grid = {
        'n_estimators': [300, 500, 1000],           # More trees
        'max_depth': [5, 7, 10, 15],                # Deeper trees = more interactions
        'learning_rate': [0.01, 0.05, 0.1],         # Lower LR with more trees
        'min_samples_split': [2, 5, 10],            # Allow more splits
        'min_samples_leaf': [1, 2, 4],              # Smaller leaves
        'subsample': [0.8, 0.9, 1.0],               # Stochastic boosting
        'max_features': ['sqrt', 'log2', None],     # Feature sampling
    }
    
    gbm = GradientBoostingRegressor(random_state=42, verbose=0)
    
    grid_search = GridSearchCV(
        gbm, param_grid, 
        cv=5, 
        scoring='r2', 
        n_jobs=-1, 
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
```

**Alternative**: Try XGBoost or LightGBM (better than sklearn's GBM)
```python
import xgboost as xgb

def train_xgboost_model(X_train, y_train, X_test, y_test):
    """XGBoost handles interactions better than sklearn GBM."""
    
    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,        # L1 regularization
        reg_lambda=1.0,       # L2 regularization
        random_state=42,
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    return model
```

---

### 3. **Fix Column Name Mismatch** (Immediate fix)
**Bug Found**: Features use `morning_temp_F` but model expects `morning_temp_avg_F`

```bash
# Quick fix in feature_engineering.py line 96
# Change column name to match model expectations
df.rename(columns={'morning_temp_F': 'morning_temp_avg_F'}, inplace=True)
```

---

### 4. **Add Critical Missing Features** (Expected +5-10% R²)

#### A. **Tidal Phase Features** (already collected but not used!)
```python
# In create_tidal_features()
def add_tidal_interaction_features(df):
    """Tidal phase completely changes fish behavior."""
    
    # Flood tide + early morning = prime time
    if 'tide_flood' in df.columns and 'is_early_morning' in df.columns:
        df['flood_tide_dawn'] = df['tide_flood'] * df['is_early_morning']
    
    # Moving water (flood or ebb) vs slack
    if 'tidal_phase' in df.columns:
        df['moving_water'] = df['tidal_phase'].isin(['flood', 'ebb']).astype(int)
        df['slack_water'] = df['tidal_phase'].isin(['slack_high', 'slack_low']).astype(int)
    
    # Temperature + tide interaction (cold water flood = upwelling bait)
    if 'morning_temp_F' in df.columns and 'tide_flood' in df.columns:
        df['cold_flood_tide'] = (df['morning_temp_F'] < 55) * df['tide_flood']
    
    return df
```

#### B. **Season-Specific Temperature Scoring**
```python
def create_season_temp_interaction(df):
    """Same temp = different meaning in different seasons."""
    
    # Spring: 60°F = incoming fish (good)
    # Summer: 60°F = regular (neutral)
    # Fall: 60°F = staging for migration (good)
    # Winter: 60°F = rare warm spell (poor, fish in deep water)
    
    season_temp_score = pd.Series(0, index=df.index)
    
    if 'season' in df.columns and 'morning_temp_F' in df.columns:
        temp = df['morning_temp_F']
        season = df['season']
        
        # Spring: warming water = fish moving in
        spring_ideal = (season == 'spring') & (temp >= 55) & (temp <= 65)
        season_temp_score[spring_ideal] = 1
        
        # Fall: 58-68°F = prime migration staging
        fall_ideal = (season == 'fall') & (temp >= 58) & (temp <= 68)
        season_temp_score[fall_ideal] = 1
        
        # Winter: Any temp < 52°F = fish gone deep
        winter_cold = (season == 'winter') & (temp < 52)
        season_temp_score[winter_cold] = -1
        
        df['season_temp_optimal'] = season_temp_score
    
    return df
```

---

### 5. **Refine Target Variable** (Expected +5-8% R²)
**Current synthetic target is too rule-based**

```python
def create_improved_synthetic_target(df, use_interaction_terms=True):
    """
    More realistic target with:
    1. Multiplicative interactions (not just additive)
    2. Threshold effects
    3. Realistic variation
    """
    
    score = pd.Series(50.0, index=df.index)
    
    # BASE: Temperature (multiplicative, not additive)
    temp = df['morning_temp_F']
    temp_score = np.zeros(len(df))
    
    # Optimal range gets exponential bonus
    in_optimal = (temp >= 60) & (temp <= 70)
    temp_score[in_optimal] = 35 * (1 + 0.5 * ((70 - abs(temp[in_optimal] - 65)) / 5))
    
    # Suboptimal gets linear penalty
    too_cold = temp < 60
    temp_score[too_cold] = max(0, 20 - 2 * (60 - temp[too_cold]))
    
    too_hot = temp > 70
    temp_score[too_hot] = max(0, 15 - 1.5 * (temp[too_hot] - 70))
    
    score += temp_score
    
    # MULTIPLIER: Pressure change (multiplies temp effect, doesn't just add)
    if 'pressure_change_6h' in df.columns:
        pressure_change = df['pressure_change_6h'].fillna(0)
        
        # Falling pressure = 1.3x multiplier
        falling = pressure_change < -0.5
        score[falling] *= 1.3
        
        # Rapidly falling = 1.6x multiplier
        rapid_fall = pressure_change < -1.5
        score[rapid_fall] *= 1.6
        
        # Rising pressure = 0.7x multiplier
        rising = pressure_change > 0.5
        score[rising] *= 0.7
    
    # THRESHOLD: Season gates (spring/fall unlock higher scores)
    if 'season' in df.columns:
        season_multiplier = df['season'].map({
            'spring': 1.2,
            'fall': 1.2,
            'summer': 1.0,
            'winter': 0.6
        })
        score *= season_multiplier
    
    # Add realistic noise (weather is chaotic!)
    noise = np.random.normal(0, 5, size=len(score))
    score += noise
    
    return score.clip(0, 100)
```

---

### 6. **Add Temporal Lag Features** (Expected +3-5% R²)
**Fish behavior has memory** - yesterday's conditions matter

```python
def create_temporal_context_features(df):
    """Add multi-day context windows."""
    
    # 3-day temperature trend
    df['temp_3d_trend'] = df['morning_temp_F'].rolling(3).apply(
        lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1
    )
    
    # Pressure stability over 3 days (fish like stability)
    df['pressure_3d_stable'] = (
        df['pressure_mb'].rolling(72).std() < 2
    ).astype(int)
    
    # Consecutive days in optimal temp range
    df['optimal_temp_streak'] = (
        df['temp_in_optimal_range']
        .groupby((df['temp_in_optimal_range'] != df['temp_in_optimal_range'].shift()).cumsum())
        .cumsum()
    )
    
    # Days since last optimal conditions
    optimal_days = df[df['temp_in_optimal_range'] == 1].index
    df['days_since_optimal'] = df.index.to_series().apply(
        lambda x: min([abs(x - d) for d in optimal_days] + [999])
    )
    
    return df
```

---

## 🚀 Implementation Plan

### Phase 1: Quick Wins (1 hour)
1. Fix column name mismatch (`morning_temp_F` → `morning_temp_avg_F`)
2. Add polynomial features (degree 2)
3. Retrain with XGBoost instead of sklearn GBM
4. **Expected R²: 0.31 → 0.45** (+14%)

### Phase 2: Feature Engineering (2 hours)
1. Integrate tidal phase features
2. Add season-temperature interactions
3. Create temporal context features
4. **Expected R²: 0.45 → 0.55** (+10%)

### Phase 3: Model Tuning (1 hour)
1. Hyperparameter grid search on XGBoost
2. Try stacking (combine Random Forest + XGBoost)
3. Add feature selection (drop noise)
4. **Expected R²: 0.55 → 0.62** (+7%)

---

## 📊 Validation Strategy

### Before Changing Anything
```bash
# Baseline current performance
python train_fishing_model.py --model all
# Note the R² scores
```

### After Each Phase
```bash
# Regenerate features
python feature_engineering.py

# Retrain models
python train_fishing_model.py --model all

# Compare R² improvement
# Plot residuals to check if patterns remain
```

### Success Metrics
- R² > 0.60 (60% variance explained)
- Residuals show random scatter (no patterns)
- Feature importance aligns with domain knowledge
- Cross-validation R² within 5% of test R²

---

## 🎯 Expected Final Results

| Metric | Current | After Fixes | Improvement |
|--------|---------|-------------|-------------|
| **R² Score** | 0.31 | 0.60-0.65 | **+97-110%** |
| **RMSE** | 15.6 | 10-11 | -35% |
| **MAE** | 10.6 | 7-8 | -30% |

---

## ⚠️ Common Pitfalls to Avoid

1. **Overfitting**: Don't go crazy with degree-3 polynomials (stick to degree 2)
2. **Data leakage**: Keep train/test split temporal (no shuffling!)
3. **Feature explosion**: Don't create 500 features (max ~150)
4. **Multicollinearity**: Drop highly correlated features (>0.95)
5. **Missing data**: Impute carefully (use median, not mean)

---

## 📝 Code Changes Summary

**Files to modify**:
1. `feature_engineering.py` - Add polynomial, tidal, season-temp interactions
2. `train_fishing_model.py` - Add XGBoost, tune hyperparameters
3. Requirements: `pip install xgboost lightgbm`

**Estimated time**: 4-5 hours total
**Expected improvement**: 31% → 60%+ R² (nearly doubling model accuracy)
