# Urban Groundwater Stress Analysis
### Predictive Modelling and Aquifer Zone Classification Under High Urban Abstraction (2005–2024)

---

## Overview

This project investigates the long-term decline of urban groundwater systems under sustained high-abstraction conditions. Using a daily time series dataset spanning **7,300 observations from 2005 to 2024**, it combines machine learning regression to predict groundwater levels (GWL) with unsupervised KMeans clustering to classify aquifer stress regimes based on hydrological, meteorological, and water quality indicators.

The analysis documents a **10.7 m deepening of groundwater levels** over 20 years, driven by a near-doubling of daily pumping rates, and identifies three distinct aquifer stress zones to support water resource management decisions.

---

## Key Findings

| Finding | Value |
|---|---|
| Mean GWL 2005–2009 | 9.6 mbgl |
| Mean GWL 2020–2024 | 20.3 mbgl |
| Total aquifer deepening | **10.7 m** |
| Mean pumping 2005–2009 | 31.2 ML/d |
| Mean pumping 2020–2024 | 56.3 ML/d |
| Mean EC 2005–2009 | 463 µS/cm |
| Mean EC 2020–2024 | 538 µS/cm |
| Best model R² | **0.942 (Linear Regression)** |
| Best model RMSE | **0.360 m** |
| High stress days 2020–2024 | **1,722** |

---

## Dataset

A synthetic dataset containimg **7,300 daily records** from **2005-01-01 to 2024-12-26** with **17 columns**:

| Feature | Unit | Description |
|---|---|---|
| `Date` | — | Daily timestamp |
| `GWL_mbgl` | m | Groundwater level (metres below ground level) — **target variable** |
| `Pumping_Rate_MLd` | ML/d | Daily groundwater abstraction rate |
| `Rainfall_mm` | mm | Daily rainfall |
| `Temperature_C` | °C | Daily air temperature |
| `Effective_Recharge_mm` | mm | Estimated daily aquifer recharge |
| `EC_uScm` | µS/cm | Electrical conductivity — water quality indicator |
| `Nitrate_mgL` | mg/L | Nitrate concentration |
| `GWL_30d_avg` | m | 30-day moving average of GWL |
| `Pumping_7d_avg` | ML/d | 7-day moving average of pumping rate |
| `Rain_30d_sum` | mm | 30-day cumulative rainfall |
| `Month` | — | Calendar month |
| `DayOfYear` | — | Day of year (1–366) |
| `Weekday` | — | Day of week |
| `Year` | — | Calendar year |

> **Note:** The dataset file is not included in this repository. Upload your CSV when prompted in Google Colab.

---

## Methodology

### 1. Exploratory Data Analysis
- 4-panel time series plot of GWL (with 30-day moving average), rainfall, pumping rate, and temperature
- Pearson correlation matrix of 8 key hydrometeorological variables
- Annual mean GWL and pumping rate trend analysis
- Monthly seasonality plot with confidence bands
- Scatter plot of pumping rate vs GWL across four time periods (2005–2024)

**Key correlations with GWL:**
```
Pumping_Rate_MLd      r = +0.840   (strongest driver)
EC_uScm               r = +0.757
Nitrate_mgL           r = +0.454
Temperature_C         r = +0.272
Rain_30d_sum          r = -0.264
Rainfall_mm           r = -0.114
Effective_Recharge_mm r = -0.114
```

### 2. Feature Engineering

Seven features were engineered to capture temporal dynamics and abstraction stress. Some features  that caused **data leakage** (`GWL_lag1`, `GWL_diff_1d`, `GWL_diff_7d`) were deliberately excluded:

| Feature | Description 
|---|---|---|
| `Time_Index` | Days since 2005-01-01 — deployment-safe trend anchor
| `GWL_lag7` | GWL 7 days prior 
| `GWL_lag30` | GWL 30 days prior 
| `Pumping_lag1` | Pumping rate 1 day prior 
| `Rain_lag7_sum` | 7-day rolling rainfall sum 
| `Pumping_x_Temp` | Pumping × Temperature interaction term 
| `Cum_Deficit` | Cumulative abstraction deficit 

> `Time_Index` is computed as `(Date - min_date).dt.days` rather than a row counter, making it reproducible and deployment-safe for any future date.

### 3. Train / Test Split

An 80/20 chronological split was used (no shuffling — temporal order preserved):

```
Training set: 5,816 samples  (approx. 2005–2020)
Test set:     1,454 samples  (approx. 2020–2024)
Features:     18
```

`StandardScaler` was fitted **only on training data** and applied to the test set to prevent data leakage.

### 4. Predictive Modelling

Four regression models were trained and evaluated:

| Model | Scaling | Key Hyperparameters |
|---|---|---|
| Linear Regression | StandardScaler | Default |
| Lasso (α=0.01) | StandardScaler | `alpha=0.01`, `max_iter=5000` |
| Random Forest | None | `n_estimators=300`, `max_depth=20`, `min_samples_leaf=5` |
| Gradient Boosting | None | `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8` |

**Results:**

| Model | MAE (m) | RMSE (m) | R² |
|---|---|---|---|
| **Linear Regression** | 0.2892 | 0.3603 | **0.94228** |
| **Lasso (α=0.01)** | 0.2938 | 0.3673 | **0.94000** |
| Random Forest | 0.9020 | 1.1921 | 0.36791 |
| Gradient Boosting | 1.2679 | 1.5420 | -0.05755 |

**Why linear models outperform tree-based models:** GWL exhibits a strong, monotonic long-term declining trend. Tree-based models cannot extrapolate beyond the value range seen during training — when the test period (2020–2024) contains GWL values deeper than any observed during training, RF and GB predictions plateau while observed GWL continues to decline. This is a known limitation of decision-tree-based methods on trending time series and is explicitly annotated on the prediction plots.

### 5. Feature Importance

Standardised coefficients from the best-performing Linear Regression model were used to rank feature contributions:

| Rank | Feature | Interpretation |
|---|---|---|
| 1 | `GWL_lag7` | GWL has strong 7-day temporal autocorrelation |
| 2 | `Time_Index` | Captures the 20-year monotonic declining trend |
| 3 | `Effective_Recharge_mm` | Direct aquifer replenishment — counteracts decline |
| 4 | `Temperature_C` | Drives evapotranspiration and urban demand |
| 5 | `EC_uScm` | Water quality deterioration signal |
| 6 | `GWL_lag30` | Longer-term aquifer memory |
| 7 | `Pumping_Rate_MLd` | Daily abstraction pressure |

> `Pumping_Rate_MLd` ranks lower than its raw correlation (r=+0.840) because its long-term cumulative effect is captured by `Cum_Deficit` and `Time_Index` in the multivariate model — a known behaviour in correlated feature sets.

### 6. Aquifer Zone Classification (KMeans Clustering)

Groundwater monitoring records were clustered using KMeans on five features:
`GWL_mbgl`, `Pumping_Rate_MLd`, `Rainfall_mm`, `EC_uScm`, `Temperature_C`

- Features standardised with `StandardScaler` prior to clustering
- Optimal k selected using the **Elbow Method** and **Silhouette Score**
- Silhouette scores: k=2 (~0.335), **k=3 (~0.323)**, k=4 (~0.288) — sharp drop after k=3 confirmed optimal clusters
- Final model: **k=3**, clusters labelled by ascending mean GWL

**Cluster profiles (mean values):**

| Stress Regime | GWL (mbgl) | Pumping (ML/d) | Rainfall (mm) | EC (µS/cm) | Temp (°C) |
|---|---|---|---|---|---|
| Low Stress | 11.51 | 35.30 | 1.83 | 473.63 | 11.23 |
| Moderate Stress | 13.04 | 40.51 | 10.54 | 497.83 | 5.87 |
| High Stress | 18.78 | 52.84 | 1.97 | 527.79 | 11.95 |

### 7. PCA Visualisation of Clusters

Principal Component Analysis (2 components) was applied to visualise cluster separation in reduced-dimensional space:

```
PC1 — Abstraction-Quality Axis (explains pumping/GWL/EC variance):
  GWL_mbgl          0.609
  Pumping_Rate_MLd  0.565
  EC_uScm           0.529

PC2 — Seasonal/Recharge Axis (explains temperature/rainfall variance):
  Temperature_C     0.690
  Rainfall_mm      -0.677
```

PC1 separates stress regimes along the abstraction–GWL–quality axis. PC2 captures seasonal variability — Moderate Stress clusters at lower temperatures and higher rainfall, corresponding to winter recharge periods.

---

## Stress Regime Shift (2005–2024)

```
High stress days 2005–2009:     0
High stress days 2020–2024:  1,722
```

No days were classified as High Stress in the earliest period. By 2020–2024, nearly every day falls under the High Stress regime — a stark indicator of progressive aquifer degradation under escalating urban abstraction.

---

## Technologies

```
Python 3.x
pandas · numpy · matplotlib
scikit-learn:
  LinearRegression · Lasso · RandomForestRegressor · GradientBoostingRegressor
  KMeans · PCA · StandardScaler · silhouette_score
  mean_absolute_error · mean_squared_error · r2_score
```

---

## Repository Structure

```
├── Groundwater_Urban_Abstraction.ipynb     # Full analysis notebook
├── README.md                               # Project documentation
└── data/
    └── groundwater_urban_abstraction.csv   # Dataset
```

---

## Deployment Notes

The `Time_Index` feature is computed relative to the dataset start date:

```python
df['Time_Index'] = (df['Date'] - df['Date'].min()).dt.days
# Reference date: 2005-01-01
```

For inference on new data after deployment, always use the **same reference date**:

```python
reference_date = pd.Timestamp('2005-01-01')
new_data['Time_Index'] = (new_data['Date'] - reference_date).dt.days
```

Save the fitted `StandardScaler` and trained `LinearRegression` model using `joblib` for consistent production inference:

```python
import joblib
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(models['Linear Regression'], 'gwl_model.pkl')
```

---

## Limitations

- Tree-based models (RF, GB) underperform due to the long-term linear declining trend in GWL — they cannot extrapolate beyond the training value range; this is discussed and annotated in the notebook
- Lasso performance is sensitive to the choice of α; `LassoCV` could be used for automated regularisation tuning
- The dataset is simulated for demonstration purposes. The results reflect the structure of the generated data rather than a specific real-world aquifer

---

## Author

Built as part of an urban hydrogeology and applied machine learning project.
Feel free to fork, raise issues, or contribute.

---

## License

Open for academic and research use.
