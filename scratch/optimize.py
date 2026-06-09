# scratch/optimize.py
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
master_path = os.path.join(BASE_DIR, 'data', 'processed', 'Master_Logistics_Data.csv')

df = pd.read_csv(master_path, parse_dates=['order_purchase_timestamp'])
df_ts = df.set_index('order_purchase_timestamp')
weekly_orders = df_ts.resample('W')['order_id'].nunique().fillna(0)
weekly_orders = weekly_orders[weekly_orders.index >= '2017-01-01']
weekly_orders = weekly_orders.asfreq('W').fillna(0).astype(float)
weekly_orders = weekly_orders[weekly_orders.index <= '2018-05-20']

FORECAST_PERIODS = 12
train_series = weekly_orders.iloc[:-FORECAST_PERIODS]
test_series = weekly_orders.iloc[-FORECAST_PERIODS:]
actuals = test_series.values

def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def create_lag_features(series, lags=[1, 2, 3]):
    df_lag = pd.DataFrame(series)
    df_lag.columns = ['y']
    for lag in lags:
        df_lag[f'lag_{lag}'] = df_lag['y'].shift(lag)
    df_lag.dropna(inplace=True)
    X = df_lag.drop(columns=['y'])
    y = df_lag['y']
    return X, y

# 1. Grid search Holt-Winters base parameters
print(">>> Optimizing Holt-Winters...")
best_hw_mape = 999.0
best_hw_params = {}
best_hw_forecast = None
best_hw_fitted = None

trend_opts = ['add', 'mul', None]
seasonal_opts = ['add', 'mul', None]
damped_opts = [True, False]
seasonal_periods_opts = [4, 12, 13]

for t in trend_opts:
    for s in seasonal_opts:
        for d in damped_opts:
            for p in seasonal_periods_opts:
                if t is None and d:
                    continue  # Damped trend requires a trend
                if s is None and p:
                    continue  # Seasonal periods requires seasonal type
                try:
                    hw = ExponentialSmoothing(
                        train_series, seasonal_periods=p, trend=t, seasonal=s,
                        damped_trend=d, initialization_method='estimated'
                    ).fit(optimized=True)
                    pred = hw.forecast(FORECAST_PERIODS)
                    mape = calculate_mape(actuals, pred.values)
                    if mape < best_hw_mape:
                        best_hw_mape = mape
                        best_hw_params = {'trend': t, 'seasonal': s, 'damped': d, 'seasonal_periods': p}
                        best_hw_forecast = pred.values
                        best_hw_fitted = hw.fittedvalues
                except:
                    continue

print(f"Best Holt-Winters params: {best_hw_params} -> MAPE: {best_hw_mape:.4f}%")

# Base HW residual
residuals = train_series - best_hw_fitted
lags_list = [1, 2, 3]
X_train_res, y_train_res = create_lag_features(residuals, lags=lags_list)

# 2. Grid search Random Forest Regressor
print("\n>>> Optimizing Random Forest Regressor on residuals...")
best_rf_mape = 999.0
best_rf_params = {}
best_rf_pred = None

rf_n_estimators = [50, 100, 200, 300]
rf_max_depth = [None, 3, 5, 7, 10]
rf_min_samples_leaf = [1, 2, 4, 6]

for n in rf_n_estimators:
    for d in rf_max_depth:
        for l in rf_min_samples_leaf:
            rf = RandomForestRegressor(n_estimators=n, max_depth=d, min_samples_leaf=l, random_state=42)
            rf.fit(X_train_res.values, y_train_res.values)
            
            res_history = list(residuals.values)
            pred_res = []
            for _ in range(FORECAST_PERIODS):
                features = [res_history[-lag] for lag in lags_list]
                pred = rf.predict([features])[0]
                pred_res.append(pred)
                res_history.append(pred)
                
            final_pred = best_hw_forecast + np.array(pred_res)
            mape = calculate_mape(actuals, final_pred)
            if mape < best_rf_mape:
                best_rf_mape = mape
                best_rf_params = {'n_estimators': n, 'max_depth': d, 'min_samples_leaf': l}
                best_rf_pred = final_pred

print(f"Best RF Hybrid params: {best_rf_params} -> MAPE: {best_rf_mape:.4f}%")

# 3. Grid search XGBoost
print("\n>>> Optimizing XGBoost on residuals...")
best_xgb_mape = 999.0
best_xgb_params = {}
best_xgb_pred = None

xgb_n_estimators = [50, 100, 150, 200]
xgb_max_depth = [2, 3, 4, 5, 6]
xgb_learning_rate = [0.01, 0.05, 0.1, 0.2]
xgb_subsample = [0.6, 0.8, 1.0]

for n in xgb_n_estimators:
    for d in xgb_max_depth:
        for lr in xgb_learning_rate:
            for s in xgb_subsample:
                model = xgb.XGBRegressor(n_estimators=n, max_depth=d, learning_rate=lr, subsample=s, random_state=42, objective='reg:squarederror')
                model.fit(X_train_res.values, y_train_res.values)
                
                res_history = list(residuals.values)
                pred_res = []
                for _ in range(FORECAST_PERIODS):
                    features = [res_history[-lag] for lag in lags_list]
                    pred = model.predict(np.array([features]))[0]
                    pred_res.append(pred)
                    res_history.append(pred)
                    
                final_pred = best_hw_forecast + np.array(pred_res)
                mape = calculate_mape(actuals, final_pred)
                if mape < best_xgb_mape:
                    best_xgb_mape = mape
                    best_xgb_params = {'n_estimators': n, 'max_depth': d, 'learning_rate': lr, 'subsample': s}
                    best_xgb_pred = final_pred

print(f"Best XGBoost Hybrid params: {best_xgb_params} -> MAPE: {best_xgb_mape:.4f}%")

# 4. Grid search LightGBM
print("\n>>> Optimizing LightGBM on residuals...")
best_lgb_mape = 999.0
best_lgb_params = {}
best_lgb_pred = None

lgb_n_estimators = [50, 100, 150]
lgb_max_depth = [-1, 2, 3, 4, 5]
lgb_learning_rate = [0.01, 0.05, 0.1, 0.2]
lgb_min_child_samples = [2, 5, 8, 10]

for n in lgb_n_estimators:
    for d in lgb_max_depth:
        for lr in lgb_learning_rate:
            for m in lgb_min_child_samples:
                model = lgb.LGBMRegressor(n_estimators=n, max_depth=d, learning_rate=lr, min_child_samples=m, random_state=42, verbose=-1)
                model.fit(X_train_res.values, y_train_res.values)
                
                res_history = list(residuals.values)
                pred_res = []
                for _ in range(FORECAST_PERIODS):
                    features = [res_history[-lag] for lag in lags_list]
                    pred = model.predict(np.array([features]))[0]
                    pred_res.append(pred)
                    res_history.append(pred)
                    
                final_pred = best_hw_forecast + np.array(pred_res)
                mape = calculate_mape(actuals, final_pred)
                if mape < best_lgb_mape:
                    best_lgb_mape = mape
                    best_lgb_params = {'n_estimators': n, 'max_depth': d, 'learning_rate': lr, 'min_child_samples': m}
                    best_lgb_pred = final_pred

print(f"Best LightGBM Hybrid params: {best_lgb_params} -> MAPE: {best_lgb_mape:.4f}%")
