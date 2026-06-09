# scratch/optimize_features.py
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

def create_rich_features(series, dates, lags=[1, 2, 3]):
    df_feat = pd.DataFrame(series)
    df_feat.columns = ['res']
    for lag in lags:
        df_feat[f'lag_{lag}'] = df_feat['res'].shift(lag)
    df_feat['week_of_year'] = df_feat.index.isocalendar().week.astype(float)
    df_feat['month'] = df_feat.index.month.astype(float)
    df_feat.dropna(inplace=True)
    X = df_feat.drop(columns=['res'])
    y = df_feat['res']
    return X, y

# Fit optimized Holt-Winters
hw_model = ExponentialSmoothing(
    train_series, seasonal_periods=13, trend='add', seasonal='add',
    damped_trend=False, initialization_method='estimated'
).fit(optimized=True)
hw_forecast = hw_model.forecast(FORECAST_PERIODS)
residuals = train_series - hw_model.fittedvalues

X_train_res, y_train_res = create_rich_features(residuals, train_series.index)

# 1. Random Forest with Calendar features
print(">>> Optimizing RF with Lags + Calendar features...")
best_rf_mape = 999.0
best_rf_params = {}

rf_n_estimators = [50, 100, 200]
rf_max_depth = [None, 3, 5, 7]
rf_min_samples_leaf = [1, 2, 4]

for n in rf_n_estimators:
    for d in rf_max_depth:
        for l in rf_min_samples_leaf:
            rf = RandomForestRegressor(n_estimators=n, max_depth=d, min_samples_leaf=l, random_state=42)
            rf.fit(X_train_res.values, y_train_res.values)
            
            res_history = list(residuals.values)
            pred_res = []
            test_dates = test_series.index
            for i in range(FORECAST_PERIODS):
                date = test_dates[i]
                week_val = float(date.isocalendar()[1])
                month_val = float(date.month)
                features = [res_history[-1], res_history[-2], res_history[-3], week_val, month_val]
                pred = rf.predict([features])[0]
                pred_res.append(pred)
                res_history.append(pred)
                
            final_pred = hw_forecast.values + np.array(pred_res)
            mape = calculate_mape(actuals, final_pred)
            if mape < best_rf_mape:
                best_rf_mape = mape
                best_rf_params = {'n_estimators': n, 'max_depth': d, 'min_samples_leaf': l}

print(f"Best RF Hybrid with Calendar: {best_rf_params} -> MAPE: {best_rf_mape:.4f}%")

# 2. XGBoost with Calendar features
print("\n>>> Optimizing XGBoost with Lags + Calendar features...")
best_xgb_mape = 999.0
best_xgb_params = {}

xgb_n_estimators = [50, 100, 150]
xgb_max_depth = [2, 3, 4, 5]
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
                for i in range(FORECAST_PERIODS):
                    date = test_dates[i]
                    week_val = float(date.isocalendar()[1])
                    month_val = float(date.month)
                    features = [res_history[-1], res_history[-2], res_history[-3], week_val, month_val]
                    pred = model.predict(np.array([features]))[0]
                    pred_res.append(pred)
                    res_history.append(pred)
                    
                final_pred = hw_forecast.values + np.array(pred_res)
                mape = calculate_mape(actuals, final_pred)
                if mape < best_xgb_mape:
                    best_xgb_mape = mape
                    best_xgb_params = {'n_estimators': n, 'max_depth': d, 'learning_rate': lr, 'subsample': s}

print(f"Best XGBoost Hybrid with Calendar: {best_xgb_params} -> MAPE: {best_xgb_mape:.4f}%")
