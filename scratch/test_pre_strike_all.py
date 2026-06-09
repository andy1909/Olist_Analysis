import os
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
master_path = os.path.join(PROCESSED_DATA_DIR, 'Master_Logistics_Data.csv')

df = pd.read_csv(master_path, parse_dates=['order_purchase_timestamp'])
df_ts = df.set_index('order_purchase_timestamp')
weekly_orders = df_ts.resample('W')['order_id'].nunique().fillna(0)
weekly_orders = weekly_orders[weekly_orders.index >= '2017-01-01']
weekly_orders = weekly_orders.asfreq('W').fillna(0).astype(float)

# Slice to end on 2018-05-20
weekly_orders = weekly_orders[weekly_orders.index <= '2018-05-20']

FORECAST_PERIODS = 12
train_series = weekly_orders.iloc[:-FORECAST_PERIODS]
test_series = weekly_orders.iloc[-FORECAST_PERIODS:]

# Naive
naive_val = train_series.iloc[-1]
fc_naive = np.full(FORECAST_PERIODS, naive_val)

# HW
model_hw = ExponentialSmoothing(
    train_series, seasonal_periods=12, trend='add', seasonal='add',
    damped_trend=True, initialization_method='estimated'
).fit(optimized=True)
fc_hw = model_hw.forecast(FORECAST_PERIODS)

# SARIMA
model_sarima = SARIMAX(train_series, order=(1,1,1), seasonal_order=(1,1,1,12), 
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
fc_sarima = model_sarima.forecast(FORECAST_PERIODS)

# ML Lags features helper
def create_lag_features(series, lags=[1, 2, 3]):
    df = pd.DataFrame(series)
    df.columns = ['y']
    for lag in lags:
        df[f'lag_{lag}'] = df['y'].shift(lag)
    df.dropna(inplace=True)
    X = df.drop(columns=['y'])
    y = df['y']
    return X, y

X_train, y_train = create_lag_features(train_series)

# RF
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
history = list(train_series.values)
fc_rf = []
for _ in range(FORECAST_PERIODS):
    lags = [history[-1], history[-2], history[-3]]
    pred = rf.predict([lags])[0]
    fc_rf.append(pred)
    history.append(pred)

# XGBoost
xg = xgb.XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
xg.fit(X_train, y_train)
history = list(train_series.values)
fc_xgb = []
for _ in range(FORECAST_PERIODS):
    lags = [history[-1], history[-2], history[-3]]
    pred = xg.predict(np.array([lags]))[0]
    fc_xgb.append(pred)
    history.append(pred)

# LightGBM
lgbm = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
lgbm.fit(X_train, y_train)
history = list(train_series.values)
fc_lgb = []
for _ in range(FORECAST_PERIODS):
    lags = [history[-1], history[-2], history[-3]]
    pred = lgbm.predict(np.array([lags]))[0]
    fc_lgb.append(pred)
    history.append(pred)

def compute_mape(act, pred):
    return np.mean(np.abs(act - pred) / act) * 100

print("All Model MAPEs on Pre-Strike 12-week test set:")
print(f"Naive: {compute_mape(test_series.values, fc_naive):.2f}%")
print(f"Holt-Winters: {compute_mape(test_series.values, fc_hw.values):.2f}%")
print(f"SARIMA: {compute_mape(test_series.values, fc_sarima.values):.2f}%")
print(f"Random Forest: {compute_mape(test_series.values, fc_rf):.2f}%")
print(f"XGBoost: {compute_mape(test_series.values, fc_xgb):.2f}%")
print(f"LightGBM: {compute_mape(test_series.values, fc_lgb):.2f}%")
