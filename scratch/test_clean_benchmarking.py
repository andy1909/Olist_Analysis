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

# 1. Drop the incomplete last week (2018-09-02)
weekly_orders = weekly_orders.iloc[:-1]

# 2. Also drop the next-to-last week (2018-08-26) since it is also heavily truncated
weekly_orders = weekly_orders.iloc[:-1]

# 3. Cleanse the strike weeks (2018-05-27 and 2018-06-03) using average of neighboring normal weeks
strike_weeks = ['2018-05-27', '2018-06-03']
normal_avg = weekly_orders.loc[(weekly_orders.index >= '2018-04-29') & (weekly_orders.index <= '2018-05-20')].mean()
print(f"Normal avg computed: {normal_avg:.2f}")

weekly_orders_clean = weekly_orders.copy()
for week in strike_weeks:
    if week in weekly_orders_clean.index:
        weekly_orders_clean.loc[week] = normal_avg

print("Cleaned weekly orders tail:")
print(weekly_orders_clean.tail(15))

FORECAST_PERIODS = 11  # validation test window is now 11 weeks (ending 2018-08-19)
train_series = weekly_orders_clean.iloc[:-FORECAST_PERIODS]
test_series = weekly_orders_clean.iloc[-FORECAST_PERIODS:]

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

# Compute MAPEs
def compute_mape(act, pred):
    return np.mean(np.abs(act - pred) / act) * 100

print("\nModel MAPEs on Cleaned Series:")
print(f"Naive: {compute_mape(test_series.values, fc_naive):.2f}%")
print(f"Holt-Winters: {compute_mape(test_series.values, fc_hw.values):.2f}%")
print(f"SARIMA: {compute_mape(test_series.values, fc_sarima.values):.2f}%")
print(f"Random Forest: {compute_mape(test_series.values, fc_rf):.2f}%")
print(f"XGBoost: {compute_mape(test_series.values, fc_xgb):.2f}%")
print(f"LightGBM: {compute_mape(test_series.values, fc_lgb):.2f}%")

print("\nTest Set Values vs HW vs SARIMA:")
comp = pd.DataFrame({
    'Actual': test_series.values,
    'HW': fc_hw.values,
    'SARIMA': fc_sarima.values
}, index=test_series.index)
print(comp)
