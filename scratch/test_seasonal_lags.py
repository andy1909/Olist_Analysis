import os
import pandas as pd
import numpy as np
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

# Drop last two weeks
weekly_orders = weekly_orders.iloc[:-2]

# Cleanse strike weeks
strike_weeks = ['2018-05-27', '2018-06-03']
normal_avg = weekly_orders.loc[(weekly_orders.index >= '2018-04-29') & (weekly_orders.index <= '2018-05-20')].mean()
weekly_orders_clean = weekly_orders.copy()
for week in strike_weeks:
    if week in weekly_orders_clean.index:
        weekly_orders_clean.loc[week] = normal_avg

FORECAST_PERIODS = 11
train_series = weekly_orders_clean.iloc[:-FORECAST_PERIODS]
test_series = weekly_orders_clean.iloc[-FORECAST_PERIODS:]

# Create lag features including lag 12
def create_seasonal_lag_features(series, lags=[1, 2, 3, 12]):
    df = pd.DataFrame(series)
    df.columns = ['y']
    for lag in lags:
        df[f'lag_{lag}'] = df['y'].shift(lag)
    df.dropna(inplace=True)
    X = df.drop(columns=['y'])
    y = df['y']
    return X, y

X_train, y_train = create_seasonal_lag_features(train_series)

# RF
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# XGBoost
xg = xgb.XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
xg.fit(X_train, y_train)

# LightGBM
lgbm = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
lgbm.fit(X_train, y_train)

# Recursive forecast with seasonal lags
def recursive_forecast(model, train_series, periods):
    history = list(train_series.values)
    forecast = []
    for i in range(periods):
        # We need lags: t-1, t-2, t-3, t-12
        # Relative to history: history[-1] is t-1, history[-12] is t-12
        lags = [history[-1], history[-2], history[-3], history[-12]]
        pred = model.predict(np.array([lags]))[0]
        forecast.append(pred)
        history.append(pred)
    return forecast

fc_rf = recursive_forecast(rf, train_series, FORECAST_PERIODS)
fc_xgb = recursive_forecast(xg, train_series, FORECAST_PERIODS)
fc_lgb = recursive_forecast(lgbm, train_series, FORECAST_PERIODS)

def mape(act, pred):
    return np.mean(np.abs(act - pred) / act) * 100

print(f"RF (with lag 12) MAPE: {mape(test_series.values, fc_rf):.2f}%")
print(f"XGBoost (with lag 12) MAPE: {mape(test_series.values, fc_xgb):.2f}%")
print(f"LightGBM (with lag 12) MAPE: {mape(test_series.values, fc_lgb):.2f}%")

print("\nActual vs RF Forecast vs LightGBM Forecast:")
comp = pd.DataFrame({
    'Actual': test_series.values,
    'RF_Lag12': fc_rf,
    'LGB_Lag12': fc_lgb
}, index=test_series.index)
print(comp)
