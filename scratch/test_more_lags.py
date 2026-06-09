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

# Slice to end on 2018-05-20
weekly_orders = weekly_orders[weekly_orders.index <= '2018-05-20']

FORECAST_PERIODS = 12
train_series = weekly_orders.iloc[:-FORECAST_PERIODS]
test_series = weekly_orders.iloc[-FORECAST_PERIODS:]

# Create lag features including seasonal lags
LAGS_LIST = [1, 2, 3, 4, 12, 13]

def create_rich_lags(series, lags=LAGS_LIST):
    df = pd.DataFrame(series)
    df.columns = ['y']
    for lag in lags:
        df[f'lag_{lag}'] = df['y'].shift(lag)
    df.dropna(inplace=True)
    X = df.drop(columns=['y'])
    y = df['y']
    return X, y

X_train, y_train = create_rich_lags(train_series)

# RF
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# XGBoost
xg = xgb.XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
xg.fit(X_train, y_train)

# LightGBM (with small min_data_in_leaf to allow splits on small dataset)
lgbm = lgb.LGBMRegressor(n_estimators=100, random_state=42, min_child_samples=5, verbose=-1)
lgbm.fit(X_train, y_train)

# Recursive forecast helper
def recursive_forecast(model, train_series, periods, lags=LAGS_LIST):
    history = list(train_series.values)
    forecast = []
    for i in range(periods):
        # Extract features for prediction: history[-lag] for each lag in lags
        features = [history[-lag] for lag in lags]
        pred = model.predict(np.array([features]))[0]
        forecast.append(pred)
        history.append(pred)
    return forecast

fc_rf = recursive_forecast(rf, train_series, FORECAST_PERIODS)
fc_xgb = recursive_forecast(xg, train_series, FORECAST_PERIODS)
fc_lgb = recursive_forecast(lgbm, train_series, FORECAST_PERIODS)

def mape(act, pred):
    return np.mean(np.abs(act - pred) / act) * 100

print("All Model MAPEs with Rich Lags:")
print(f"Random Forest (Rich Lags): {mape(test_series.values, fc_rf):.2f}%")
print(f"XGBoost (Rich Lags): {mape(test_series.values, fc_xgb):.2f}%")
print(f"LightGBM (Rich Lags): {mape(test_series.values, fc_lgb):.2f}%")

print("\nActual vs RF Rich Lags vs LightGBM Rich Lags Forecasts:")
comp = pd.DataFrame({
    'Actual': test_series.values,
    'RF_Rich': fc_rf,
    'LGB_Rich': fc_lgb,
    'XGB_Rich': fc_xgb
}, index=test_series.index)
print(comp)
