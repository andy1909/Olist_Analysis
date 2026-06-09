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

# Drop last two weeks
weekly_orders = weekly_orders.iloc[:-2]

# Split details: Test set is 12 weeks from 2018-02-11 to 2018-04-29
# Index of 2018-04-29 in weekly_orders
end_date = '2018-04-29'
test_series = weekly_orders.loc[:end_date].iloc[-12:]
train_series = weekly_orders.loc[:test_series.index[0]].iloc[:-1]

print("train_series end date:", train_series.index[-1], "value:", train_series.iloc[-1])
print("test_series date range:", test_series.index[0], "to", test_series.index[-1])
print("test_series values:")
print(test_series)

# Holt-Winters (with trend='add' since it's in the growth phase)
model_hw = ExponentialSmoothing(
    train_series, seasonal_periods=12, trend='add', seasonal='add',
    damped_trend=True, initialization_method='estimated'
).fit(optimized=True)
fc_hw = model_hw.forecast(12)

# SARIMA
model_sarima = SARIMAX(train_series, order=(1,1,1), seasonal_order=(1,1,1,12), 
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
fc_sarima = model_sarima.forecast(12)

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
for _ in range(12):
    lags = [history[-1], history[-2], history[-3]]
    pred = rf.predict([lags])[0]
    fc_rf.append(pred)
    history.append(pred)

def mape(act, pred):
    return np.mean(np.abs(act - pred) / act) * 100

print(f"\nModel MAPEs on Clean Split:")
print(f"Naive: {mape(test_series.values, np.full(12, train_series.iloc[-1])):.2f}%")
print(f"Holt-Winters: {mape(test_series.values, fc_hw.values):.2f}%")
print(f"SARIMA: {mape(test_series.values, fc_sarima.values):.2f}%")
print(f"Random Forest: {mape(test_series.values, fc_rf):.2f}%")

print("\nActual vs HW vs SARIMA:")
comp = pd.DataFrame({
    'Actual': test_series.values,
    'HW': fc_hw.values,
    'SARIMA': fc_sarima.values,
    'RF': fc_rf
}, index=test_series.index)
print(comp)
