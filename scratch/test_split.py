import os
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
master_path = os.path.join(PROCESSED_DATA_DIR, 'Master_Logistics_Data.csv')

df = pd.read_csv(master_path, parse_dates=['order_purchase_timestamp'])
df_ts = df.set_index('order_purchase_timestamp')
weekly_orders = df_ts.resample('W')['order_id'].nunique().fillna(0)
weekly_orders = weekly_orders[weekly_orders.index >= '2017-01-01']
weekly_orders = weekly_orders.asfreq('W').fillna(0).astype(float)

# Drop the last TWO weeks because they are incomplete (2018-09-02 and 2018-08-26)
weekly_orders = weekly_orders.iloc[:-2]

FORECAST_PERIODS = 12
train_series = weekly_orders.iloc[:-FORECAST_PERIODS]
test_series = weekly_orders.iloc[-FORECAST_PERIODS:]

print("weekly_orders length:", len(weekly_orders))
print("train_series end date:", train_series.index[-1], "value:", train_series.iloc[-1])
print("test_series range:", test_series.index[0], "to", test_series.index[-1])

# Holt-Winters
model_hw = ExponentialSmoothing(
    train_series, seasonal_periods=12, trend='add', seasonal='add',
    damped_trend=True, initialization_method='estimated'
).fit(optimized=True)
fc_hw = model_hw.forecast(FORECAST_PERIODS)

# SARIMA
model_sarima = SARIMAX(train_series, order=(1,1,1), seasonal_order=(1,1,1,12), 
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
fc_sarima = model_sarima.forecast(FORECAST_PERIODS)

# Naive
last_val = train_series.iloc[-1]
fc_naive = np.full(FORECAST_PERIODS, last_val)

def calculate_metrics(act, pred):
    mae = np.mean(np.abs(act - pred))
    rmse = np.sqrt(np.mean((act - pred)**2))
    mape = np.mean(np.abs((act - pred) / act)) * 100
    return mae, rmse, mape

print("\nMetrics on new split:")
print("Holt-Winters:", calculate_metrics(test_series.values, fc_hw.values))
print("SARIMA:", calculate_metrics(test_series.values, fc_sarima.values))
print("Naive:", calculate_metrics(test_series.values, fc_naive))

print("\nActual vs HW Forecast vs SARIMA Forecast:")
comp = pd.DataFrame({
    'Actual': test_series.values,
    'HW': fc_hw.values,
    'SARIMA': fc_sarima.values
}, index=test_series.index)
print(comp)
