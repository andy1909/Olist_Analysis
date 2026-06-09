import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
master_path = os.path.join(PROCESSED_DATA_DIR, 'Master_Logistics_Data.csv')

df = pd.read_csv(master_path, parse_dates=['order_purchase_timestamp'])
df_ts = df.set_index('order_purchase_timestamp')
weekly_orders = df_ts.resample('W')['order_id'].nunique().fillna(0)
weekly_orders = weekly_orders[weekly_orders.index >= '2017-01-01']
weekly_orders = weekly_orders.asfreq('W').fillna(0).astype(float)
if len(weekly_orders) > 0:
    weekly_orders = weekly_orders.iloc[:-1]

FORECAST_PERIODS = 12
train_series = weekly_orders.iloc[:-FORECAST_PERIODS]
test_series = weekly_orders.iloc[-FORECAST_PERIODS:]

print("weekly_orders length:", len(weekly_orders))
print("train_series date range:", train_series.index[0], "to", train_series.index[-1])
print("test_series date range:", test_series.index[0], "to", test_series.index[-1])

# Check how Holt-Winters is fitted and forecast index
from statsmodels.tsa.holtwinters import ExponentialSmoothing
model = ExponentialSmoothing(
    train_series, seasonal_periods=12, trend='add', seasonal='add',
    damped_trend=True, initialization_method='estimated'
).fit(optimized=True)
hw_forecast = model.forecast(FORECAST_PERIODS)
print("\nhw_forecast index and values:")
print(hw_forecast)

print("\ntest_series index and values:")
print(test_series)
