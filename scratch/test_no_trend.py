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

# 1. HW with trend=None
model_hw_no_trend = ExponentialSmoothing(
    train_series, seasonal_periods=12, trend=None, seasonal='add',
    initialization_method='estimated'
).fit(optimized=True)
fc_hw_no_trend = model_hw_no_trend.forecast(FORECAST_PERIODS)

# 2. HW with trend='add' (original)
model_hw_trend = ExponentialSmoothing(
    train_series, seasonal_periods=12, trend='add', seasonal='add',
    damped_trend=True, initialization_method='estimated'
).fit(optimized=True)
fc_hw_trend = model_hw_trend.forecast(FORECAST_PERIODS)

# 3. SARIMA(1,0,1)(1,1,1,12) - without differencing trend
model_sarima_no_diff = SARIMAX(train_series, order=(1,0,1), seasonal_order=(1,1,1,12),
                                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
fc_sarima_no_diff = model_sarima_no_diff.forecast(FORECAST_PERIODS)

def mape(act, pred):
    return np.mean(np.abs(act - pred) / act) * 100

print(f"HW (with trend) MAPE: {mape(test_series.values, fc_hw_trend.values):.2f}%")
print(f"HW (no trend) MAPE: {mape(test_series.values, fc_hw_no_trend.values):.2f}%")
print(f"SARIMA (no diff) MAPE: {mape(test_series.values, fc_sarima_no_diff.values):.2f}%")

print("\nActual vs HW (no trend) vs SARIMA (no diff):")
comp = pd.DataFrame({
    'Actual': test_series.values,
    'HW_NoTrend': fc_hw_no_trend.values,
    'SARIMA_NoDiff': fc_sarima_no_diff.values
}, index=test_series.index)
print(comp)
