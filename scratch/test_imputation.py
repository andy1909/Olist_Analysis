import os
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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
train_series = weekly_orders.iloc[:-FORECAST_PERIODS].copy()
test_series = weekly_orders.iloc[-FORECAST_PERIODS:]

print("Original train tail:")
print(train_series.tail(5))

# Let's try imputing the last two weeks of train (strike period) with rolling average of previous 4 weeks
normal_val = train_series.iloc[:-2].iloc[-4:].mean()
print(f"\nCalculated normal volume before strike: {normal_val:.2f}")

train_series_imputed = train_series.copy()
train_series_imputed.iloc[-2:] = normal_val
print("\nImputed train tail:")
print(train_series_imputed.tail(5))

# Original HW Forecast
model_orig = ExponentialSmoothing(
    train_series, seasonal_periods=12, trend='add', seasonal='add',
    damped_trend=True, initialization_method='estimated'
).fit(optimized=True)
fc_orig = model_orig.forecast(FORECAST_PERIODS)

# Imputed HW Forecast
model_imp = ExponentialSmoothing(
    train_series_imputed, seasonal_periods=12, trend='add', seasonal='add',
    damped_trend=True, initialization_method='estimated'
).fit(optimized=True)
fc_imp = model_imp.forecast(FORECAST_PERIODS)

# Compute MAPEs
def mape(act, pred):
    return np.mean(np.abs(act - pred) / act) * 100

print(f"\nOriginal HW MAPE: {mape(test_series.values, fc_orig.values):.2f}%")
print(f"Imputed HW MAPE: {mape(test_series.values, fc_imp.values):.2f}%")

print("\nOriginal vs Imputed HW Forecasts compared to Actuals:")
comparison = pd.DataFrame({
    'Actual': test_series.values,
    'Original_HW': fc_orig.values,
    'Imputed_HW': fc_imp.values
}, index=test_series.index)
print(comparison)
