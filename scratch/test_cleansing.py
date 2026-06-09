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

# Drop the last week (incomplete)
weekly_orders = weekly_orders.iloc[:-1]

FORECAST_PERIODS = 12
train_series = weekly_orders.iloc[:-FORECAST_PERIODS].copy()
test_series = weekly_orders.iloc[-FORECAST_PERIODS:]

# Automatic cleansing function
def cleanse_outliers(series):
    rolling_median = series.rolling(window=6, min_periods=1, center=True).median()
    rolling_std = series.rolling(window=6, min_periods=1, center=True).std()
    
    # The strike caused a drop of ~50% in volume, which is > 2.5 std devs
    lower_bound = rolling_median - 2.0 * rolling_std
    upper_bound = rolling_median + 2.0 * rolling_std
    
    is_outlier = (series < lower_bound) | (series > upper_bound)
    
    series_cleansed = series.copy()
    series_cleansed[is_outlier] = rolling_median[is_outlier]
    return series_cleansed, is_outlier

train_cleansed, outliers = cleanse_outliers(train_series)
print("Detected outliers in train set:")
print(train_series[outliers])

# Train HW on cleansed data
model_hw_clean = ExponentialSmoothing(
    train_cleansed, seasonal_periods=12, trend='add', seasonal='add',
    damped_trend=True, initialization_method='estimated'
).fit(optimized=True)
fc_hw_clean = model_hw_clean.forecast(FORECAST_PERIODS)

# Train SARIMA on cleansed data
model_sarima_clean = SARIMAX(train_cleansed, order=(1,1,1), seasonal_order=(1,1,1,12), 
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
fc_sarima_clean = model_sarima_clean.forecast(FORECAST_PERIODS)

# Mapes
def mape(act, pred):
    return np.mean(np.abs(act - pred) / act) * 100

print(f"\nOriginal HW MAPE: {mape(test_series.values, ExponentialSmoothing(train_series, seasonal_periods=12, trend='add', seasonal='add', damped_trend=True, initialization_method='estimated').fit(optimized=True).forecast(FORECAST_PERIODS).values):.2f}%")
print(f"Cleansed HW MAPE: {mape(test_series.values, fc_hw_clean.values):.2f}%")
print(f"Original SARIMA MAPE: {mape(test_series.values, SARIMAX(train_series, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False).forecast(FORECAST_PERIODS).values):.2f}%")
print(f"Cleansed SARIMA MAPE: {mape(test_series.values, fc_sarima_clean.values):.2f}%")

print("\nActual vs Cleansed HW vs Cleansed SARIMA:")
comp = pd.DataFrame({
    'Actual': test_series.values,
    'Cleansed_HW': fc_hw_clean.values,
    'Cleansed_SARIMA': fc_sarima_clean.values
}, index=test_series.index)
print(comp)
