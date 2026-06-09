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

weekly_orders_clean = weekly_orders.copy()

# 1. Cleanse strike weeks (2018-05-27 and 2018-06-03)
strike_weeks = ['2018-05-27', '2018-06-03']
pre_strike_avg = weekly_orders.loc[(weekly_orders.index >= '2018-04-29') & (weekly_orders.index <= '2018-05-20')].mean()
for week in strike_weeks:
    if week in weekly_orders_clean.index:
        weekly_orders_clean.loc[week] = pre_strike_avg

# 2. Cleanse World Cup drop weeks (2018-07-08 and 2018-07-15)
world_cup_weeks = ['2018-07-08', '2018-07-15']
post_strike_avg = weekly_orders.loc[(weekly_orders.index >= '2018-06-10') & (weekly_orders.index <= '2018-07-01')].mean()
for week in world_cup_weeks:
    if week in weekly_orders_clean.index:
        weekly_orders_clean.loc[week] = post_strike_avg

FORECAST_PERIODS = 12
train_series = weekly_orders_clean.iloc[:-FORECAST_PERIODS]
test_series = weekly_orders_clean.iloc[-FORECAST_PERIODS:]

# Holt-Winters (with damped trend)
model_hw = ExponentialSmoothing(
    train_series, seasonal_periods=12, trend='add', seasonal='add',
    damped_trend=True, initialization_method='estimated'
).fit(optimized=True)
fc_hw = model_hw.forecast(FORECAST_PERIODS)

# SARIMA
model_sarima = SARIMAX(train_series, order=(1,1,1), seasonal_order=(1,1,1,12), 
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
fc_sarima = model_sarima.forecast(FORECAST_PERIODS)

def mape(act, pred):
    return np.mean(np.abs(act - pred) / act) * 100

print(f"HW MAPE: {mape(test_series.values, fc_hw.values):.2f}%")
print(f"SARIMA MAPE: {mape(test_series.values, fc_sarima.values):.2f}%")

print("\nActual Cleaned vs HW vs SARIMA:")
comp = pd.DataFrame({
    'Actual_Cleaned': test_series.values,
    'HW': fc_hw.values,
    'SARIMA': fc_sarima.values
}, index=test_series.index)
print(comp)
