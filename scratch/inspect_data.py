import os
import pandas as pd

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

print("Weekly Orders Descriptive Stats:")
print(weekly_orders.describe())

print("\nLast 20 weeks of historical data before test set:")
train_end_idx = len(weekly_orders) - 12
print(weekly_orders.iloc[train_end_idx - 20 : train_end_idx])

print("\nTest set (last 12 weeks):")
print(weekly_orders.iloc[train_end_idx:])
