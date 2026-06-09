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

# Drop last two weeks
weekly_orders = weekly_orders.iloc[:-2]

print("Weekly Orders in 2017 Summer (June - August):")
print(weekly_orders.loc['2017-06-01':'2017-08-31'])

print("\nWeekly Orders in 2018 Summer (June - August) - Raw:")
print(weekly_orders.loc['2018-06-01':'2018-08-31'])
