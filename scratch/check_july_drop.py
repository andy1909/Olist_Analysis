import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
master_path = os.path.join(PROCESSED_DATA_DIR, 'Master_Logistics_Data.csv')

df = pd.read_csv(master_path, parse_dates=['order_purchase_timestamp'])
df['order_date'] = df['order_purchase_timestamp'].dt.date
daily_orders = df.groupby('order_date')['order_id'].nunique()

print("Daily orders in July 2018:")
print(daily_orders[(daily_orders.index >= pd.to_datetime('2018-07-01').date()) & 
                   (daily_orders.index <= pd.to_datetime('2018-07-25').date())])
