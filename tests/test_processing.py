# tests/test_processing.py
import unittest
import pandas as pd
import numpy as np
from src.data.processing import clean_and_convert_types, handle_missing_data, create_logistics_features

class TestProcessing(unittest.TestCase):
    def test_clean_and_convert_types(self):
        df_orders = pd.DataFrame({
            'order_purchase_timestamp': ['2018-01-01 10:00:00'],
            'order_approved_at': ['2018-01-01 11:00:00'],
            'order_delivered_carrier_date': ['2018-01-02 10:00:00'],
            'order_delivered_customer_date': ['2018-01-05 10:00:00'],
            'order_estimated_delivery_date': ['2018-01-10 10:00:00']
        })
        df_holidays = pd.DataFrame({
            'date': ['2018-01-01']
        })
        
        df_orders_clean, df_holidays_clean = clean_and_convert_types(df_orders, df_holidays)
        
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_orders_clean['order_purchase_timestamp']))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_holidays_clean['date']))

    def test_handle_missing_data(self):
        df = pd.DataFrame({
            'order_status': ['delivered', 'shipped', 'delivered'],
            'order_purchase_timestamp': [pd.Timestamp('2018-01-01'), pd.Timestamp('2018-01-01'), pd.Timestamp('2018-01-01')],
            'order_delivered_customer_date': [pd.Timestamp('2018-01-05'), pd.NaT, pd.NaT],
            'product_category_name': [np.nan, 'furniture', np.nan]
        })
        
        df_clean = handle_missing_data(df)
        
        # Only the first row is 'delivered' and has delivered customer date
        self.assertEqual(len(df_clean), 1)
        self.assertEqual(df_clean.iloc[0]['product_category_name'], 'Unknown')

    def test_create_logistics_features(self):
        df = pd.DataFrame({
            'order_purchase_timestamp': [pd.Timestamp('2018-01-01 10:00:00')],
            'order_delivered_customer_date': [pd.Timestamp('2018-01-05 10:00:00')],
            'order_estimated_delivery_date': [pd.Timestamp('2018-01-04 10:00:00')]
        })
        df_holidays = pd.DataFrame({
            'date': pd.to_datetime(['2018-01-02'])
        })
        
        df_features = create_logistics_features(df, df_holidays)
        
        self.assertEqual(df_features.iloc[0]['lead_time_days'], 4)
        self.assertEqual(df_features.iloc[0]['estimated_lead_time_days'], 3)
        self.assertEqual(df_features.iloc[0]['days_diff_estimated'], 1)
        self.assertEqual(df_features.iloc[0]['is_late'], 1)
        self.assertEqual(df_features.iloc[0]['holidays_in_transit'], 1)

if __name__ == '__main__':
    unittest.main()
