# tests/unit/data/test_processing.py
"""
Unit tests for src/data/processing.py

Covers:
    - clean_and_convert_types : datetime conversion
    - handle_missing_data     : row filtering and imputation
    - create_logistics_features : KPI feature calculation
    - create_aggregated_features: weekly business metrics
    - drop_unnecessary_columns  : column pruning
"""
import unittest
import pandas as pd
import numpy as np

from src.data.processing import (
    clean_and_convert_types,
    handle_missing_data,
    create_logistics_features,
    create_aggregated_features,
    drop_unnecessary_columns,
)


class TestCleanAndConvertTypes(unittest.TestCase):
    """Tests for clean_and_convert_types()."""

    def _make_orders_df(self):
        return pd.DataFrame({
            'order_purchase_timestamp':      ['2018-01-01 10:00:00'],
            'order_approved_at':             ['2018-01-01 11:00:00'],
            'order_delivered_carrier_date':  ['2018-01-02 10:00:00'],
            'order_delivered_customer_date': ['2018-01-05 10:00:00'],
            'order_estimated_delivery_date': ['2018-01-10 10:00:00'],
        })

    def test_converts_all_order_date_columns_to_datetime(self):
        df_orders, _ = clean_and_convert_types(
            self._make_orders_df(),
            pd.DataFrame({'date': ['2018-01-01']})
        )
        date_cols = [
            'order_purchase_timestamp',
            'order_approved_at',
            'order_delivered_carrier_date',
            'order_delivered_customer_date',
            'order_estimated_delivery_date',
        ]
        for col in date_cols:
            self.assertTrue(
                pd.api.types.is_datetime64_any_dtype(df_orders[col]),
                msg=f"Column '{col}' was not converted to datetime64."
            )

    def test_converts_holidays_date_column_to_datetime(self):
        _, df_holidays = clean_and_convert_types(
            self._make_orders_df(),
            pd.DataFrame({'date': ['2018-01-01', '2018-04-21']})
        )
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_holidays['date']))

    def test_coerces_malformed_dates_to_nat_without_raising(self):
        df_orders = self._make_orders_df()
        df_orders.loc[0, 'order_approved_at'] = 'not-a-date'
        result, _ = clean_and_convert_types(
            df_orders,
            pd.DataFrame({'date': []})
        )
        self.assertTrue(pd.isna(result.loc[0, 'order_approved_at']))


class TestHandleMissingData(unittest.TestCase):
    """Tests for handle_missing_data()."""

    def _make_input_df(self):
        return pd.DataFrame({
            'order_status': ['delivered', 'shipped', 'delivered', 'delivered'],
            'order_purchase_timestamp': [
                pd.Timestamp('2018-01-01'),
                pd.Timestamp('2018-01-01'),
                pd.Timestamp('2018-01-01'),
                pd.Timestamp('2018-01-01'),
            ],
            'order_delivered_customer_date': [
                pd.Timestamp('2018-01-05'),
                pd.Timestamp('2018-01-06'),  # shipped — gets dropped anyway
                pd.NaT,                      # delivered but missing date → dropped
                pd.Timestamp('2018-01-07'),
            ],
            'product_category_name': [None, 'furniture', 'electronics', None],
        })

    def test_retains_only_delivered_orders_with_valid_delivery_date(self):
        result = handle_missing_data(self._make_input_df())
        self.assertEqual(len(result), 2)

    def test_imputes_missing_product_category_with_unknown(self):
        result = handle_missing_data(self._make_input_df())
        self.assertTrue((result['product_category_name'] != '').all())
        self.assertIn('Unknown', result['product_category_name'].values)

    def test_returns_copy_and_does_not_mutate_input(self):
        df_input = self._make_input_df()
        original_len = len(df_input)
        _ = handle_missing_data(df_input)
        self.assertEqual(len(df_input), original_len)


class TestCreateLogisticsFeatures(unittest.TestCase):
    """Tests for create_logistics_features()."""

    def _make_single_order(
        self,
        purchase='2018-01-01',
        delivered='2018-01-05',
        estimated='2018-01-04',
    ):
        return pd.DataFrame({
            'order_purchase_timestamp':     [pd.Timestamp(purchase)],
            'order_delivered_customer_date':[pd.Timestamp(delivered)],
            'order_estimated_delivery_date':[pd.Timestamp(estimated)],
        })

    def _make_holidays(self, dates):
        return pd.DataFrame({'date': pd.to_datetime(dates)})

    def test_calculates_lead_time_days_correctly(self):
        df = create_logistics_features(
            self._make_single_order(purchase='2018-01-01', delivered='2018-01-05'),
            self._make_holidays([])
        )
        self.assertEqual(df.iloc[0]['lead_time_days'], 4)

    def test_calculates_estimated_lead_time_days_correctly(self):
        df = create_logistics_features(
            self._make_single_order(purchase='2018-01-01', estimated='2018-01-04'),
            self._make_holidays([])
        )
        self.assertEqual(df.iloc[0]['estimated_lead_time_days'], 3)

    def test_flags_order_as_late_when_delivered_after_estimated_date(self):
        # delivered 05/01, estimated 04/01 → trễ 1 ngày
        df = create_logistics_features(
            self._make_single_order(delivered='2018-01-05', estimated='2018-01-04'),
            self._make_holidays([])
        )
        self.assertEqual(df.iloc[0]['days_diff_estimated'], 1)
        self.assertEqual(df.iloc[0]['is_late'], 1)

    def test_flags_order_as_on_time_when_delivered_before_estimated_date(self):
        # delivered 03/01, estimated 04/01 → sớm 1 ngày
        df = create_logistics_features(
            self._make_single_order(delivered='2018-01-03', estimated='2018-01-04'),
            self._make_holidays([])
        )
        self.assertEqual(df.iloc[0]['is_late'], 0)

    def test_counts_public_holidays_that_fall_within_transit_window(self):
        # Transit 01/01 → 05/01, holiday on 02/01 → should count 1
        df = create_logistics_features(
            self._make_single_order(purchase='2018-01-01', delivered='2018-01-05'),
            self._make_holidays(['2018-01-02'])
        )
        self.assertEqual(df.iloc[0]['holidays_in_transit'], 1)

    def test_does_not_count_holidays_outside_transit_window(self):
        # Transit 01/01 → 05/01, holiday on 10/01 → should NOT count
        df = create_logistics_features(
            self._make_single_order(purchase='2018-01-01', delivered='2018-01-05'),
            self._make_holidays(['2018-01-10'])
        )
        self.assertEqual(df.iloc[0]['holidays_in_transit'], 0)

    def test_handles_empty_holidays_dataframe_without_raising(self):
        try:
            create_logistics_features(
                self._make_single_order(),
                self._make_holidays([])
            )
        except Exception as e:
            self.fail(f"create_logistics_features raised unexpectedly: {e}")


class TestCreateAggregatedFeatures(unittest.TestCase):
    """Tests for create_aggregated_features()."""

    def _make_master_df(self):
        return pd.DataFrame({
            'order_purchase_timestamp': pd.to_datetime([
                '2018-01-01', '2018-01-03', '2018-01-08',
            ]),
            'order_id':     ['o1', 'o2', 'o3'],
            'seller_id':    ['s1', 's1', 's2'],
            'customer_id':  ['c1', 'c2', 'c3'],
            'product_id':   ['p1', 'p2', 'p1'],
            'order_item_id':[1,    1,    2   ],
            'price':        [100., 200., 150.],
            'freight_value':[10.,  20.,  15. ],
        })

    def test_adds_weekly_aggregate_columns_to_output(self):
        result = create_aggregated_features(self._make_master_df())
        expected_cols = [
            'weekly_active_sellers', 'weekly_active_customers',
            'weekly_product_variety', 'weekly_gmv',
            'weekly_avg_basket_size', 'weekly_avg_price',
            'weekly_avg_freight_value',
        ]
        for col in expected_cols:
            self.assertIn(col, result.columns, msg=f"Missing column: {col}")

    def test_row_count_is_preserved_after_aggregation(self):
        df = self._make_master_df()
        result = create_aggregated_features(df)
        self.assertEqual(len(result), len(df))

    def test_returns_original_df_when_timestamp_column_is_missing(self):
        df = pd.DataFrame({'order_id': ['o1'], 'price': [100.]})
        result = create_aggregated_features(df)
        # Should return without adding weekly columns and without crashing
        self.assertEqual(list(result.columns), list(df.columns))


class TestDropUnnecessaryColumns(unittest.TestCase):
    """Tests for drop_unnecessary_columns()."""

    def _make_df_with_all_columns(self):
        return pd.DataFrame({
            'order_id':                  ['o1'],
            'order_status':              ['delivered'],
            'product_name_lenght':       [10],
            'product_description_lenght':[200],
            'product_photos_qty':        [3],
            'order_approved_at':         [pd.Timestamp('2018-01-01')],
            'customer_unique_id':        ['uid1'],
            'product_category_name':     ['electronics'],
            'price':                     [100.0],
        })

    def test_removes_all_predefined_unnecessary_columns(self):
        cols_to_drop = [
            'order_status', 'product_name_lenght', 'product_description_lenght',
            'product_photos_qty', 'order_approved_at', 'customer_unique_id',
            'product_category_name',
        ]
        result = drop_unnecessary_columns(self._make_df_with_all_columns())
        for col in cols_to_drop:
            self.assertNotIn(col, result.columns, msg=f"Column '{col}' was not dropped.")

    def test_retains_business_critical_columns(self):
        result = drop_unnecessary_columns(self._make_df_with_all_columns())
        self.assertIn('order_id', result.columns)
        self.assertIn('price', result.columns)

    def test_does_not_raise_when_column_to_drop_is_already_absent(self):
        df = pd.DataFrame({'order_id': ['o1'], 'price': [100.0]})
        try:
            drop_unnecessary_columns(df)
        except Exception as e:
            self.fail(f"drop_unnecessary_columns raised unexpectedly: {e}")


if __name__ == '__main__':
    unittest.main()
