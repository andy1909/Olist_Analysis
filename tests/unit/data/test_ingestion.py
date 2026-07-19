# tests/unit/data/test_ingestion.py
"""
Unit tests for src/data/ingestion.py

Covers:
    - convert_and_save_files : CSV → JSON / CSV → XML file conversion
    - fetch_holidays_api     : REST API call (mocked with unittest.mock)
    - load_raw_data          : loading CSV / JSON / XML into DataFrames

External dependencies (requests) are mocked so tests run offline.
Temporary directories are used for all file I/O to avoid touching the project data/.
"""
import os
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

from src.data.ingestion import convert_and_save_files, fetch_holidays_api, load_raw_data


class TestConvertAndSaveFiles(unittest.TestCase):
    """Tests for convert_and_save_files()."""

    def setUp(self):
        """Create a fresh temp directory for each test."""
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove temp directory and all its contents after each test."""
        shutil.rmtree(self.tmp_dir)

    # ── Customers CSV → JSON ──────────────────────────────────────────────────

    def test_creates_json_file_when_customers_csv_exists(self):
        df = pd.DataFrame({
            'customer_id': ['c1', 'c2'],
            'customer_state': ['SP', 'RJ'],
        })
        df.to_csv(os.path.join(self.tmp_dir, 'olist_customers_dataset.csv'), index=False)

        convert_and_save_files(self.tmp_dir)

        self.assertTrue(
            os.path.exists(os.path.join(self.tmp_dir, 'source_customers.json'))
        )

    def test_json_output_contains_same_row_count_as_source_csv(self):
        df = pd.DataFrame({'customer_id': ['c1', 'c2', 'c3'], 'customer_state': ['SP', 'RJ', 'MG']})
        df.to_csv(os.path.join(self.tmp_dir, 'olist_customers_dataset.csv'), index=False)

        convert_and_save_files(self.tmp_dir)

        json_path = os.path.join(self.tmp_dir, 'source_customers.json')
        with open(json_path) as f:
            records = json.load(f)
        self.assertEqual(len(records), 3)

    def test_does_not_raise_when_customers_csv_is_absent(self):
        try:
            convert_and_save_files(self.tmp_dir)
        except Exception as e:
            self.fail(f"convert_and_save_files raised unexpectedly: {e}")

    # ── Products CSV → XML ────────────────────────────────────────────────────

    def test_creates_xml_file_when_products_csv_exists(self):
        df = pd.DataFrame({
            'product_id': ['p1', 'p2'],
            'product_category_name': ['electronics', 'toys'],
        })
        df.to_csv(os.path.join(self.tmp_dir, 'olist_products_dataset.csv'), index=False)

        convert_and_save_files(self.tmp_dir)

        self.assertTrue(
            os.path.exists(os.path.join(self.tmp_dir, 'source_products.xml'))
        )

    def test_xml_column_names_have_underscores_stripped(self):
        """ingestion.py strips '_' from column names before exporting to XML."""
        df = pd.DataFrame({
            'product_id': ['p1'],
            'product_category_name': ['books'],
        })
        df.to_csv(os.path.join(self.tmp_dir, 'olist_products_dataset.csv'), index=False)

        convert_and_save_files(self.tmp_dir)

        xml_path = os.path.join(self.tmp_dir, 'source_products.xml')
        df_loaded = pd.read_xml(xml_path)
        # After stripping '_': product_id → productid
        self.assertIn('productid', df_loaded.columns)
        self.assertNotIn('product_id', df_loaded.columns)


class TestFetchHolidaysApi(unittest.TestCase):
    """Tests for fetch_holidays_api() — external HTTP calls are mocked."""

    @patch('src.data.ingestion.requests.get')
    def test_returns_dataframe_with_expected_columns_when_api_succeeds(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                'date': '2018-01-01',
                'localName': 'Confraternização Universal',
                'name': "New Year's Day",
            }
        ]
        mock_get.return_value = mock_response

        df = fetch_holidays_api(years=[2018])

        self.assertIsInstance(df, pd.DataFrame)
        for col in ('date', 'localName', 'name'):
            self.assertIn(col, df.columns, msg=f"Expected column '{col}' in result.")

    @patch('src.data.ingestion.requests.get')
    def test_aggregates_holidays_across_multiple_years(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {'date': '2017-01-01', 'localName': 'Ano Novo', 'name': "New Year's Day"}
        ]
        mock_get.return_value = mock_response

        df = fetch_holidays_api(years=[2017, 2018])

        # One holiday per year × 2 years = 2 rows
        self.assertEqual(len(df), 2)

    @patch('src.data.ingestion.requests.get')
    def test_returns_empty_dataframe_when_api_raises_connection_error(self, mock_get):
        mock_get.side_effect = Exception("Network unreachable")

        df = fetch_holidays_api(years=[2018])

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 0)

    @patch('src.data.ingestion.requests.get')
    def test_date_column_is_parsed_as_datetime(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {'date': '2018-04-21', 'localName': 'Tiradentes', 'name': 'Tiradentes'}
        ]
        mock_get.return_value = mock_response

        df = fetch_holidays_api(years=[2018])

        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['date']))


class TestLoadRawData(unittest.TestCase):
    """Tests for load_raw_data()."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

        # Seed minimal CSV / JSON / XML files that the function expects
        orders_df = pd.DataFrame({
            'order_id': ['o1', 'o2'],
            'customer_id': ['c1', 'c2'],
            'order_status': ['delivered', 'shipped'],
        })
        orders_df.to_csv(
            os.path.join(self.tmp_dir, 'olist_orders_dataset.csv'), index=False
        )

        customers_df = pd.DataFrame({
            'customer_id': ['c1', 'c2'],
            'customer_state': ['SP', 'RJ'],
        })
        customers_df.to_json(
            os.path.join(self.tmp_dir, 'source_customers.json'),
            orient='records', indent=2
        )

        products_df = pd.DataFrame({
            'productid': ['p1'],
            'productcategoryname': ['electronics'],
        })
        products_df.to_xml(
            os.path.join(self.tmp_dir, 'source_products.xml'),
            index=False, root_name='Products', row_name='Item'
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_returns_three_dataframes(self):
        orders, customers, products = load_raw_data(self.tmp_dir)
        for df in (orders, customers, products):
            self.assertIsInstance(df, pd.DataFrame)

    def test_orders_dataframe_has_correct_row_count(self):
        orders, _, _ = load_raw_data(self.tmp_dir)
        self.assertEqual(len(orders), 2)

    def test_customers_dataframe_has_correct_row_count(self):
        _, customers, _ = load_raw_data(self.tmp_dir)
        self.assertEqual(len(customers), 2)

    def test_products_dataframe_has_correct_row_count(self):
        _, _, products = load_raw_data(self.tmp_dir)
        self.assertEqual(len(products), 1)


if __name__ == '__main__':
    unittest.main()
