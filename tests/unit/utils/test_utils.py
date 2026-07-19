# tests/unit/utils/test_utils.py
"""
Unit tests for src/utils/utils.py

Covers:
    - ensure_dir       : directory creation (idempotent)
    - save_dataframe   : DataFrame → CSV / Excel persistence
    - load_config      : JSON configuration loading
"""
import json
import os
import shutil
import tempfile
import unittest

import pandas as pd

from src.utils.utils import ensure_dir, save_dataframe, load_config


class TestEnsureDir(unittest.TestCase):
    """Tests for ensure_dir()."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_creates_directory_when_path_does_not_exist(self):
        new_path = os.path.join(self.tmp_dir, 'brand_new_subdir')
        self.assertFalse(os.path.exists(new_path))

        ensure_dir(new_path)

        self.assertTrue(os.path.exists(new_path))
        self.assertTrue(os.path.isdir(new_path))

    def test_creates_nested_directories_in_a_single_call(self):
        deep_path = os.path.join(self.tmp_dir, 'level1', 'level2', 'level3')

        ensure_dir(deep_path)

        self.assertTrue(os.path.exists(deep_path))

    def test_does_not_raise_when_directory_already_exists(self):
        try:
            ensure_dir(self.tmp_dir)   # tmp_dir already exists
            ensure_dir(self.tmp_dir)   # second call must also be safe
        except Exception as e:
            self.fail(f"ensure_dir raised unexpectedly on existing directory: {e}")

    def test_returns_the_same_path_that_was_passed_in(self):
        new_path = os.path.join(self.tmp_dir, 'returned_path_test')
        returned = ensure_dir(new_path)
        self.assertEqual(returned, new_path)


class TestSaveDataframe(unittest.TestCase):
    """Tests for save_dataframe()."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.df = pd.DataFrame({
            'order_id': ['o1', 'o2', 'o3'],
            'revenue':  [100., 200., 300.],
        })

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    # ── CSV ───────────────────────────────────────────────────────────────────

    def test_saves_csv_file_to_specified_path(self):
        csv_path = os.path.join(self.tmp_dir, 'output.csv')
        save_dataframe(self.df, csv_path, format='csv')
        self.assertTrue(os.path.exists(csv_path))

    def test_saved_csv_can_be_reloaded_with_matching_shape(self):
        csv_path = os.path.join(self.tmp_dir, 'output.csv')
        save_dataframe(self.df, csv_path, format='csv')

        df_loaded = pd.read_csv(csv_path)
        self.assertEqual(df_loaded.shape, self.df.shape)

    def test_saved_csv_values_match_original_dataframe(self):
        csv_path = os.path.join(self.tmp_dir, 'output.csv')
        save_dataframe(self.df, csv_path)

        df_loaded = pd.read_csv(csv_path)
        pd.testing.assert_frame_equal(df_loaded, self.df)

    def test_creates_parent_directory_if_absent_when_saving_csv(self):
        nested_path = os.path.join(self.tmp_dir, 'nested', 'output.csv')
        save_dataframe(self.df, nested_path, format='csv')
        self.assertTrue(os.path.exists(nested_path))

    # ── Excel ─────────────────────────────────────────────────────────────────

    def test_saves_excel_file_when_format_is_excel(self):
        excel_path = os.path.join(self.tmp_dir, 'output.xlsx')
        save_dataframe(self.df, excel_path, format='excel')
        self.assertTrue(os.path.exists(excel_path))

    def test_saved_excel_can_be_reloaded_with_matching_shape(self):
        excel_path = os.path.join(self.tmp_dir, 'output.xlsx')
        save_dataframe(self.df, excel_path, format='excel')

        df_loaded = pd.read_excel(excel_path)
        self.assertEqual(df_loaded.shape, self.df.shape)


class TestLoadConfig(unittest.TestCase):
    """Tests for load_config()."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _write_config(self, data: dict) -> str:
        path = os.path.join(self.tmp_dir, 'config.json')
        with open(path, 'w') as f:
            json.dump(data, f)
        return path

    def test_returns_dict_matching_json_content(self):
        config_data = {'model': 'xgboost', 'n_estimators': 50, 'learning_rate': 0.05}
        path = self._write_config(config_data)

        result = load_config(path)

        self.assertEqual(result, config_data)

    def test_returns_nested_dict_correctly(self):
        config_data = {
            'forecasting': {'periods': 12, 'seasonal_periods': 13},
            'logistics': {'late_threshold_days': 1},
        }
        path = self._write_config(config_data)

        result = load_config(path)

        self.assertEqual(result['forecasting']['periods'], 12)
        self.assertEqual(result['logistics']['late_threshold_days'], 1)

    def test_raises_file_not_found_for_missing_config(self):
        bad_path = os.path.join(self.tmp_dir, 'nonexistent.json')
        with self.assertRaises(FileNotFoundError):
            load_config(bad_path)


if __name__ == '__main__':
    unittest.main()
