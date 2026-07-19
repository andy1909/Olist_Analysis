# tests/unit/utils/test_visualization.py
"""
Unit tests for src/utils/visualization.py

Covers:
    - plot_monthly_trend    : dual-axis chart → Chart_1_Monthly_Trend.png
    - plot_state_performance: bar chart       → Chart_2_State_Late_Rate.png
    - plot_forecast         : line chart      → Chart_3_Forecast.png

All tests verify that the expected PNG file is created and non-empty.
No pixel-level assertions are made — visual correctness is handled by
reviewing the charts manually or with a dedicated visual regression tool.
"""
import os
import shutil
import tempfile
import unittest

import pandas as pd

from src.utils.visualization import (
    plot_monthly_trend,
    plot_state_performance,
    plot_forecast,
)


def _write_csv(df: pd.DataFrame, directory: str, filename: str) -> str:
    """Helper: write a DataFrame to CSV and return the full path."""
    path = os.path.join(directory, filename)
    df.to_csv(path, index=False)
    return path


class TestPlotMonthlyTrend(unittest.TestCase):
    """Tests for plot_monthly_trend()."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.kpi_month_df = pd.DataFrame({
            'order_month':  ['2018-01', '2018-02', '2018-03'],
            'total_orders': [120,        180,        210      ],
            'revenue':      [6000.,      9000.,      10500.   ],
            'avg_lead_time':[10.2,       11.5,       9.8      ],
        })
        self.kpi_path = _write_csv(self.kpi_month_df, self.tmp_dir, 'KPI_by_Month.csv')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_creates_chart_png_in_output_directory(self):
        plot_monthly_trend(self.kpi_path, self.tmp_dir)
        chart_path = os.path.join(self.tmp_dir, 'Chart_1_Monthly_Trend.png')
        self.assertTrue(os.path.exists(chart_path),
                        msg="Chart_1_Monthly_Trend.png was not created.")

    def test_chart_file_is_non_empty(self):
        plot_monthly_trend(self.kpi_path, self.tmp_dir)
        chart_path = os.path.join(self.tmp_dir, 'Chart_1_Monthly_Trend.png')
        self.assertGreater(os.path.getsize(chart_path), 0)

    def test_does_not_raise_when_kpi_file_is_absent(self):
        """Should log a warning, not crash."""
        bad_path = os.path.join(self.tmp_dir, 'nonexistent.csv')
        try:
            plot_monthly_trend(bad_path, self.tmp_dir)
        except Exception as e:
            self.fail(f"plot_monthly_trend raised unexpectedly with missing file: {e}")


class TestPlotStatePerformance(unittest.TestCase):
    """Tests for plot_state_performance()."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        states = ['SP', 'RJ', 'MG', 'RS', 'PR', 'BA', 'SC', 'GO', 'PE', 'CE', 'PA']
        self.kpi_state_df = pd.DataFrame({
            'customer_state':   states,
            'total_orders':     [50000, 12000, 8000, 5000, 4500, 3200, 3000, 2500, 2000, 1500, 900],
            'late_rate_percent':[5.2,   12.97, 7.1, 4.3, 6.5, 8.9, 3.2, 9.1, 11.2, 6.7, 14.3],
            'avg_lead_time_days':[8.9,  15.1,  9.3, 7.8, 8.2, 12.1, 7.5, 10.3, 11.8, 9.9, 16.2],
        })
        self.kpi_path = _write_csv(self.kpi_state_df, self.tmp_dir, 'KPI_by_State.csv')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_creates_chart_png_in_output_directory(self):
        plot_state_performance(self.kpi_path, self.tmp_dir)
        chart_path = os.path.join(self.tmp_dir, 'Chart_2_State_Late_Rate.png')
        self.assertTrue(os.path.exists(chart_path),
                        msg="Chart_2_State_Late_Rate.png was not created.")

    def test_chart_file_is_non_empty(self):
        plot_state_performance(self.kpi_path, self.tmp_dir)
        chart_path = os.path.join(self.tmp_dir, 'Chart_2_State_Late_Rate.png')
        self.assertGreater(os.path.getsize(chart_path), 0)

    def test_does_not_raise_when_kpi_file_is_absent(self):
        bad_path = os.path.join(self.tmp_dir, 'nonexistent.csv')
        try:
            plot_state_performance(bad_path, self.tmp_dir)
        except Exception as e:
            self.fail(f"plot_state_performance raised unexpectedly with missing file: {e}")


class TestPlotForecast(unittest.TestCase):
    """Tests for plot_forecast()."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        import numpy as np
        history_dates = pd.date_range('2017-01-01', periods=70, freq='W')
        forecast_dates = pd.date_range(history_dates[-1], periods=13, freq='W')[1:]
        history_df = pd.DataFrame({
            'date':        history_dates,
            'order_count': np.linspace(100, 350, 70) + np.random.normal(0, 10, 70),
            'type':        'History',
        })
        forecast_df = pd.DataFrame({
            'date':        forecast_dates,
            'order_count': np.linspace(355, 400, 12),
            'type':        'Forecast',
        })
        self.forecast_df = pd.concat([history_df, forecast_df], ignore_index=True)
        self.forecast_path = _write_csv(self.forecast_df, self.tmp_dir, 'Forecast_Results.csv')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_creates_chart_png_in_output_directory(self):
        plot_forecast(self.forecast_path, self.tmp_dir)
        chart_path = os.path.join(self.tmp_dir, 'Chart_3_Forecast.png')
        self.assertTrue(os.path.exists(chart_path),
                        msg="Chart_3_Forecast.png was not created.")

    def test_chart_file_is_non_empty(self):
        plot_forecast(self.forecast_path, self.tmp_dir)
        chart_path = os.path.join(self.tmp_dir, 'Chart_3_Forecast.png')
        self.assertGreater(os.path.getsize(chart_path), 0)

    def test_does_not_raise_when_forecast_file_is_absent(self):
        bad_path = os.path.join(self.tmp_dir, 'nonexistent.csv')
        try:
            plot_forecast(bad_path, self.tmp_dir)
        except Exception as e:
            self.fail(f"plot_forecast raised unexpectedly with missing file: {e}")


if __name__ == '__main__':
    unittest.main()
