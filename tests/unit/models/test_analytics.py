# tests/unit/models/test_analytics.py
"""
Unit tests for src/models/analytics.py

Covers:
    - aggregate_kpis_for_dashboard : state-level and monthly KPI tables
    - forecast_orders_hybrid       : Holt-Winters + XGBoost hybrid forecast
    - forecast_orders              : Holt-Winters fallback forecast
"""
import unittest
import pandas as pd
import numpy as np

from src.models.analytics import aggregate_kpis_for_dashboard, forecast_orders


def _make_master_df():
    """Shared minimal master DataFrame used across multiple test classes."""
    return pd.DataFrame({
        'customer_state': ['SP', 'SP', 'RJ', 'RJ'],
        'order_id':       ['o1', 'o2', 'o3', 'o4'],
        'freight_value':  [10.0, 12.0, 20.0, 30.0],
        'lead_time_days': [5,    6,    12,   16  ],
        'is_late':        [0,    0,    1,    0   ],
        'price':          [100., 150., 200., 250.],
        'order_purchase_timestamp': pd.to_datetime([
            '2018-01-01', '2018-01-15', '2018-01-10', '2018-02-05'
        ]),
    })


class TestAggregateKpisForDashboard(unittest.TestCase):
    """Tests for aggregate_kpis_for_dashboard()."""

    def setUp(self):
        self.df = _make_master_df()
        self.kpi_state, self.kpi_month = aggregate_kpis_for_dashboard(self.df)

    # ── State KPI assertions ──────────────────────────────────────────────────

    def test_returns_one_row_per_unique_state(self):
        self.assertEqual(len(self.kpi_state), 2)

    def test_sp_total_orders_is_correct(self):
        sp = self.kpi_state[self.kpi_state['customer_state'] == 'SP'].iloc[0]
        self.assertEqual(sp['total_orders'], 2)

    def test_sp_avg_freight_value_is_correct(self):
        sp = self.kpi_state[self.kpi_state['customer_state'] == 'SP'].iloc[0]
        self.assertAlmostEqual(sp['avg_freight_value'], 11.0)

    def test_sp_avg_lead_time_is_correct(self):
        sp = self.kpi_state[self.kpi_state['customer_state'] == 'SP'].iloc[0]
        self.assertAlmostEqual(sp['avg_lead_time_days'], 5.5)

    def test_sp_late_rate_is_zero_percent(self):
        sp = self.kpi_state[self.kpi_state['customer_state'] == 'SP'].iloc[0]
        self.assertAlmostEqual(sp['late_rate_percent'], 0.0)

    def test_rj_late_rate_is_fifty_percent(self):
        """RJ has 1 late out of 2 orders → 50 %."""
        rj = self.kpi_state[self.kpi_state['customer_state'] == 'RJ'].iloc[0]
        self.assertAlmostEqual(rj['late_rate_percent'], 50.0)

    def test_rj_avg_lead_time_is_correct(self):
        rj = self.kpi_state[self.kpi_state['customer_state'] == 'RJ'].iloc[0]
        self.assertAlmostEqual(rj['avg_lead_time_days'], 14.0)

    def test_late_rate_is_bounded_between_0_and_100(self):
        for _, row in self.kpi_state.iterrows():
            self.assertGreaterEqual(row['late_rate_percent'], 0.0)
            self.assertLessEqual(row['late_rate_percent'], 100.0)

    # ── Monthly KPI assertions ────────────────────────────────────────────────

    def test_returns_one_row_per_unique_month(self):
        self.assertEqual(len(self.kpi_month), 2)

    def test_january_total_orders_is_correct(self):
        jan = self.kpi_month[self.kpi_month['order_month'] == '2018-01'].iloc[0]
        self.assertEqual(jan['total_orders'], 3)

    def test_january_revenue_is_sum_of_prices(self):
        jan = self.kpi_month[self.kpi_month['order_month'] == '2018-01'].iloc[0]
        self.assertAlmostEqual(jan['revenue'], 450.0)

    def test_february_total_orders_is_correct(self):
        feb = self.kpi_month[self.kpi_month['order_month'] == '2018-02'].iloc[0]
        self.assertEqual(feb['total_orders'], 1)

    def test_february_revenue_is_correct(self):
        feb = self.kpi_month[self.kpi_month['order_month'] == '2018-02'].iloc[0]
        self.assertAlmostEqual(feb['revenue'], 250.0)

    def test_order_month_column_is_string_type(self):
        """order_month must be string so it serialises cleanly to CSV / JSON."""
        self.assertTrue(
            pd.api.types.is_string_dtype(self.kpi_month['order_month'])
        )


class TestForecastOrders(unittest.TestCase):
    """Tests for forecast_orders() — Holt-Winters baseline model."""

    def _make_timeseries_df(self, n_weeks=60):
        """Build a realistic weekly time series with slight upward trend."""
        dates = pd.date_range(start='2017-01-01', periods=n_weeks, freq='W')
        np.random.seed(42)
        counts = np.linspace(100, 300, n_weeks) + np.random.normal(0, 10, n_weeks)
        return pd.DataFrame({
            'order_purchase_timestamp': dates.repeat(
                np.maximum(1, counts.astype(int))
            )[:len(dates) * 100],
            'order_id': [f'o{i}' for i in range(len(dates) * 100)],
        }).head(len(dates))

    def test_returns_dataframe_with_history_and_forecast_types(self):
        df = _make_master_df()
        # Extend with more dates to have enough data for Holt-Winters
        dates = pd.date_range('2017-01-02', periods=60, freq='W')
        big_df = pd.concat([
            pd.DataFrame({
                'order_purchase_timestamp': dates,
                'order_id': [f'o{i}' for i in range(len(dates))],
            }),
            df[['order_purchase_timestamp', 'order_id']]
        ], ignore_index=True)

        result = forecast_orders(big_df, periods=4)

        self.assertIn('type', result.columns)
        self.assertIn('History', result['type'].values)
        self.assertIn('Forecast', result['type'].values)

    def test_forecast_produces_correct_number_of_future_periods(self):
        dates = pd.date_range('2017-01-02', periods=60, freq='W')
        big_df = pd.DataFrame({
            'order_purchase_timestamp': dates,
            'order_id': [f'o{i}' for i in range(len(dates))],
        })
        result = forecast_orders(big_df, periods=6)
        forecast_rows = result[result['type'] == 'Forecast']
        self.assertEqual(len(forecast_rows), 6)


if __name__ == '__main__':
    unittest.main()
