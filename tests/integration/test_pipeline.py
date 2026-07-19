# tests/integration/test_pipeline.py
"""
Integration tests for the Olist Logistics Pipeline.

Unlike unit tests (which test one function in isolation), these tests
exercise multiple src/ modules working together end-to-end with a
realistic — but minimal — in-memory dataset.

No real files from data/raw/ are loaded.  No external APIs are called.
All I/O is performed inside a temporary directory that is cleaned up
after the test class finishes.

Test scope:
    Processing → Analytics     : clean data flows into correct KPIs
    Processing → Visualization : KPI CSVs produce valid PNG charts
"""
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd


def _build_master_df(n_weeks: int = 70, seed: int = 42) -> pd.DataFrame:
    """
    Build a minimal but realistic master DataFrame that mirrors the output
    of processing.create_logistics_features() and processing.merge_data().

    Parameters
    ----------
    n_weeks : int
        Number of weekly data points (orders) to generate.
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start='2017-01-02', periods=n_weeks, freq='W')

    states = rng.choice(['SP', 'RJ', 'MG', 'RS', 'PR'], size=n_weeks)
    prices = rng.uniform(50, 500, size=n_weeks)
    freight = rng.uniform(5, 60, size=n_weeks)
    lead_times = rng.integers(5, 25, size=n_weeks).astype(float)
    is_late = (rng.random(n_weeks) > 0.88).astype(int)

    return pd.DataFrame({
        'order_id':                   [f'o{i:04d}' for i in range(n_weeks)],
        'customer_id':                [f'c{i:04d}' for i in range(n_weeks)],
        'seller_id':                  [f's{i % 10:02d}' for i in range(n_weeks)],
        'product_id':                 [f'p{i % 20:02d}' for i in range(n_weeks)],
        'order_item_id':              rng.integers(1, 4, size=n_weeks),
        'order_purchase_timestamp':   dates,
        'order_delivered_customer_date': dates + pd.to_timedelta(lead_times, unit='D'),
        'customer_state':             states,
        'price':                      prices,
        'freight_value':              freight,
        'lead_time_days':             lead_times,
        'is_late':                    is_late,
    })


class TestProcessingToAnalyticsPipeline(unittest.TestCase):
    """
    Verifies that a cleaned master DataFrame flows correctly through
    aggregate_kpis_for_dashboard() and produces valid KPI outputs.
    """

    @classmethod
    def setUpClass(cls):
        """Build the shared master DataFrame once for the whole class."""
        cls.master_df = _build_master_df()

    def test_kpi_state_contains_all_five_states(self):
        from src.models.analytics import aggregate_kpis_for_dashboard
        kpi_state, _ = aggregate_kpis_for_dashboard(self.master_df)
        self.assertEqual(len(kpi_state), 5)

    def test_kpi_state_late_rate_is_between_0_and_100_for_all_states(self):
        from src.models.analytics import aggregate_kpis_for_dashboard
        kpi_state, _ = aggregate_kpis_for_dashboard(self.master_df)
        self.assertTrue((kpi_state['late_rate_percent'] >= 0).all())
        self.assertTrue((kpi_state['late_rate_percent'] <= 100).all())

    def test_kpi_month_total_revenue_equals_sum_of_all_prices(self):
        from src.models.analytics import aggregate_kpis_for_dashboard
        _, kpi_month = aggregate_kpis_for_dashboard(self.master_df)
        self.assertAlmostEqual(
            kpi_month['revenue'].sum(),
            self.master_df['price'].sum(),
            places=2,
        )

    def test_kpi_month_total_orders_equals_nunique_order_ids(self):
        from src.models.analytics import aggregate_kpis_for_dashboard
        _, kpi_month = aggregate_kpis_for_dashboard(self.master_df)
        self.assertEqual(
            kpi_month['total_orders'].sum(),
            self.master_df['order_id'].nunique(),
        )

    def test_kpi_state_avg_lead_time_is_positive_for_all_states(self):
        from src.models.analytics import aggregate_kpis_for_dashboard
        kpi_state, _ = aggregate_kpis_for_dashboard(self.master_df)
        self.assertTrue((kpi_state['avg_lead_time_days'] > 0).all())


class TestProcessingToVisualizationPipeline(unittest.TestCase):
    """
    Verifies that KPI DataFrames produced by analytics can be serialised
    and then consumed by visualization functions to produce valid PNG files.
    """

    @classmethod
    def setUpClass(cls):
        """Prepare KPI CSVs and output directory once for the whole class."""
        from src.models.analytics import aggregate_kpis_for_dashboard

        cls.tmp_dir = tempfile.mkdtemp()
        master_df = _build_master_df()
        kpi_state, kpi_month = aggregate_kpis_for_dashboard(master_df)

        cls.kpi_state_path = os.path.join(cls.tmp_dir, 'KPI_by_State.csv')
        cls.kpi_month_path = os.path.join(cls.tmp_dir, 'KPI_by_Month.csv')
        kpi_state.to_csv(cls.kpi_state_path, index=False)
        kpi_month.to_csv(cls.kpi_month_path, index=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_monthly_trend_chart_is_generated_from_analytics_output(self):
        from src.utils.visualization import plot_monthly_trend
        plot_monthly_trend(self.kpi_month_path, self.tmp_dir)
        chart_path = os.path.join(self.tmp_dir, 'Chart_1_Monthly_Trend.png')
        self.assertTrue(os.path.exists(chart_path))
        self.assertGreater(os.path.getsize(chart_path), 0)

    def test_state_performance_chart_is_generated_from_analytics_output(self):
        from src.utils.visualization import plot_state_performance
        plot_state_performance(self.kpi_state_path, self.tmp_dir)
        chart_path = os.path.join(self.tmp_dir, 'Chart_2_State_Late_Rate.png')
        self.assertTrue(os.path.exists(chart_path))
        self.assertGreater(os.path.getsize(chart_path), 0)


class TestLogisticsFeaturesToKpiConsistency(unittest.TestCase):
    """
    Verifies numerical consistency between raw feature values and the
    KPIs derived from them — i.e. the aggregation math is end-to-end correct.
    """

    def setUp(self):
        # Controlled, deterministic dataset for precise assertions
        self.df = pd.DataFrame({
            'order_id':                   ['o1', 'o2', 'o3', 'o4'],
            'customer_state':             ['SP',  'SP',  'RJ',  'RJ' ],
            'price':                      [100.,  200.,  150.,  250. ],
            'freight_value':              [10.,   20.,   15.,   30.  ],
            'lead_time_days':             [5.,    10.,   20.,   25.  ],
            'is_late':                    [0,     0,     1,     0    ],
            'order_purchase_timestamp':   pd.to_datetime(
                ['2018-01-05', '2018-01-12', '2018-01-08', '2018-02-03']
            ),
        })

    def test_sp_revenue_equals_manual_sum(self):
        from src.models.analytics import aggregate_kpis_for_dashboard
        _, kpi_month = aggregate_kpis_for_dashboard(self.df)
        jan = kpi_month[kpi_month['order_month'] == '2018-01'].iloc[0]
        # Jan orders: o1(100) + o2(200) + o3(150) = 450
        self.assertAlmostEqual(jan['revenue'], 450.0)

    def test_rj_late_rate_calculated_correctly_across_pipeline(self):
        from src.models.analytics import aggregate_kpis_for_dashboard
        kpi_state, _ = aggregate_kpis_for_dashboard(self.df)
        rj = kpi_state[kpi_state['customer_state'] == 'RJ'].iloc[0]
        # RJ: 1 late out of 2 → 50%
        self.assertAlmostEqual(rj['late_rate_percent'], 50.0)

    def test_overall_order_count_is_preserved_through_aggregation(self):
        from src.models.analytics import aggregate_kpis_for_dashboard
        _, kpi_month = aggregate_kpis_for_dashboard(self.df)
        total_from_kpi = kpi_month['total_orders'].sum()
        self.assertEqual(total_from_kpi, self.df['order_id'].nunique())


if __name__ == '__main__':
    unittest.main()
