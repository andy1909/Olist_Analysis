# tests/test_analytics.py
import unittest
import pandas as pd
from src.models.analytics import aggregate_kpis_for_dashboard

class TestAnalytics(unittest.TestCase):
    def test_aggregate_kpis_for_dashboard(self):
        df = pd.DataFrame({
            'customer_state': ['SP', 'SP', 'RJ', 'RJ'],
            'order_id': ['o1', 'o2', 'o3', 'o4'],
            'freight_value': [10.0, 12.0, 20.0, 30.0],
            'lead_time_days': [5, 6, 12, 16],
            'is_late': [0, 0, 1, 0],
            'price': [100.0, 150.0, 200.0, 250.0],
            'order_purchase_timestamp': pd.to_datetime([
                '2018-01-01 10:00:00',
                '2018-01-15 10:00:00',
                '2018-01-10 10:00:00',
                '2018-02-05 10:00:00'
            ])
        })
        
        kpi_state, kpi_month = aggregate_kpis_for_dashboard(df)
        
        # State KPIs assertions
        self.assertEqual(len(kpi_state), 2)
        sp_kpi = kpi_state[kpi_state['customer_state'] == 'SP'].iloc[0]
        rj_kpi = kpi_state[kpi_state['customer_state'] == 'RJ'].iloc[0]
        
        self.assertEqual(sp_kpi['total_orders'], 2)
        self.assertEqual(sp_kpi['avg_freight_value'], 11.0)
        self.assertEqual(sp_kpi['avg_lead_time_days'], 5.5)
        self.assertEqual(sp_kpi['late_rate_percent'], 0.0)
        
        self.assertEqual(rj_kpi['total_orders'], 2)
        self.assertEqual(rj_kpi['avg_freight_value'], 25.0)
        self.assertEqual(rj_kpi['avg_lead_time_days'], 14.0)
        self.assertEqual(rj_kpi['late_rate_percent'], 50.0)
        
        # Monthly KPIs assertions
        self.assertEqual(len(kpi_month), 2)
        jan_kpi = kpi_month[kpi_month['order_month'] == '2018-01'].iloc[0]
        feb_kpi = kpi_month[kpi_month['order_month'] == '2018-02'].iloc[0]
        
        self.assertEqual(jan_kpi['total_orders'], 3)
        self.assertEqual(jan_kpi['revenue'], 450.0)
        self.assertEqual(feb_kpi['total_orders'], 1)
        self.assertEqual(feb_kpi['revenue'], 250.0)

if __name__ == '__main__':
    unittest.main()
