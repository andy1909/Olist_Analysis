# src/analytics.py
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings("ignore")

def aggregate_kpis_for_dashboard(df):
    """
    Compute key logistics KPIs for dashboards.
    Returns:
        kpi_state: Geo-performance table (aggregated by state)
        kpi_month: Temporal trend table (aggregated by month)
    """
    print(">>> [Analytics] Aggregating KPIs for dashboard...")

    # 1. State Performance
    kpi_state = df.groupby('customer_state').agg(
        total_orders=('order_id', 'count'),
        avg_freight_value=('freight_value', 'mean'),
        avg_lead_time_days=('lead_time_days', 'mean'),
        total_late_orders=('is_late', 'sum')
    ).reset_index()

    if kpi_state['total_orders'].sum() > 0:
        kpi_state['late_rate_percent'] = (kpi_state['total_late_orders'] / kpi_state['total_orders']) * 100
    else:
        kpi_state['late_rate_percent'] = 0

    # 2. Monthly Trend
    df['order_month'] = df['order_purchase_timestamp'].dt.to_period('M')
    kpi_month = df.groupby('order_month').agg(
        total_orders=('order_id', 'count'),
        revenue=('price', 'sum'),
        avg_lead_time=('lead_time_days', 'mean')
    ).reset_index()

    kpi_month['order_month'] = kpi_month['order_month'].astype(str)

    return kpi_state, kpi_month

def forecast_orders(df, periods=12):
    """
    Use Holt-Winters Exponential Smoothing to forecast weekly order demand.
    """
    print(">>> [Analytics] Running Holt-Winters Exponential Smoothing model...")

    df_ts = df.set_index('order_purchase_timestamp')
    weekly_orders = df_ts.resample('W')['order_id'].nunique().fillna(0)
    weekly_orders = weekly_orders[weekly_orders.index >= '2017-01-01']
    weekly_orders = weekly_orders.asfreq('W').fillna(0).astype(float)

    try:
        model = ExponentialSmoothing(
            weekly_orders,
            seasonal_periods=12,
            trend='add',
            seasonal='add',
            damped_trend=True,
            initialization_method='estimated'
        ).fit(optimized=True)

        forecast_values = model.forecast(periods)

        history_df = pd.DataFrame({'date': weekly_orders.index, 'order_count': weekly_orders.values, 'type': 'History'})
        future_dates = pd.date_range(start=weekly_orders.index[-1], periods=periods+1, freq='W')[1:]
        forecast_df = pd.DataFrame({'date': future_dates, 'order_count': forecast_values.values, 'type': 'Forecast'})

        final_forecast = pd.concat([history_df, forecast_df])
        print(f"    - Forecasted {periods} weeks ahead successfully.")
        return final_forecast

    except Exception as e:
        print(f"    ! Error during Holt-Winters forecasting: {e}")
        return pd.DataFrame({'date': weekly_orders.index, 'order_count': weekly_orders.values, 'type': 'History'})
