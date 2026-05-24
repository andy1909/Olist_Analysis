# modules/data_analytics.py
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

# Tắt cảnh báo để output sạch hơn
warnings.filterwarnings("ignore")

def aggregate_kpis_for_dashboard(df):
    """
    Tính toán các KPI quan trọng cho Logistics để vẽ Dashboard.
    Trả về 2 bảng: Theo Bang (Geo) và Theo Thời gian (Trend).
    """
    print(">>> [Analytics] Đang tổng hợp KPI cho Dashboard...")

    # 1. KPI theo Bang (State Performance)
    kpi_state = df.groupby('customer_state').agg(
        total_orders=('order_id', 'count'),
        avg_freight_value=('freight_value', 'mean'),
        avg_lead_time_days=('lead_time_days', 'mean'),
        total_late_orders=('is_late', 'sum')
    ).reset_index()

    # Tính % trễ (Late Rate)
    # Xử lý chia cho 0 nếu có
    if kpi_state['total_orders'].sum() > 0:
        kpi_state['late_rate_percent'] = (kpi_state['total_late_orders'] / kpi_state['total_orders']) * 100
    else:
        kpi_state['late_rate_percent'] = 0

    # 2. KPI theo Tháng (Monthly Trend)
    # Cần cột tháng năm (YYYY-MM)
    df['order_month'] = df['order_purchase_timestamp'].dt.to_period('M')

    kpi_month = df.groupby('order_month').agg(
        total_orders=('order_id', 'count'),
        revenue=('price', 'sum'),
        avg_lead_time=('lead_time_days', 'mean')
    ).reset_index()

    kpi_month['order_month'] = kpi_month['order_month'].astype(str) # Convert về string để lưu CSV

    return kpi_state, kpi_month

def forecast_orders(df, periods=12):
    """
    Sử dụng Time-Series (Holt-Winters) để dự báo số lượng đơn hàng theo tuần.
    """
    print(">>> [Analytics] Đang chạy mô hình dự báo Time-Series (Holt-Winters)...")

    # 1. Chuẩn bị dữ liệu: Resample theo Tuần (Weekly)
    df_ts = df.set_index('order_purchase_timestamp')

    # Resample theo tuần (W) và điền 0 vào tuần không có đơn
    weekly_orders = df_ts.resample('W')['order_id'].count().fillna(0)

    # Lọc lấy dữ liệu từ 2017 trở đi để mô hình ổn định
    weekly_orders = weekly_orders[weekly_orders.index >= '2017-01-01']

    # [QUAN TRỌNG] Ép lại tần suất là Weekly (W) để Statsmodels hiểu
    # Nếu không có dòng này, sau khi lọc index, Pandas sẽ làm mất 'freq'
    weekly_orders = weekly_orders.asfreq('W').fillna(0)

    # Chuyển sang kiểu float để tránh lỗi tính toán
    weekly_orders = weekly_orders.astype(float)

    try:
        # 2. Xây dựng mô hình Holt-Winters
        # initialization_method='estimated': Giúp mô hình tự tìm tham số khởi tạo tốt nhất
        model = ExponentialSmoothing(
            weekly_orders,
            seasonal_periods=12,
            trend='add',
            seasonal='add',
            damped_trend=True,
            initialization_method='estimated'
        ).fit(optimized=True)

        # 3. Dự báo tương lai
        forecast_values = model.forecast(periods)

        # 4. Gom kết quả lại để xuất file
        history_df = pd.DataFrame({'date': weekly_orders.index, 'order_count': weekly_orders.values, 'type': 'History'})

        # Tạo index cho tương lai
        future_dates = pd.date_range(start=weekly_orders.index[-1], periods=periods+1, freq='W')[1:]
        forecast_df = pd.DataFrame({'date': future_dates, 'order_count': forecast_values.values, 'type': 'Forecast'})

        # Nối lại
        final_forecast = pd.concat([history_df, forecast_df])

        print("    -> Đã dự báo thành công 12 tuần tiếp theo.")
        return final_forecast

    except Exception as e:
        print(f"    ! LỖI NGHIÊM TRỌNG KHI CHẠY MODEL: {e}")
        import traceback
        traceback.print_exc() # In chi tiết lỗi ra màn hình để debug

        # Trả về dữ liệu gốc nếu lỗi mô hình
        return pd.DataFrame({'date': weekly_orders.index, 'order_count': weekly_orders.values, 'type': 'History'})
