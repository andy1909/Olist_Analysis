# modules/visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Cấu hình giao diện biểu đồ cho đẹp
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def plot_monthly_trend(kpi_month_path, output_dir):
    """
    Vẽ biểu đồ xu hướng đơn hàng và thời gian giao hàng theo tháng.
    """
    print(">>> [Viz] Đang vẽ biểu đồ xu hướng tháng (Monthly Trend)...")
    if not os.path.exists(kpi_month_path): return

    df = pd.read_csv(kpi_month_path)

    fig, ax1 = plt.subplots()

    # Trục trái: Số lượng đơn hàng (Bar chart)
    sns.barplot(data=df, x='order_month', y='total_orders', color='skyblue', alpha=0.6, ax=ax1)
    ax1.set_ylabel('Total Orders', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

    # Trục phải: Thời gian giao hàng trung bình (Line chart)
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x='order_month', y='avg_lead_time', color='red', marker='o', ax=ax2, linewidth=2)
    ax2.set_ylabel('Avg Lead Time (Days)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Logistics Trend: Orders Volume vs. Delivery Speed', fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'Chart_1_Monthly_Trend.png')
    plt.savefig(save_path)
    plt.close()
    print(f"    -> Đã lưu: {save_path}")

def plot_state_performance(kpi_state_path, output_dir):
    """
    Vẽ biểu đồ so sánh hiệu quả giao hàng giữa các bang (Top 10).
    """
    print(">>> [Viz] Đang vẽ biểu đồ hiệu quả theo bang...")
    if not os.path.exists(kpi_state_path): return

    df = pd.read_csv(kpi_state_path)
    # Lấy Top 10 bang nhiều đơn nhất để vẽ cho đỡ rối
    df_top = df.nlargest(10, 'total_orders')

    # Vẽ biểu đồ kết hợp: Cột là % Trễ, Màu sắc là Phí ship
    plt.figure(figsize=(12, 6))
    chart = sns.barplot(
        data=df_top,
        x='customer_state',
        y='late_rate_percent',
        palette='Reds'
    )

    # Thêm text lên cột
    for p in chart.patches:
        chart.annotate(f'{p.get_height():.1f}%',
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha = 'center', va = 'center',
                       xytext = (0, 9),
                       textcoords = 'offset points')

    plt.title('Top 10 States: Late Delivery Rate (%)', fontsize=16)
    plt.ylabel('Late Rate (%)')
    plt.xlabel('State')

    save_path = os.path.join(output_dir, 'Chart_2_State_Late_Rate.png')
    plt.savefig(save_path)
    plt.close()
    print(f"    -> Đã lưu: {save_path}")

def plot_forecast(forecast_path, output_dir):
    """
    Vẽ biểu đồ dự báo (Forecast) từ kết quả bước 3.
    """
    print(">>> [Viz] Đang vẽ biểu đồ dự báo (Forecast)...")
    if not os.path.exists(forecast_path): return

    df = pd.read_csv(forecast_path)
    df['date'] = pd.to_datetime(df['date'])

    plt.figure(figsize=(14, 7))

    # Vẽ đường lịch sử
    sns.lineplot(data=df[df['type']=='History'], x='date', y='order_count', label='Historical Data', color='gray')

    # Vẽ đường dự báo (nổi bật)
    sns.lineplot(data=df[df['type']=='Forecast'], x='date', y='order_count', label='Forecast (Next 12 Weeks)', color='green', linewidth=3, linestyle='--')

    plt.title('Demand Forecasting: Order Volume Prediction', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Weekly Orders')
    plt.legend()

    save_path = os.path.join(output_dir, 'Chart_3_Forecast.png')
    plt.savefig(save_path)
    plt.close()
    print(f"    -> Đã lưu: {save_path}")
