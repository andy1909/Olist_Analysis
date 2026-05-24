import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

# Cấu hình
sns.set_theme(style="whitegrid")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'Outputs')
CHARTS_DIR = os.path.join(OUTPUT_DIR, 'Charts')
os.makedirs(CHARTS_DIR, exist_ok=True)

# Ngưỡng p-value để xác định tính dừng
P_VALUE_THRESHOLD = 0.05

def prepare_timeseries_data(master_data_path):
    """
    Nạp và chuẩn bị dữ liệu chuỗi thời gian hàng tuần.
    """
    print(">>> [1] Đang chuẩn bị dữ liệu Time Series...")
    df = pd.read_csv(master_data_path, parse_dates=['order_purchase_timestamp'])
    df_ts = df.set_index('order_purchase_timestamp')
    weekly_orders = df_ts.resample('W')['order_id'].count().fillna(0)
    weekly_orders = weekly_orders[weekly_orders.index >= '2017-01-01']
    weekly_orders = weekly_orders.asfreq('W').fillna(0).astype(float)
    return weekly_orders

def test_stationarity(series, series_name=""):
    """
    Thực hiện ADF test và in kết quả một cách dễ hiểu.
    """
    print(f"\n--- Kiểm định ADF cho: '{series_name}' ---")
    # Bỏ qua các giá trị NaN để adfuller không lỗi
    result = adfuller(series.dropna())
    p_value = result[1]

    print(f'    ADF Statistic: {result[0]:.4f}')
    print(f'    p-value: {p_value:.10f}')

    if p_value <= P_VALUE_THRESHOLD:
        print(f'    -> Kết luận: Chuỗi DỪNG (Stationary) (p-value <= {P_VALUE_THRESHOLD})')
    else:
        print(f'    -> Kết luận: Chuỗi KHÔNG DỪNG (Non-Stationary) (p-value > {P_VALUE_THRESHOLD})')

    return p_value

def main():
    print("====== STARTING STATIONARITY ANALYSIS PIPELINE ======")
    master_data_path = os.path.join(OUTPUT_DIR, 'Master_Logistics_Data.csv')

    if not os.path.exists(master_data_path):
        print(f"LỖI: Không tìm thấy file {master_data_path}. Hãy chạy 'main.py' trước.")
        return

    # 1. Chuẩn bị dữ liệu
    weekly_orders = prepare_timeseries_data(master_data_path)

    # 2. Thực hiện phân tích và lưu vào DataFrame
    df_analysis = pd.DataFrame({'original': weekly_orders})
    df_analysis['diff_1'] = df_analysis['original'].diff()
    df_analysis['diff_2'] = df_analysis['diff_1'].diff()

    # Kiểm định cho từng chuỗi
    test_stationarity(df_analysis['original'], 'Dữ liệu gốc (d=0)')
    test_stationarity(df_analysis['diff_1'], 'Sai phân bậc 1 (d=1)')
    test_stationarity(df_analysis['diff_2'], 'Sai phân bậc 2 (d=2)')

    # 3. Lưu bảng dữ liệu CSV
    analysis_path = os.path.join(OUTPUT_DIR, 'Stationarity_Analysis.csv')
    df_analysis.to_csv(analysis_path)
    print(f"\n>>> [2] Đã lưu bảng phân tích các bậc sai phân tại: {analysis_path}")

    # 4. Trực quan hóa
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Biểu đồ gốc
    axes[0].plot(df_analysis.index, df_analysis['original'], label='Original Series (d=0)')
    axes[0].set_title('Dữ liệu gốc (Original Weekly Orders)', fontsize=14)
    axes[0].legend()

    # Biểu đồ sai phân bậc 1
    axes[1].plot(df_analysis.index, df_analysis['diff_1'], label='First-Order Differencing (d=1)', color='orange')
    axes[1].hlines(0, xmin=df_analysis.index.min(), xmax=df_analysis.index.max(), linestyles='--', color='gray')
    axes[1].set_title('Sau khi Sai phân bậc 1', fontsize=14)
    axes[1].legend()

    # Biểu đồ sai phân bậc 2
    axes[2].plot(df_analysis.index, df_analysis['diff_2'], label='Second-Order Differencing (d=2)', color='green')
    axes[2].hlines(0, xmin=df_analysis.index.min(), xmax=df_analysis.index.max(), linestyles='--', color='gray')
    axes[2].set_title('Sau khi Sai phân bậc 2', fontsize=14)
    axes[2].set_xlabel('Date')
    axes[2].legend()

    plt.tight_layout()
    chart_path = os.path.join(CHARTS_DIR, 'Chart_4_Stationarity_Analysis.png')
    plt.savefig(chart_path)
    plt.close()
    print(f">>> [3] Đã lưu biểu đồ phân tích tại: {chart_path}")

    # 5. PHÂN TÍCH TỰ ĐỘNG: TÌM BẬC SAI PHÂN TỐI ƯU
    print("\n--- PHÂN TÍCH TỰ ĐỘNG ---")
    temp_series = weekly_orders.copy()
    for d in range(3): # Kiểm tra từ d=0 đến d=2
        p_value = test_stationarity(temp_series, f'Kiểm tra tự động bậc d={d}')
        if p_value <= P_VALUE_THRESHOLD:
            print(f"\n✅ Đã đạt được tính dừng ở bậc sai phân d = {d}.")
            break
        temp_series = temp_series.diff()

    print("\n====== STATIONARITY ANALYSIS PIPELINE COMPLETE ======")

if __name__ == "__main__":
    main()
