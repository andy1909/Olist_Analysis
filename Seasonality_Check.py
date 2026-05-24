import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 1. Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'Outputs')
CHARTS_DIR = os.path.join(OUTPUT_DIR, 'Charts')
os.makedirs(CHARTS_DIR, exist_ok=True)

def analyze_autocorrelation(master_data_path):
    print(">>> [1] Đang nạp dữ liệu và chuẩn bị chuỗi thời gian...")
    df = pd.read_csv(master_data_path, parse_dates=['order_purchase_timestamp'])

    # Tổng hợp theo tuần (nunique order_id như đã thống nhất)
    series = df.set_index('order_purchase_timestamp').resample('W')['order_id'].nunique().fillna(0)
    series = series[series.index >= '2017-01-01']

    print(f"    -> Số lượng quan sát (tuần): {len(series)}")

    # 2. Tính toán giá trị ACF và PACF (tối đa 26 tuần - khoảng nửa năm)
    n_lags = 26
    acf_values = acf(series, nlags=n_lags)
    pacf_values = pacf(series, nlags=n_lags)

    # 3. Xuất dữ liệu ra file CSV để biện luận
    df_corr = pd.DataFrame({
        'Lag': np.arange(len(acf_values)),
        'ACF': acf_values,
        'PACF': pacf_values
    })
    csv_path = os.path.join(OUTPUT_DIR, 'Autocorrelation_Analysis_Results.csv')
    df_corr.to_csv(csv_path, index=False)
    print(f">>> [2] Đã lưu kết quả chỉ số tại: {csv_path}")

    # 4. Trực quan hóa Biểu đồ
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Biểu đồ ACF: Dùng để xác định tính mùa vụ (Seasonality)
    plot_acf(series, lags=n_lags, ax=ax1, color='blue', vlines_kwargs={"colors": 'blue'})
    ax1.set_title('Autocorrelation Function (ACF) - Chứng minh tính mùa vụ', fontsize=14)
    ax1.set_xlabel('Lags (Weeks)')
    ax1.set_ylabel('Correlation')

    # Biểu đồ PACF: Dùng để xác định bậc của mô hình AR trong LSTM/ARIMA
    plot_pacf(series, lags=n_lags, ax=ax2, color='red', vlines_kwargs={"colors": 'red'})
    ax2.set_title('Partial Autocorrelation Function (PACF) - Xác định phụ thuộc ngắn hạn', fontsize=14)
    ax2.set_xlabel('Lags (Weeks)')
    ax2.set_ylabel('Correlation Coefficient')

    plt.tight_layout()
    chart_path = os.path.join(CHARTS_DIR, 'Chart_7_Autocorrelation_Proof.png')
    plt.savefig(chart_path)
    plt.show()
    print(f">>> [3] Đã lưu biểu đồ kiểm định tại: {chart_path}")

    # 5. Phân tích nhanh kết quả
    print("\n--- PHÂN TÍCH CHUYÊN GIA ---")
    significant_acf = df_corr[df_corr['ACF'].abs() > 0.2]['Lag'].tolist()
    if 4 in significant_acf or 5 in significant_acf:
        print("✅ Kết quả: Có tín hiệu tự tương quan tại Lag 4-5 (tương đương 1 tháng).")
        print("   => Bằng chứng toán học cho tính Mùa vụ theo tháng (Monthly Seasonality).")
    if 13 in significant_acf:
        print("✅ Kết quả: Có tín hiệu tự tương quan tại Lag 13 (tương đương 1 quý).")
        print("   => Bằng chứng cho tính chu kỳ Quý (Quarterly Seasonality).")

if __name__ == "__main__":
    master_path = os.path.join(OUTPUT_DIR, 'Master_Logistics_Data.csv')
    if os.path.exists(master_path):
        analyze_autocorrelation(master_path)
    else:
        print("LỖI: Không tìm thấy file Master_Logistics_Data.csv")
