import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Thư viện cho Time Series
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Thư viện cho Machine Learning (LSTM)
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Cấu hình
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'Outputs')
CHARTS_DIR = os.path.join(OUTPUT_DIR, 'Charts')
os.makedirs(CHARTS_DIR, exist_ok=True)

# --- Các tham số cho mô hình ---
FORECAST_PERIODS = 12  # Dự báo 12 tuần tới
N_STEPS = 10            # LSTM sẽ nhìn vào 4 tuần trước để dự đoán
N_FEATURES = 1         # Chỉ dự báo trên 1 biến (số lượng đơn hàng)
EPOCHS = 100           # Số lần lặp để train model

#=============================================================================#
# HÀM CHUẨN BỊ DỮ LIỆU (Dùng chung cho các model)
#=============================================================================#
def prepare_timeseries_data(master_data_path):
    """
    Nạp và chuẩn bị dữ liệu chuỗi thời gian hàng tuần từ file master.
    """
    print(">>> [1] Đang chuẩn bị dữ liệu Time Series...")
    df = pd.read_csv(master_data_path, parse_dates=['order_purchase_timestamp'])
    df_ts = df.set_index('order_purchase_timestamp')
    weekly_orders = df_ts.resample('W')['order_id'].count().fillna(0)
    weekly_orders = weekly_orders[weekly_orders.index >= '2017-01-01']
    weekly_orders = weekly_orders.asfreq('W').fillna(0).astype(float)
    print("    -> Dữ liệu hàng tuần đã sẵn sàng.")
    return weekly_orders

#=============================================================================#
# CÁC HÀM DỰ BÁO
#=============================================================================#

def naive_forecast(series, periods):
    """
    Mô hình Naive: Dự báo cho kỳ tới bằng giá trị của kỳ cuối cùng.
    """
    print(">>> [Model] Đang chạy Naive Forecast (Baseline)...")
    last_value = series.iloc[-1]
    return np.full(periods, last_value)

def holt_winters_forecast(series, periods):
    """
    Mô hình Holt-Winters (lấy từ file cũ để so sánh).
    """
    print(">>> [Model] Đang chạy Holt-Winters Forecast...")
    try:
        model = ExponentialSmoothing(
            series, seasonal_periods=12, trend='add', seasonal='add',
            damped_trend=True, initialization_method='estimated'
        ).fit(optimized=True)
        return model.forecast(periods)
    except Exception as e:
        print(f"    ! Lỗi Holt-Winters: {e}")
        return np.full(periods, series.mean()) # Trả về giá trị trung bình nếu lỗi

def lstm_forecast(series, periods, n_steps, n_features, epochs):
    """
    Mô hình LSTM: Sử dụng mạng nơ-ron để dự báo.
    """
    print(">>> [Model] Đang chạy LSTM Forecast...")

    # --- 1. Kiểm tra tính dừng (Stationarity) ---
    adf_result = adfuller(series)
    print(f'    - ADF Statistic: {adf_result[0]}')
    print(f'    - p-value: {adf_result[1]}')
    if adf_result[1] > 0.05:
        print("    -> Dữ liệu không dừng (p > 0.05). Sẽ áp dụng Sai phân (Differencing).")
        # Lưu lại giá trị cuối cùng để khôi phục sau này
        last_original_value = series.iloc[-1]
        series_diff = series.diff().dropna()
        is_differenced = True
    else:
        print("    -> Dữ liệu đã dừng (p <= 0.05).")
        series_diff = series
        is_differenced = False

    # --- 2. Chuẩn hóa dữ liệu ---
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(series_diff.values.reshape(-1, 1))

    # --- 3. Tạo chuỗi dữ liệu (Generator) ---
    generator = TimeseriesGenerator(scaled_data, scaled_data, length=n_steps, batch_size=1)

    # --- 4. Xây dựng và Huấn luyện mô hình LSTM ---
    print("    - Đang xây dựng và huấn luyện LSTM...")
    model = Sequential([
        LSTM(15, activation='relu', input_shape=(n_steps, n_features)),
        Dropout(0.5),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(generator, epochs=epochs, verbose=0)

    # --- 5. Dự báo lặp (Iterative Forecasting) ---
    print("    - Đang thực hiện dự báo...")
    forecast = []
    current_batch = scaled_data[-n_steps:].reshape((1, n_steps, n_features))

    for i in range(periods):
        current_pred = model.predict(current_batch, verbose=0)[0]
        forecast.append(current_pred)
        # Cập nhật batch: bỏ giá trị cũ nhất, thêm giá trị dự báo mới
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    # --- 6. Khôi phục dữ liệu về thang đo gốc ---
    # a. Inverse Scale
    forecast_inversed = scaler.inverse_transform(forecast)

    # b. Inverse Difference (Nếu đã sai phân)
    if is_differenced:
        final_forecast = np.cumsum(forecast_inversed) + last_original_value
    else:
        final_forecast = forecast_inversed.flatten()

    return final_forecast

#=============================================================================#
# HÀM CHÍNH ĐỂ THỰC THI VÀ SO SÁNH
#=============================================================================#
def main():
    print("====== STARTING ADVANCED FORECASTING PIPELINE ======")
    master_data_path = os.path.join(OUTPUT_DIR, 'Master_Logistics_Data.csv')

    if not os.path.exists(master_data_path):
        print(f"LỖI: Không tìm thấy file {master_data_path}. Hãy chạy 'main.py' trước.")
        return

    # 1. Chuẩn bị dữ liệu
    weekly_orders = prepare_timeseries_data(master_data_path)

    # 2. Chạy các mô hình
    forecast_naive = naive_forecast(weekly_orders, FORECAST_PERIODS)
    forecast_hw = holt_winters_forecast(weekly_orders, FORECAST_PERIODS)
    forecast_lstm = lstm_forecast(weekly_orders, FORECAST_PERIODS, N_STEPS, N_FEATURES, EPOCHS)

    # 3. Tạo DataFrame kết quả để so sánh
    future_dates = pd.date_range(start=weekly_orders.index[-1], periods=FORECAST_PERIODS + 1, freq='W')[1:]

    df_forecast = pd.DataFrame({
        'date': future_dates,
        'Naive_Forecast': forecast_naive,
        'HoltWinters_Forecast': forecast_hw.values,
        'LSTM_Forecast': forecast_lstm
    })

    # Lưu kết quả ra CSV
    comparison_path = os.path.join(OUTPUT_DIR, 'Forecast_Comparison_Results.csv')
    df_forecast.to_csv(comparison_path, index=False)
    print(f"\n>>> [4] Đã lưu kết quả so sánh tại: {comparison_path}")

    # 4. Trực quan hóa so sánh
    plt.figure(figsize=(16, 8))
    plt.plot(weekly_orders.index, weekly_orders.values, label='Historical Data', color='gray')
    plt.plot(df_forecast['date'], df_forecast['Naive_Forecast'], label='Naive (Baseline)', linestyle=':')
    plt.plot(df_forecast['date'], df_forecast['HoltWinters_Forecast'], label='Holt-Winters', linestyle='--')
    plt.plot(df_forecast['date'], df_forecast['LSTM_Forecast'], label='LSTM', linestyle='-', color='red', linewidth=2)

    plt.title('Comparison of Forecasting Models', fontsize=18)
    plt.xlabel('Date')
    plt.ylabel('Weekly Orders')
    plt.legend()
    plt.grid(True)

    chart_path = os.path.join(CHARTS_DIR, 'Chart_5_Forecast_Comparison.png')
    plt.savefig(chart_path)
    plt.close()
    print(f">>> [5] Đã lưu biểu đồ so sánh tại: {chart_path}")
    print("\n====== ADVANCED FORECASTING PIPELINE COMPLETE ======")


if __name__ == "__main__":
    main()
