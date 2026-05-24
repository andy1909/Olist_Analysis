import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Cấu hình
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'Outputs')
CHARTS_DIR = os.path.join(OUTPUT_DIR, 'Charts')
os.makedirs(CHARTS_DIR, exist_ok=True)

# --- Các tham số ---
TEST_WEEKS = 12
VAL_WEEKS = 12
N_STEPS = 8      # Số tuần nhìn lại
EPOCHS = 200
FORECAST_WEEKS = 12 # Số tuần dự báo vào tương lai

def prepare_lstm_data(master_data_path):
    print(">>> [1] Đang chuẩn bị dữ liệu (Log, Diff, Outlier, Features)...")

    df = pd.read_csv(master_data_path, parse_dates=['order_purchase_timestamp'])
    df.set_index('order_purchase_timestamp', inplace=True)

    # Tổng hợp dữ liệu hàng tuần
    feature_cols = [
        'order_id', 'weekly_active_sellers', 'weekly_active_customers', 'weekly_product_variety',
        'weekly_weekly_gmv', 'weekly_avg_basket_size', 'weekly_avg_price', 'weekly_avg_freight_value'
    ]
    existing_cols = [col for col in feature_cols if col in df.columns]
    df_weekly = df[existing_cols].resample('W').first().fillna(0)
    order_counts = df['order_id'].resample('W').nunique()
    df_weekly['order_count'] = order_counts
    if 'order_id' in df_weekly.columns:
        df_weekly = df_weekly.drop(columns=['order_id'])
    df_weekly.rename(columns=lambda c: c.replace('weekly_', ''), inplace=True)

    df_weekly = df_weekly[df_weekly.index >= '2017-01-01']

    # Xử lý Outlier bằng Dummy Variable
    peak_date = df_weekly['order_count'].idxmax()
    df_weekly['black_friday_peak'] = 0
    df_weekly.loc[peak_date, 'black_friday_peak'] = 1

    # Biến đổi Log (log1p để xử lý giá trị 0)
    for col in df_weekly.columns:
        if col != 'black_friday_peak': # Không transform biến giả
            df_weekly[col] = np.log1p(df_weekly[col])

    # Lưu lại dữ liệu log để inverse transform sau này
    original_log_data = df_weekly.copy()

    # Làm dừng dữ liệu bằng Differencing
    for col in df_weekly.columns:
        if col != 'black_friday_peak':
            df_weekly[col] = df_weekly[col].diff()

    df_weekly.dropna(inplace=True)

    print("    -> Chuẩn bị dữ liệu V5 hoàn tất.")
    return df_weekly, original_log_data

def main():
    print("====== STARTING LSTM FORECASTING PIPELINE ======")
    master_data_path = os.path.join(OUTPUT_DIR, 'Master_Logistics_Data.csv')

    # 1. Chuẩn bị dữ liệu
    df_processed, df_log_original = prepare_lstm_data(master_data_path)

    # 2. Chia dữ liệu
    train_df = df_processed.iloc[:-(VAL_WEEKS + TEST_WEEKS)]
    val_df = df_processed.iloc[-(VAL_WEEKS + TEST_WEEKS):-TEST_WEEKS]
    test_df = df_processed.iloc[-TEST_WEEKS:]

    print(f"\n>>> [2] Kích thước dữ liệu: Train={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}")

    target_col = 'order_count'
    feature_cols = [col for col in df_processed.columns if col != target_col]

    # 3. Chuẩn hóa dữ liệu
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(train_df[feature_cols])
    y_train_scaled = scaler_y.fit_transform(train_df[[target_col]])
    X_val_scaled = scaler_X.transform(val_df[feature_cols])
    y_val_scaled = scaler_y.transform(val_df[[target_col]])

    # 4. Tạo Generator
    n_features = X_train_scaled.shape[1]
    train_generator = TimeseriesGenerator(X_train_scaled, y_train_scaled, length=N_STEPS, batch_size=1)
    val_generator = TimeseriesGenerator(X_val_scaled, y_val_scaled, length=N_STEPS, batch_size=1)

    # 5. Xây dựng và Huấn luyện mô hình
    print("\n>>> [3] Đang xây dựng và huấn luyện mô hình LSTM...")
    model = Sequential([
        LSTM(75, activation='relu', input_shape=(N_STEPS, n_features), return_sequences=True),
        Dropout(0.3),
        LSTM(50, activation='relu'),
        Dropout(0.3),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=25, mode='min', verbose=1, restore_best_weights=True)

    model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS, callbacks=[early_stopping], verbose=1)

    # 6. DỰ BÁO VÀO TƯƠNG LAI
    print(f"\n>>> [4] Đang thực hiện dự báo cho {FORECAST_WEEKS} tuần vào tương lai...")

    # Lấy toàn bộ dữ liệu đã xử lý để làm điểm khởi đầu
    full_processed_X = df_processed[feature_cols]
    scaled_full_X = scaler_X.transform(full_processed_X)

    future_predictions_scaled = []
    current_batch = scaled_full_X[-N_STEPS:].reshape((1, N_STEPS, n_features))

    for i in range(FORECAST_WEEKS):
        current_pred_scaled = model.predict(current_batch, verbose=0)[0]
        future_predictions_scaled.append(current_pred_scaled)

        # Giả định các feature khác không thay đổi (giữ nguyên giá trị differenced cuối cùng, thường gần 0)
        last_known_features_diff = current_batch[0, -1, :]

        # Tạo vector mới
        new_row = last_known_features_diff.reshape(1, 1, n_features)

        current_batch = np.append(current_batch[:, 1:, :], new_row, axis=1)

    # 7. KHÔI PHỤC KẾT QUẢ (2 BƯỚC)
    # Bước 7a: Inverse scale
    predictions_diff_log = scaler_y.inverse_transform(future_predictions_scaled)

    # Bước 7b: Inverse difference & Inverse log
    last_log_value = df_log_original['order_count'].iloc[-1]
    predictions_log = last_log_value + np.cumsum(predictions_diff_log)
    final_predictions = np.expm1(predictions_log)

    # 8. Trực quan hóa và Lưu kết quả
    print(">>> [5] Đang vẽ biểu đồ kết quả...")
    future_dates = pd.date_range(start=df_log_original.index[-1], periods=FORECAST_WEEKS + 1, freq='W')[1:]

    df_forecast = pd.DataFrame({'Predicted_Order_Count': final_predictions.flatten()}, index=future_dates)

    plt.figure(figsize=(18, 9))
    plt.plot(np.expm1(df_log_original['order_count']), label='Historical Data', color='gray')
    plt.plot(df_forecast.index, df_forecast['Predicted_Order_Count'], label=f'LSTM Forecast ({FORECAST_WEEKS} Weeks)', color='red', marker='x')

    plt.title('LSTM Future Forecast', fontsize=20)
    plt.xlabel('Date')
    plt.ylabel('Weekly Orders')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)

    chart_path = os.path.join(CHARTS_DIR, 'Chart_6_LSTM_Forecast.png')
    plt.savefig(chart_path)
    plt.close()
    print(f"    -> Đã lưu biểu đồ tại: {chart_path}")

    df_forecast.to_csv(os.path.join(OUTPUT_DIR, 'LSTM_Forecast_Results.csv'))
    print(f"    -> Đã lưu file kết quả tại: {os.path.join(OUTPUT_DIR, 'LSTM_Forecast_Results.csv')}")

    print("\n====== LSTM FORECASTING PIPELINE COMPLETE ======")


if __name__ == "__main__":
    main()
