# main.py
import os
import pandas as pd
from modules import data_ingestion, data_processing, data_analytics ,visualization

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'RawData')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Outputs')

# Tạo thư mục output nếu chưa có
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("=== STARTING DATA PIPELINE ===")

    # BƯỚC 1: DATA INGESTION (THU THẬP & CHUYỂN ĐỔI)
    # 1.1 Tạo file giả lập (JSON/XML)
    data_ingestion.convert_and_save_files(DATA_DIR)

    # 1.2 Lấy dữ liệu API
    df_holidays = data_ingestion.fetch_holidays_api()

    # 1.3 Load các dữ liệu lên
    df_orders, df_customers, df_products = data_ingestion.load_raw_data(DATA_DIR)

    # Kiểm tra nhanh
    print(f"\n--- Data Summary ---")
    print(f"Orders (CSV): {df_orders.shape}")
    print(f"Customers (JSON): {df_customers.shape}")
    print(f"Products (XML): {df_products.shape}")
    print(f"Holidays (API): {df_holidays.shape}")

    # Lưu ngày lễ ra file CSV
    df_holidays.to_csv(os.path.join(OUTPUT_DIR, 'brazil_holidays.csv'), index=False)
    print(f"\n-> Đã lưu file ngày lễ tại: {os.path.join(OUTPUT_DIR, 'brazil_holidays.csv')}")

    print("\n=== BƯỚC 1 HOÀN TẤT ===")


# # # ----------------------------------------------------------------------------------------------------------# # #
    # LOAD DỮ LIỆU ĐỂ CHUẨN BỊ CHO BƯỚC 2
    print("\n--- LOADING DATA ---")
    # 1. Load các file đã xử lý ở Bước 1
    orders, customers, products = data_ingestion.load_raw_data(DATA_DIR)

    # 2. Load thêm Order Items
    items_path = os.path.join(DATA_DIR, 'olist_order_items_dataset.csv')
    items = pd.read_csv(items_path)
    print(f"Loaded Order Items: {items.shape}")

    # 3. Load Holidays (từ file csv đã lưu ở Bước 1 để đỡ gọi API lại)
    holidays_path = os.path.join(OUTPUT_DIR, 'brazil_holidays.csv')
    if os.path.exists(holidays_path):
        holidays = pd.read_csv(holidays_path)
    else:
        # Fallback nếu chưa có file thì gọi hàm API rỗng hoặc xử lý lỗi
        print("Cảnh báo: Không tìm thấy file holidays, hãy chạy lại Bước 1.")
        holidays = pd.DataFrame()

    # --- BƯỚC 2: DATA INTEGRATION & CLEANING ---
    print("\n--- STARTING STEP 2: PROCESSING ---")

    # 2.1 Clean Date Types
    orders, holidays = data_processing.clean_and_convert_types(orders, holidays)

    # 2.2 Merge Data (Tạo bảng Master)
    master_df = data_processing.merge_data(orders, items, customers, products)

    # 2.3 Handle Missing Data (Lọc dữ liệu trước khi tính toán)
    master_df = data_processing.handle_missing_data(master_df)

    # 2.4 Feature Engineering (Tạo cột Logistics)
    master_df = data_processing.create_logistics_features(master_df, holidays)

    # 2.5 Optimization (Xóa cột thừa)
    master_df = data_processing.drop_unnecessary_columns(master_df)

    # Kiểm tra kết quả
    print("\n--- FINAL DATA PREVIEW ---")
    print(master_df[['order_id', 'lead_time_days', 'is_late', 'holidays_in_transit']].head())

    # Lưu file Master cuối cùng ra CSV để dùng cho Dashboard
    final_output_path = os.path.join(OUTPUT_DIR, 'Master_Logistics_Data.csv')
    master_df.to_csv(final_output_path, index=False)
    print(f"\n✅ Đã lưu file Master Data tại: {final_output_path}")
    print("Sẵn sàng cho Bước 3 (Visual/Model)!")


# # # ----------------------------------------------------------------------------------------------------------# # #
    # --- BƯỚC 3: ADVANCED ANALYTICS & KPI ---
    print("\n--- STARTING STEP 3: ANALYTICS ---")

    # 3.1 Tính toán KPI tổng hợp (Aggregation)
    kpi_state, kpi_month = data_analytics.aggregate_kpis_for_dashboard(master_df)

    # Lưu KPI ra CSV riêng (File này rất nhẹ, dùng cho Tableau/PowerBI cực nhanh)
    kpi_state.to_csv(os.path.join(OUTPUT_DIR, 'KPI_by_State.csv'), index=False)
    kpi_month.to_csv(os.path.join(OUTPUT_DIR, 'KPI_by_Month.csv'), index=False)

    # 3.2 Dự báo nhu cầu (Forecasting)
    forecast_df = data_analytics.forecast_orders(master_df, periods=12) # Dự báo 12 tuần tới
    forecast_df.to_csv(os.path.join(OUTPUT_DIR, 'Forecast_Results.csv'), index=False)

    print(f"\n✅ Đã tạo các file phân tích tại thư mục {OUTPUT_DIR}:")
    print("   1. KPI_by_State.csv (Dùng vẽ bản đồ)")
    print("   2. KPI_by_Month.csv (Dùng vẽ biểu đồ xu hướng)")
    print("   3. Forecast_Results.csv (Dùng vẽ đường dự báo tương lai)")

    print("\n=== HOÀN TẤT TOÀN BỘ QUY TRÌNH PYTHON ===")
    print("Bây giờ bạn hãy dùng các file trong thư mục Outputs để vẽ Dashboard.")


# # # ----------------------------------------------------------------------------------------------------------# # #
    # --- BƯỚC 4: VISUALIZATION ---
    print("\n--- STARTING STEP 4: VISUALIZATION ---")

    # Tạo thư mục chứa ảnh biểu đồ
    CHARTS_DIR = os.path.join(OUTPUT_DIR, 'Charts')
    os.makedirs(CHARTS_DIR, exist_ok=True)

    # 4.1 Vẽ Trend
    visualization.plot_monthly_trend(
        os.path.join(OUTPUT_DIR, 'KPI_by_Month.csv'),
        CHARTS_DIR
    )

    # 4.2 Vẽ State Performance
    visualization.plot_state_performance(
        os.path.join(OUTPUT_DIR, 'KPI_by_State.csv'),
        CHARTS_DIR
    )

    # 4.3 Vẽ Forecast
    visualization.plot_forecast(
        os.path.join(OUTPUT_DIR, 'Forecast_Results.csv'),
        CHARTS_DIR
    )

    print("\n=== HOÀN TẤT TOÀN BỘ DỰ ÁN ===")
    print(f"1. Dữ liệu sạch nằm trong folder: {OUTPUT_DIR}")
    print(f"2. Biểu đồ báo cáo nằm trong folder: {CHARTS_DIR}")


if __name__ == "__main__":
    main()
