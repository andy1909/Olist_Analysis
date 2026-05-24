# main_timeseries.py
import os
import pandas as pd
from modules import data_ingestion, data_processing # Không cần data_analytics và visualization ở đây nữa

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'RawData')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("=== STARTING SIMPLIFIED DATA PIPELINE ===")

    # --- BƯỚC 1: DATA INGESTION ---
    # 1.1 Lấy dữ liệu API và lưu ra file JSON
    # Chúng ta truyền OUTPUT_DIR vào để hàm biết nơi lưu file
    data_ingestion.fetch_and_save_holidays(output_dir=OUTPUT_DIR)

    # 1.2 Nạp TẤT CẢ dữ liệu thô từ các file CSV
    all_data = data_ingestion.load_all_raw_data(DATA_DIR)

    # Gán các dataframe vào các biến quen thuộc
    orders = all_data['orders']
    customers = all_data['customers']
    products = all_data['products']
    items = all_data['order_items']

    # 1.3 Nạp dữ liệu ngày lễ từ file JSON vừa tạo
    print("\n>>> [3] Đang nạp dữ liệu ngày lễ từ file JSON...")
    holidays_path = os.path.join(OUTPUT_DIR, 'source_holidays.json')
    if os.path.exists(holidays_path):
        holidays = pd.read_json(holidays_path)
        # Chỉ lấy cột quan trọng và chuyển đổi kiểu
        if not holidays.empty:
            holidays = holidays[['date', 'localName', 'name']]
            holidays['date'] = pd.to_datetime(holidays['date'])
        print(f"    - Đã nạp {len(holidays)} ngày lễ từ JSON.")
    else:
        print("    ! Cảnh báo: Không tìm thấy file holidays JSON.")
        holidays = pd.DataFrame()

    print("\n=== BƯỚC 1 HOÀN TẤT ===")

    # --- BƯỚC 2: DATA PROCESSING & MASTER FILE CREATION ---
    print("\n--- STARTING STEP 2: PROCESSING ---")

    # 2.1 Chuẩn hóa kiểu dữ liệu thời gian
    orders, holidays = data_processing.clean_and_convert_types(orders, holidays)

    # 2.2 Merge Data
    master_df = data_processing.merge_data(orders, items, customers, products)

    # 2.3 Xử lý dữ liệu thiếu
    master_df = data_processing.handle_missing_data(master_df)

    # 2.4 Feature Engineering (Logistics)
    master_df = data_processing.create_logistics_features(master_df, holidays)

    # 2.5 Feature Engineering (Aggregated)
    master_df = data_processing.create_aggregated_features(master_df)

    # 2.6 Optimization
    master_df = data_processing.drop_unnecessary_columns(master_df)

    # 2.7 Lọc bỏ tháng cuối cùng
    # (Giữ nguyên logic lọc tháng cuối cùng của bạn ở đây)
    print(">>> [Final Clean] Đang lọc bỏ dữ liệu của tháng cuối cùng...")
    if not master_df.empty and 'order_purchase_timestamp' in master_df.columns:
        last_date = master_df['order_purchase_timestamp'].max()
        last_year = last_date.year
        last_month = last_date.month
        condition = ~((master_df['order_purchase_timestamp'].dt.year == last_year) &
                      (master_df['order_purchase_timestamp'].dt.month == last_month))
        master_df = master_df[condition].copy()
        print(f"    -> Dữ liệu mới kết thúc vào ngày: {master_df['order_purchase_timestamp'].max().date()}")

    # 2.8 Lưu file Master cuối cùng
    final_output_path = os.path.join(OUTPUT_DIR, 'Master_Logistics_Data.csv')
    master_df.to_csv(final_output_path, index=False)

    print(f"\n✅ ĐÃ TẠO XONG FILE MASTER TẠI: {final_output_path}")
    print("Pipeline xử lý dữ liệu hoàn tất. Các bước phân tích sau (LSTM, SARIMAX...) sẽ sử dụng file này.")

if __name__ == "__main__":
    main()
