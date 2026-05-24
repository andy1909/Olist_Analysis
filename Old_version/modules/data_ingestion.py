# modules/data_ingestion.py
import pandas as pd
import requests
import os
import json

def fetch_and_save_holidays(years=[2016, 2017, 2018], output_dir=None):
    """
    Gọi API lấy ngày lễ Brazil, trả về DataFrame và LƯU ra file JSON.
    """
    print(f">>> [1] Đang gọi API lấy ngày lễ Brazil cho năm {years}...")
    all_holidays = []

    for year in years:
        url = f"https://date.nager.at/api/v3/publicholidays/{year}/BR"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                all_holidays.extend(response.json())
                print(f"    - Đã lấy xong năm {year}")
        except Exception as e:
            print(f"    ! Lỗi kết nối năm {year}: {e}")

    # --- THAY ĐỔI LOGIC: LƯU RA FILE JSON ---
    if output_dir and all_holidays:
        json_path = os.path.join(output_dir, 'source_holidays.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_holidays, f, ensure_ascii=False, indent=2)
        print(f"    -> Đã lưu kết quả API vào file JSON: {json_path}")
    # ----------------------------------------

    df_holidays = pd.DataFrame(all_holidays)
    if not df_holidays.empty:
        df_holidays = df_holidays[['date', 'localName', 'name']]
        df_holidays['date'] = pd.to_datetime(df_holidays['date'])

    return df_holidays

def load_all_raw_data(data_dir):
    """
    Đọc TẤT CẢ dữ liệu gốc từ định dạng CSV.
    """
    print(">>> [2] Đang nạp tất cả dữ liệu gốc (CSV) vào Pandas...")

    # Tạo một dictionary để chứa tất cả các DataFrame
    dataframes = {}
    files_to_load = [
        'olist_orders_dataset.csv',
        'olist_customers_dataset.csv',
        'olist_products_dataset.csv',
        'olist_order_items_dataset.csv'
        # Thêm các file khác nếu cần
    ]

    for file in files_to_load:
        path = os.path.join(data_dir, file)
        # Lấy tên file không có phần mở rộng để làm key
        df_name = file.split('.')[0].replace('olist_', '').replace('_dataset', '')

        if os.path.exists(path):
            dataframes[df_name] = pd.read_csv(path)
            print(f"    - Đã nạp: {file} -> df_name: '{df_name}'")
        else:
            print(f"    ! Cảnh báo: Không tìm thấy file {path}")
            dataframes[df_name] = pd.DataFrame() # Trả về df rỗng nếu không có file

    return dataframes
