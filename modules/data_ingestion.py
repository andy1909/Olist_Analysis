# modules/data_ingestion.py
import pandas as pd
import requests
import os
import json

def convert_and_save_files(data_dir):
    """
    Chuyển đổi CSV sang JSON và XML để tạo đa dạng nguồn dữ liệu.
    """
    print(">>> [1] Đang chuyển đổi định dạng file (CSV -> JSON/XML)...")

    # 1. Customers -> JSON
    cust_path = os.path.join(data_dir, 'olist_customers_dataset.csv')
    json_path = os.path.join(data_dir, 'source_customers.json')

    if os.path.exists(cust_path):
        df = pd.read_csv(cust_path)
        df.to_json(json_path, orient='records', indent=2)
        print(f"    - Đã tạo: {json_path}")
    else:
        print(f"    ! Cảnh báo: Không thấy file {cust_path}")

    # 2. Products -> XML
    prod_path = os.path.join(data_dir, 'olist_products_dataset.csv')
    xml_path = os.path.join(data_dir, 'source_products.xml')

    if os.path.exists(prod_path):
        df = pd.read_csv(prod_path)
        # Sửa tên cột để tránh lỗi XML
        df.columns = [c.replace('_', '') for c in df.columns]
        df.to_xml(xml_path, index=False, root_name='Products', row_name='Item')
        print(f"    - Đã tạo: {xml_path}")
    else:
        print(f"    ! Cảnh báo: Không thấy file {prod_path}")

def fetch_holidays_api(years=[2016, 2017, 2018]):
    """
    Gọi API lấy ngày lễ Brazil.
    Trả về DataFrame.
    """
    print(f">>> [2] Đang gọi API lấy ngày lễ Brazil cho năm {years}...")
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

    df_holidays = pd.DataFrame(all_holidays)
    # Chỉ lấy cột quan trọng
    if not df_holidays.empty:
        df_holidays = df_holidays[['date', 'localName', 'name']]
        df_holidays['date'] = pd.to_datetime(df_holidays['date'])

    return df_holidays

def load_raw_data(data_dir):
    """
    Đọc dữ liệu từ 3 nguồn: CSV, JSON, XML và trả về các DataFrame.
    """
    print(">>> [3] Đang nạp dữ liệu vào Pandas...")

    # Đọc Orders (CSV)
    orders = pd.read_csv(os.path.join(data_dir, 'olist_orders_dataset.csv'))

    # Đọc Customers (JSON)
    customers = pd.read_json(os.path.join(data_dir, 'source_customers.json'))

    # Đọc Products (XML)
    products = pd.read_xml(os.path.join(data_dir, 'source_products.xml'))

    return orders, customers, products
