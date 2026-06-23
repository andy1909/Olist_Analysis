# src/ingestion.py
import pandas as pd
import requests
import os

def convert_and_save_files(data_dir):
    """
    Convert CSV to JSON and XML to create diverse data source formats for simulation.
    """
    print(">>> [Ingestion] Converting file formats (CSV -> JSON/XML)...")

    # 1. Customers -> JSON
    cust_path = os.path.join(data_dir, 'olist_customers_dataset.csv')
    json_path = os.path.join(data_dir, 'source_customers.json')

    if os.path.exists(cust_path):
        df = pd.read_csv(cust_path)
        df.to_json(json_path, orient='records', indent=2)
        print(f"    - Created: {json_path}")
    else:
        print(f"    ! Warning: File not found {cust_path}")

    # 2. Products -> XML
    prod_path = os.path.join(data_dir, 'olist_products_dataset.csv')
    xml_path = os.path.join(data_dir, 'source_products.xml')

    if os.path.exists(prod_path):
        df = pd.read_csv(prod_path)
        # Clean column names to prevent XML tag errors
        df.columns = [c.replace('_', '') for c in df.columns]
        df.to_xml(xml_path, index=False, root_name='Products', row_name='Item')
        print(f"    - Created: {xml_path}")
    else:
        print(f"    ! Warning: File not found {prod_path}")

def fetch_holidays_api(years=[2016, 2017, 2018]):
    """
    Fetch public holidays in Brazil from Nager.Date API.
    Returns a pandas DataFrame.
    """
    print(f">>> [Ingestion] Fetching Brazil public holidays from API for years {years}...")
    all_holidays = []

    for year in years:
        url = f"https://date.nager.at/api/v3/publicholidays/{year}/BR"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                all_holidays.extend(response.json())
                print(f"    - Retrieved holidays for year {year}")
        except Exception as e:
            print(f"    ! Connection error for year {year}: {e}")

    df_holidays = pd.DataFrame(all_holidays)
    if not df_holidays.empty:
        df_holidays = df_holidays[['date', 'localName', 'name']]
        df_holidays['date'] = pd.to_datetime(df_holidays['date'])

    return df_holidays

def load_raw_data(data_dir):
    """
    Load data from diverse sources: CSV (Orders), JSON (Customers), XML (Products).
    """
    print(">>> [Ingestion] Loading raw data into Pandas DataFrames...")

    orders = pd.read_csv(os.path.join(data_dir, 'olist_orders_dataset.csv'))
    customers = pd.read_json(os.path.join(data_dir, 'source_customers.json'))
    products = pd.read_xml(os.path.join(data_dir, 'source_products.xml'))

    return orders, customers, products
