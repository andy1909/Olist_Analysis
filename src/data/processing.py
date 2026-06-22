# src/processing.py
import pandas as pd
import numpy as np

def clean_and_convert_types(df_orders, df_holidays):
    """
    Standardize datetime formats for the dataframes.
    """
    print(">>> [Processing] Converting datetime fields...")

    date_cols = [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]

    for col in date_cols:
        df_orders[col] = pd.to_datetime(df_orders[col], errors='coerce')

    if 'date' in df_holidays.columns:
        df_holidays['date'] = pd.to_datetime(df_holidays['date'])

    return df_orders, df_holidays

def merge_data(orders, items, customers, products):
    """
    Merge separate data sources into a master data frame (Data Fusion).
    """
    print(">>> [Processing] Executing Data Fusion (Merging datasets)...")

    # Merge Orders with Items (1-to-many join)
    df_merged = orders.merge(items, on='order_id', how='inner')

    # Merge with Customers
    df_merged = df_merged.merge(customers, on='customer_id', how='left')

    # Rename and merge with Products
    products = products.rename(columns={'productid': 'product_id', 'productcategoryname': 'product_category_name'})
    df_merged = df_merged.merge(products[['product_id', 'product_category_name']], on='product_id', how='left')

    print(f"    - Merged shape: {df_merged.shape}")
    return df_merged

def handle_missing_data(df):
    """
    Handle missing values based on supply chain logic (Logistics-focused filtering).
    """
    print(">>> [Processing] Handling missing data...")
    initial_rows = len(df)

    # Focus on delivered orders for lead time and logistics analysis
    df_clean = df[df['order_status'] == 'delivered'].copy()

    # Drop orders missing essential timestamps
    df_clean = df_clean.dropna(subset=['order_delivered_customer_date', 'order_purchase_timestamp'])

    # Impute product category name if missing
    if 'product_category_name' in df_clean.columns:
        df_clean['product_category_name'] = df_clean['product_category_name'].fillna('Unknown')

    cleaned_rows = len(df_clean)
    print(f"    - Filtered out {initial_rows - cleaned_rows} invalid/non-delivered orders.")
    print(f"    - Cleaned dataset rows: {cleaned_rows}")

    return df_clean

def create_logistics_features(df, df_holidays):
    """
    Feature engineering: Calculate specific supply chain KPIs and holiday overlap.
    """
    print(">>> [Processing] Engineering logistics metrics...")

    # 1. Actual Lead Time (Days): Delivered date - Purchase date
    df['lead_time_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

    # 2. Estimated Lead Time (Days)
    df['estimated_lead_time_days'] = (df['order_estimated_delivery_date'] - df['order_purchase_timestamp']).dt.days

    # 3. Delivery Delay (Days): positive means late, negative means early
    df['days_diff_estimated'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days

    # 4. Late flag indicator (1 = Late, 0 = On Time)
    df['is_late'] = np.where(df['days_diff_estimated'] > 0, 1, 0)

    # 5. Holiday Overlap (Count of holidays during transit)
    print("    - Calculating holidays in transit...")
    holiday_dates = set(df_holidays['date'].dt.date)

    def count_holidays(row):
        start = row['order_purchase_timestamp']
        end = row['order_delivered_customer_date']
        if pd.isna(start) or pd.isna(end):
            return 0
        daterange = pd.date_range(start=start, end=end).date
        return sum(1 for day in daterange if day in holiday_dates)

    df['holidays_in_transit'] = df.apply(count_holidays, axis=1)

    return df

def drop_unnecessary_columns(df):
    """
    Remove columns that are redundant or not relevant to the logistics pipeline.
    """
    print(">>> [Processing] Dropping unnecessary columns for optimization...")

    cols_to_drop = [
        'order_status',                  # 100% delivered
        'product_name_lenght',           # Marketing attributes
        'product_description_lenght',    # Marketing attributes
        'product_photos_qty',            # Marketing attributes
        'order_approved_at',             # Low impact on delivery performance
        'customer_unique_id',            # Redundant unique identifier
        'product_category_name'          # Categorical metadata
    ]

    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df_optimized = df.drop(columns=existing_cols_to_drop)

    print(f"    - Dropped {len(existing_cols_to_drop)} columns: {existing_cols_to_drop}")
    print(f"    - Remaining columns count: {df_optimized.shape[1]}")

    return df_optimized

def create_aggregated_features(df):
    """
    Create weekly aggregated business features from transactions.
    Used for multivariate time-series forecasting (LSTM models).
    """
    print(">>> [Processing] Generating weekly business aggregates...")

    if 'order_purchase_timestamp' not in df.columns:
        print("    ! Warning: 'order_purchase_timestamp' column missing. Skipping weekly aggregation.")
        return df

    # Prepare temp frame with timestamp index
    df_temp = df.set_index('order_purchase_timestamp')

    # Weekly aggregations
    weekly_features = df_temp.resample('W').agg(
        active_sellers=('seller_id', 'nunique'),
        active_customers=('customer_id', 'nunique'),
        product_variety=('product_id', 'nunique'),
        weekly_gmv=('price', 'sum'),
        avg_basket_size=('order_item_id', 'mean'),
        avg_price=('price', 'mean'),
        avg_freight_value=('freight_value', 'mean')
    ).fillna(0)

    # Prefix columns
    weekly_features.rename(columns=lambda c: f'weekly_{c}', inplace=True)

    df['order_week'] = df['order_purchase_timestamp'].dt.to_period('W')
    weekly_features.index = weekly_features.index.to_period('W')

    df_enriched = df.merge(weekly_features, left_on='order_week', right_index=True, how='left')
    df_enriched.drop(columns=['order_week'], inplace=True)

    print(f"    - Added {len(weekly_features.columns)} aggregated business metrics.")

    return df_enriched
