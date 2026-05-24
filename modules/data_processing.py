# modules/data_processing.py
import pandas as pd
import numpy as np


#--------------------------------------------------#
def clean_and_convert_types(df_orders, df_holidays):
    """
    Chuẩn hóa định dạng thời gian cho các bảng.
    """
    print(">>> [Processing] Đang chuẩn hóa định dạng thời gian...")

    # 1. Chuyển đổi các cột ngày tháng trong Orders
    date_cols = [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]

    for col in date_cols:
        # errors='coerce': Nếu lỗi thì biến thành NaT (Not a Time) chứ không dừng chương trình
        df_orders[col] = pd.to_datetime(df_orders[col], errors='coerce')

    # 2. Chuẩn hóa ngày lễ (đảm bảo là datetime)
    if 'date' in df_holidays.columns:
        df_holidays['date'] = pd.to_datetime(df_holidays['date'])

    return df_orders, df_holidays


#--------------------------------------------------#
def merge_data(orders, items, customers, products):
    """
    Hợp nhất (Join/Merge) các nguồn dữ liệu rời rạc thành một bảng Master.
    """
    print(">>> [Processing] Đang thực hiện Data Fusion (Merge các bảng)...")

    # Bước 1: Merge Orders với Items (1 đơn có nhiều hàng -> Dữ liệu sẽ nở ra)
    # Dùng Inner Join vì chúng ta chỉ quan tâm đơn có hàng hóa
    df_merged = orders.merge(items, on='order_id', how='inner')

    # Bước 2: Merge với Customers (từ nguồn JSON)
    # Lưu ý: Orders dùng customer_id để nối
    df_merged = df_merged.merge(customers, on='customer_id', how='left')

    # Bước 3: Merge với Products (từ nguồn XML)
    # Lưu ý: Ở Bước 1 ta đã xóa dấu '_' trong XML (product_id -> productid), giờ phải mapping đúng
    # Rename lại cột của bảng products cho khớp để dễ merge
    products = products.rename(columns={'productid': 'product_id', 'productcategoryname': 'product_category_name'})

    df_merged = df_merged.merge(products[['product_id', 'product_category_name']], on='product_id', how='left')

    print(f"    -> Kích thước sau khi merge: {df_merged.shape}")
    return df_merged


#--------------------------------------------------#
def handle_missing_data(df):
    """
    Xử lý các giá trị bị thiếu (Missing Values) theo logic nghiệp vụ Logistics.
    """
    print(">>> [Cleaning] Đang xử lý dữ liệu thiếu (Missing Handling)...")

    initial_rows = len(df)

    # 1. Xử lý trạng thái đơn hàng (Order Status)
    # Vì bài toán là 'Logistics Performance' (Phân tích hiệu quả giao vận),
    # chúng ta chỉ quan tâm đến các đơn ĐÃ GIAO (delivered).
    # Các đơn 'canceled', 'invoiced'... không có ngày giao hàng -> Lead Time = NaN -> Không phân tích được.

    print(f"    - Tổng số dòng ban đầu: {initial_rows}")
    print(f"    - Các trạng thái đơn hàng hiện có: {df['order_status'].unique()}")

    # Lọc: Chỉ lấy đơn 'delivered'
    df_clean = df[df['order_status'] == 'delivered'].copy()

    # 2. Xóa các dòng mà dù status='delivered' nhưng vẫn thiếu ngày giao (Dữ liệu lỗi hệ thống)
    # subset: chỉ kiểm tra thiếu dữ liệu ở cột quan trọng
    df_clean = df_clean.dropna(subset=['order_delivered_customer_date', 'order_purchase_timestamp'])

    # 3. Điền dữ liệu thiếu cho các cột ít quan trọng hơn (Imputation)
    # Ví dụ: Nếu thiếu tên sản phẩm, điền là 'Unknown' thay vì xóa
    if 'product_category_name' in df_clean.columns:
        df_clean['product_category_name'] = df_clean['product_category_name'].fillna('Unknown')

    cleaned_rows = len(df_clean)
    print(f"    -> Đã loại bỏ {initial_rows - cleaned_rows} dòng (Đơn hủy/chưa giao/lỗi).")
    print(f"    -> Dữ liệu sạch còn lại: {cleaned_rows} dòng.")

    return df_clean


#--------------------------------------------------#
def create_logistics_features(df, df_holidays):
    """
    Tạo các cột tính toán chuyên sâu cho Logistics (Feature Engineering).
    """
    print(">>> [Processing] Đang tạo các chỉ số Logistics (Feature Engineering)...")

    # 1. Actual Lead Time (Ngày): Khách nhận - Khách đặt
    df['lead_time_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

    # 2. Estimated Lead Time (Ngày): Dự kiến - Khách đặt
    df['estimated_lead_time_days'] = (df['order_estimated_delivery_date'] - df['order_purchase_timestamp']).dt.days

    # 3. Delay (Ngày): Ngày thực tế - Ngày dự kiến
    # Nếu dương (+) là trễ, âm (-) là sớm
    df['days_diff_estimated'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days

    # 4. Late Flag (Cờ báo trễ): 1 là trễ, 0 là đúng hạn
    df['is_late'] = np.where(df['days_diff_estimated'] > 0, 1, 0)

    # 5. Holiday Overlap (Logic cao cấp)
    # Kiểm tra xem trong khoảng thời gian vận chuyển có dính ngày lễ nào không?
    # Logic: Lấy danh sách ngày lễ, đếm xem có bao nhiêu ngày nằm giữa Purchase và Delivered

    print("    -> Đang tính toán ảnh hưởng của ngày lễ (có thể mất chút thời gian)...")

    # Lấy danh sách ngày lễ thành một Set để tra cứu cho nhanh
    holiday_dates = set(df_holidays['date'].dt.date)

    def count_holidays(row):
        start = row['order_purchase_timestamp']
        end = row['order_delivered_customer_date']

        if pd.isna(start) or pd.isna(end):
            return 0

        # Tạo dãy ngày từ start đến end
        daterange = pd.date_range(start=start, end=end).date
        # Đếm số ngày giao thoa với danh sách ngày lễ
        return sum(1 for day in daterange if day in holiday_dates)


    df['holidays_in_transit'] = df.apply(count_holidays, axis=1)

    return df


#--------------------------------------------------#
def drop_unnecessary_columns(df):
    """
    Loại bỏ các cột không cần thiết cho bài toán Logistics để giảm nhẹ dữ liệu.
    """
    print(">>> [Optimization] Đang loại bỏ các cột dữ liệu thừa...")

    # Danh sách các cột cần xóa (Drop list)
    cols_to_drop = [
        'order_status',                  # Vì 100% là delivered
        'product_name_lenght',           # Dữ liệu Marketing
        'product_description_lenght',    # Dữ liệu Marketing
        'product_photos_qty',            # Dữ liệu Marketing
        'order_approved_at',             # Ít quan trọng với Lead Time
        'customer_unique_id',            # Redundant
        'product_category_name'          # Giữ lại bản gốc tiếng Bồ hay bản dịch?
                                         # (Nếu bạn muốn giữ category thì xóa dòng này trong list)
    ]

    # Chỉ xóa những cột CÓ tồn tại trong df để tránh lỗi
    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]

    df_optimized = df.drop(columns=existing_cols_to_drop)

    # Xử lý thêm: Các ID (order_id, customer_id) có thể cần cho Dashboard để Drill-down
    # Nhưng nếu muốn gọn nữa thì có thể xóa, tuy nhiên tôi khuyên nên GIỮ ID.

    print(f"    -> Đã xóa {len(existing_cols_to_drop)} cột: {existing_cols_to_drop}")
    print(f"    -> Số cột còn lại: {df_optimized.shape[1]}")

    return df_optimized

#--------------------------------------------------#
def create_aggregated_features(df):
    """
    Tạo các feature tổng hợp hàng tuần từ dữ liệu giao dịch.
    Đây là bước làm giàu dữ liệu (Data Enrichment) cho file Master.
    """
    print(">>> [Enrichment] Đang tạo các features kinh doanh tổng hợp...")

    if 'order_purchase_timestamp' not in df.columns:
        print("    ! Cảnh báo: Thiếu cột 'order_purchase_timestamp'. Bỏ qua bước này.")
        return df

    # Đảm bảo index là datetime để resample
    df_temp = df.set_index('order_purchase_timestamp')

    # Tổng hợp dữ liệu hàng tuần
    weekly_features = df_temp.resample('W').agg(
        active_sellers=('seller_id', 'nunique'),
        active_customers=('customer_id', 'nunique'),
        product_variety=('product_id', 'nunique'),
        weekly_gmv=('price', 'sum'),
        avg_basket_size=('order_item_id', 'mean'),
        avg_price=('price', 'mean'),
        avg_freight_value=('freight_value', 'mean')
    ).fillna(0)

    # Đặt lại tên cột để dễ hiểu khi merge
    weekly_features.rename(columns=lambda c: f'weekly_{c}', inplace=True)

    df['order_week'] = df['order_purchase_timestamp'].dt.to_period('W')
    weekly_features.index = weekly_features.index.to_period('W')

    df_enriched = df.merge(weekly_features, left_on='order_week', right_index=True, how='left')
    df_enriched.drop(columns=['order_week'], inplace=True)

    print(f"    -> Đã thêm {len(weekly_features.columns)} cột features tổng hợp mới.")

    return df_enriched
