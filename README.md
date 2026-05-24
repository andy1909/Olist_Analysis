# Olist Logistics Performance & Demand Forecasting Project 🇧🇷📦

Dự án này triển khai một **Hệ thống xử lý dữ liệu đầu-cuối (End-to-End Data Pipeline)** toàn diện nhằm phân tích, tối ưu hóa hoạt động logistics và dự báo nhu cầu khách hàng cho nền tảng thương mại điện tử Olist (Brazil). Dự án tích hợp các công cụ xử lý dữ liệu quy mô lớn, kiểm định thống kê chuỗi thời gian chuyên sâu (Time-Series Checking), mô hình học máy dự báo mạng nơ-ron tuần hoàn (LSTM) và xử lý ngôn ngữ tự nhiên (NLP) trên bình luận của khách hàng.

---

## 1. Cấu Trúc Dự Án & Vai Trò Các Thành Phần

```text
Olist_Analysis/
├── RawData/                         # Thư mục dữ liệu thô đầu vào (.csv, .json, .xml)
├── modules/                         # Thư mục chứa các mô-đun Python cốt lõi
│   ├── data_ingestion.py            # ETL: Thu thập dữ liệu đa nguồn & gọi API ngày lễ Brazil
│   ├── data_processing.py           # ETL: Hợp nhất dữ liệu, làm sạch và tạo thuộc tính logistics
│   ├── data_analytics.py            # Aggregation & Dự báo cơ bản (Holt-Winters)
│   ├── visualization.py             # Trực quan hóa KPIs cơ bản (tháng, bang, dự báo)
│   ├── TF_IDF.py                    # NLP: Tiền xử lý văn bản và trích xuất TF-IDF
│   └── review_topic_analysis.py     # NLP: Phân tích từ khóa phản hồi khách hàng theo điểm đánh giá
├── Outputs/                         # Thư mục lưu trữ kết quả phân tích
│   ├── Charts/                      # Biểu đồ phân tích và đám mây từ khóa (Word Clouds)
│   └── outputs_csv/                 # File CSV phục vụ kết nối Power BI/Tableau
├── Dash2/                           # Metadata cấu trúc báo cáo của Microsoft Power BI
├── saved_models/                    # Thư mục chứa mô hình Word2Vec đã huấn luyện
├── main.py                          # Script chính chạy quy trình pipeline tự động hóa
├── stationarity_check.py            # Kiểm định ADF Test kiểm tra tính dừng của chuỗi thời gian
├── Seasonality_Check.py             # Tính toán ACF/PACF chứng minh tính mùa vụ theo tháng/quý
├── forcast.py                       # So sánh 3 mô hình dự báo: Naive, Holt-Winters, LSTM đơn biến
├── LSTM_forcast.py                  # Mô hình LSTM đa biến dự báo nâng cao (sử dụng Log, Diff, Outlier)
├── words_embeddings_analysis.py     # Huấn luyện mô hình Word2Vec tìm kiếm từ khóa tương đồng Cosine
├── visualization_keywords.py        # Vẽ WordCloud tổng hợp ý kiến Tích cực & Tiêu cực
├── requirements.txt                 # Các thư viện phụ thuộc của dự án
└── README.md                        # Tài liệu dự án (Tệp tin này)
```

---

## 2. Quy Trình Kỹ Thuật Dữ Liệu & ETL Pipeline (`main.py`)

Quy trình ETL điều phối trong file `main.py` tự động hóa các bước xử lý dữ liệu từ định dạng thô thành Master Data sẵn sàng cho trực quan hóa:

1.  **Thu thập dữ liệu đa nguồn (Ingestion)**:
    *   Đọc thông tin đơn hàng từ định dạng CSV (`olist_orders_dataset.csv` - **99,441 dòng**).
    *   Giả lập môi trường dữ liệu doanh nghiệp bằng cách đọc thông tin khách hàng từ JSON (`source_customers.json`) và chi tiết sản phẩm từ XML (`source_products.xml` - **32,951 dòng**).
    *   Kết nối trực tiếp tới **Nager.Date API** để lấy **42 ngày lễ quốc gia** của Brazil trong giai đoạn 2016-2018 nhằm phân tích biến động thời gian vận chuyển.
2.  **Hợp nhất & Làm sạch (Integration & Cleaning)**:
    *   Hợp nhất các bảng qua các khoá định danh (`order_id`, `customer_id`, `product_id`) thành một bảng master với kích thước **112,650 dòng**.
    *   Lọc bỏ đơn hàng hủy/chưa hoàn tất, loại bỏ các lỗi hệ thống thiếu mốc thời gian giao hàng, giữ lại **110,189 dòng Master sạch** có trạng thái `delivered`.
3.  **Tạo thuộc tính Logistics nâng cao (Feature Engineering)**:
    *   `lead_time_days`: Thời gian thực tế khách hàng nhận được hàng kể từ khi thanh toán.
    *   `days_diff_estimated`: Số ngày giao sớm (âm) hoặc trễ (dương) so với ngày dự kiến giao của hệ thống.
    *   `is_late`: Gán nhãn 1 cho các đơn hàng bị giao trễ (`days_diff_estimated > 0`) và 0 cho đơn đúng hạn.
    *   `holidays_in_transit`: Sử dụng thuật toán quét dải ngày vận chuyển thực tế đối chiếu với danh sách ngày lễ để đếm số ngày lễ rơi vào khoảng thời gian đơn hàng đang đi đường.
4.  **Tối ưu dữ liệu**: Loại bỏ các thuộc tính phi cấu trúc hoặc không liên quan đến chuỗi cung ứng (như mô tả sản phẩm, số lượng ảnh sản phẩm), xuất file master tinh gọn [Master_Logistics_Data.csv](file:///home/long/Documents/Olist_Analysis/Outputs/Master_Logistics_Data.csv).

---

## 3. Các Mô Hình Toán Học & Học Máy Được Sử Dụng

### 3.1. Phân Tích & Kiểm Định Chuỗi Thời Gian
Chuỗi thời gian được tổng hợp theo đơn vị tuần (nunique đơn hàng bắt đầu từ `2017-01-01` để loại bỏ giai đoạn đầu thiếu số liệu của năm 2016), tạo thành chuỗi gồm **88 quan sát (tuần)**.

*   **Kiểm định tính dừng (Stationarity Check - `stationarity_check.py`)**:
    *   Sử dụng kiểm định **Augmented Dickey-Fuller (ADF)**.
    *   Chuỗi dữ liệu gốc ($d=0$) có p-value = **$0.2789$** (Không dừng, có xu hướng tăng trưởng rõ rệt).
    *   Áp dụng sai phân bậc 1 ($d=1$) cho ra p-value = **$3.2 \times 10^{-7}$** ($\le 0.05$). Hệ thống kết luận chuỗi đạt tính dừng hoàn hảo ở bậc sai phân $d=1$. Do đó, các thuật toán dự báo tiếp theo được cấu hình học trên chuỗi sai phân bậc 1 để tránh lỗi xu hướng giả chia sẻ (spurious regression).
*   **Chứng minh tính mùa vụ (Seasonality Proof - `Seasonality_Check.py`)**:
    *   Tính toán đồ thị tự tương quan **ACF** và tự tương quan riêng phần **PACF** lên đến 26 độ trễ (lags).
    *    PACF đạt đỉnh tại Lag 1 là **0.807**, chứng minh tính tự hồi quy mạnh mẽ (tuần hiện tại phụ thuộc chặt chẽ vào tuần trước đó).
    *   ACF đạt các đỉnh cục bộ đáng kể tại **Lag 4-5** (~1 tháng) và **Lag 13** (~1 quý/13 tuần), chứng minh sự tồn tại của chu kỳ mùa vụ theo tháng và quý.

### 3.2. Mô Hình Dự Báo Nhu Cầu Đơn Hàng (Forecasting)
*   **Holt-Winters Exponential Smoothing**:
    *   *Cấu hình*: Additive trend kết hợp Additive seasonal, sử dụng cơ chế giảm chấn xu hướng (`damped_trend=True`), độ dài chu kỳ mùa vụ là 12 tuần.
    *   *Kết quả*: Dự báo tốt các chu kỳ tuần hoàn ngắn hạn, lượng đơn hàng tương lai dao động trong khoảng **1,131 đến 1,581 đơn/tuần**.
*   **Mạng Nơ-ron Hồi Quy Đa Biến LSTM (Long Short-Term Memory - `LSTM_forcast.py`)**:
    *   *Xử lý nhiễu ngoại lệ (Outliers)*: Tạo biến giả `black_friday_peak` nhận giá trị 1 cho tuần có lượng đơn đột biến cực đại (Black Friday 2017) và 0 cho các tuần khác để mô hình không bị lệch trọng số.
    *   *Biến đầu vào*: Tích hợp chuỗi thời gian của 8 biến số gồm: Lượng đơn hàng, số người bán hoạt động, số khách hàng hoạt động, số lượng danh mục sản phẩm, doanh số GMV tuần, kích thước giỏ hàng trung bình, giá bán trung bình, và chi phí vận chuyển trung bình hàng tuần.
    *   *Kiến trúc mô hình*: `LSTM(75, return_sequences=True)` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `LSTM(50)` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Dense(25)` $\rightarrow$ `Dense(1)`.
    *   *Huấn luyện*: Bộ tối ưu hóa Adam, hàm mất mát MSE, kết hợp cơ chế dừng sớm `EarlyStopping` (patience = 25 epochs) trên tập validation để tránh quá khớp (overfitting).
    *   *Kết quả*: Dự báo xu hướng tăng trưởng mượt mà của 12 tuần tiếp theo, số đơn hàng nằm trong khoảng **628 đến 1,275 đơn/tuần**.

### 3.3. Mô Hình Phân Tích Phản Hồi Khách Hàng (Customer Review NLP)
*   **Tiền xử lý & Trích xuất TF-IDF**: Tiền xử lý chữ thường, làm sạch regex ký tự đặc biệt, và lọc stop words tiếng Bồ Đào Nha bằng thư viện **NLTK**. Sử dụng `TfidfVectorizer` với `ngram_range=(1,3)` để thu thập các cụm từ quan trọng.
*   **Word Embeddings (Word2Vec - `words_embeddings_analysis.py`)**:
    *   Huấn luyện mô hình **Word2Vec** của thư viện Gensim trên **100k+ bình luận phản hồi khách hàng** để ánh xạ từ ngữ thành vector 150 chiều.
    *   Áp dụng độ tương đồng Cosine (Cosine Similarity) để tìm 20 từ khóa gần nhất với các nhóm từ khóa hạt nhân (seed keywords) đại diện cho: Giao hàng tích cực/tiêu cực, sản phẩm tích cực/tiêu cực.
    *   Kết quả tiếng Bồ Đào Nha được tự động dịch sang tiếng Anh qua API Google Translate để stakeholders dễ đọc hiểu.
*   **Trực quan hóa WordCloud**: Chuyển đổi điểm số tương đồng cosine thành tần suất xuất hiện để vẽ các đám mây từ khóa (Word Clouds) cho nhóm Positive và Negative phản ánh lý do hài lòng hoặc thất vọng của khách hàng.

---

## 4. Các Insights Đắt Giá Phát Hiện Từ Số Liệu Thống Kê

### 4.1. Sự Kiện & Biến Động Doanh Số (Temporal Insights)
*   **Điểm đột phá Black Friday (Tuần 26/11/2017)**: Số lượng đơn hàng tăng vọt lên **3,428 đơn/tuần** (tăng gấp gần 3 lần mức trung bình tuần), đóng góp lớn giúp doanh thu tháng 11/2017 đạt đỉnh kỷ lục **987,648 BRL**.
*   **Sự cố đình công xe tải toàn quốc (Tuần 27/05/2018)**: Số lượng đơn hàng đột ngột **sụt giảm 51.5%** xuống chỉ còn **1,095 đơn/tuần** (so với 2,104 đơn của tuần trước đó) do các tài xế xe tải bãi công lớn trên toàn Brazil từ ngày 21/05 đến 30/05/2018 gây tê liệt mạng lưới giao thông.

### 4.2. Điểm Nghẽn Logistics Theo Địa Lý (Geographical Insights)
*   **Tập trung khu vực**: Bang São Paulo (SP) chiếm thị phần áp đảo với **46,441 đơn hàng (~42.1% toàn quốc)**. Do là trung tâm kinh tế tập trung nhiều seller, SP có cước phí ship rẻ nhất (**15.11 BRL/đơn**), giao hàng nhanh nhất (**8.26 ngày**) và tỷ lệ trễ thấp chỉ **4.40%**.
*   **Nghịch lý bang Rio de Janeiro (RJ) - Cảnh báo đỏ**:
    *   RJ có lượng đơn hàng lớn thứ 2 cả nước với **14,143 đơn (~12.8%)**.
    *   Tuy nhiên, RJ lại là điểm nghẽn logistics tồi tệ nhất với tỷ lệ giao hàng trễ lên đến **11.62%** (gấp gần 3 lần São Paulo, tương ứng **1,644 đơn hàng trễ**). Lead time trung bình bị kéo dài lên tới **14.69 ngày**.
    *   *Insight*: Vì RJ nằm ngay cạnh SP, khoảng cách địa lý ngắn nhưng lead time và tỷ lệ trễ lại quá cao. Điều này phản ánh các vấn đề phi địa lý như tình trạng cướp hàng bưu chính thường xuyên xảy ra ở RJ, thủ tục kiểm tra hàng hóa nội bộ hoặc hạ tầng bưu cục địa phương yếu kém.
*   **Vùng sâu vùng xa phía Bắc**: Bang Alagoas (AL) có tỷ lệ trễ kỷ lục **20.84%** (hơn 1/5 đơn hàng bị trễ) và thời gian giao trung bình lên tới **23.99 ngày**. Các bang vùng xa khác như Acre, Amapá, Roraima cước ship đắt gấp 3 lần bình thường (khoảng **40-43 BRL**) và thời gian giao hàng xấp xỉ 1 tháng.

### 4.3. Lỗi Vận Hành Gửi Sai Thuộc Tính Sản Phẩm (NLP Review Insights)
*   Khi phân tích các từ khóa có điểm tương đồng cosine cao nhất liên kết với chủ đề sản phẩm tiêu cực (`negative_product`), hệ thống phát hiện sự xuất hiện dày đặc của các từ chỉ màu sắc tiếng Bồ Đào Nha: `preto`/`preta` (đen), `rosa` (hồng), `azul` (xanh), `vermelho`/`vermelha` (đỏ), `branco`/`branca` (trắng), `bege` (be), `colorido` (nhiều màu) đi kèm các động từ hành động `mandaram` (họ đã gửi), `pedi` (tôi đã yêu cầu), `errada` (sai).
*   *Insight*: Khách hàng Olist thường xuyên đánh giá 1 sao do **nhà bán hàng gửi sai màu sắc sản phẩm so với đơn đặt mua** (đặc biệt đối với các phụ kiện như cáp sạc - `cabo`, hộp mực máy in - `cartucho`, cuộn dây - `rolo`, linh kiện - `peça`).
*   Từ khóa `quebrado` (vỡ/hỏng) cho thấy chất lượng bọc hàng chống va đập trong quá trình vận chuyển đường dài chưa được đảm bảo tốt.

---

## 5. Đề Xuất Chiến Lược Tối Ưu Hóa Chuỗi Cung Ứng (Actionable Recommendations)

1.  **Thiết lập Kho Phân Phối Vệ Tinh (Fulfillment Centers) tại Rio de Janeiro (RJ)**:
    *   Olist nên khuyến khích các sellers ký gửi hàng hóa bán chạy tại các kho trung chuyển bảo mật cao ngay tại RJ.
    *   Hợp tác với các đơn vị vận chuyển chặng cuối chuyên biệt tại RJ để rút ngắn thời gian giao hàng trung bình xuống dưới **10 ngày** và kiểm soát tỷ lệ giao trễ dưới **5.0%**.
2.  **Kiểm soát chất lượng đóng gói của nhà bán hàng (Seller Quality Control)**:
    *   Cải tiến hệ thống quản lý đơn hàng của Olist, yêu cầu nhà bán hàng quét mã vạch (Barcode) đối chiếu thuộc tính sản phẩm (màu sắc/mẫu mã) trước khi đóng gói để giảm tỷ lệ gửi sai màu.
    *   Đưa ra các chế tài xử phạt hoặc giảm hiển thị sản phẩm đối với những seller liên tục bị khách hàng đánh giá 1 sao vì lỗi "gửi sai màu sắc".
3.  **Hoạch định tồn kho thích ứng theo mùa vụ và biến động**:
    *   Sử dụng dự báo nhu cầu 12 tuần của mô hình **LSTM đa biến** để lên kế hoạch phân phối hàng hóa và bố trí nhân viên kho trước các mùa mua sắm cao điểm.
    *   Tích hợp hệ thống cảnh báo tin tức xã hội (đình công, thiên tai) vào thuật toán tính toán thời gian giao dự kiến (Estimated Delivery Date) hiển thị trên ứng dụng khách hàng để giảm thiểu tỷ lệ thất vọng và khiếu nại trễ hạn.
