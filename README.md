# OLIST E-COMMERCE LOGISTICS PERFORMANCE & DEMAND FORECASTING REPORT 🇧🇷📦

---

## 1. EXECUTIVE SUMMARY

This project implements an end-to-end supply chain analytics system for the Brazilian e-commerce platform **Olist**, focusing on two core business challenges: **Last-Mile Logistics Performance** and **Weekly Order Demand Forecasting**. 

By consolidating an ETL pipeline of **110,189 clean transaction records** and performing Natural Language Processing (NLP) on over **100,000 customer reviews**, we have uncovered critical operational insights:
*   **The São Paulo (SP) Efficiency**: Serving as the logistical backbone, SP handles ~42.1% of national orders with an outstanding average lead time of **8.26 days** and a minimal delay rate of **4.40%**.
*   **The Rio de Janeiro (RJ) Logistics Anomaly**: Despite being the second-largest market (~12.8% volume), RJ represents the most severe operational bottleneck. Its delay rate reaches **11.62%** (nearly triple that of SP) with average lead times stretching to **14.69 days**, driven by local security challenges and mail distribution inefficiencies.
*   **Demand Forecasting**: An advanced multivariate LSTM neural network, incorporating exogenous business variables and peak events (e.g., Black Friday, national truck strikes), projects a stable demand outlook for the next 12 weeks, ranging between **628 and 1,275 orders/week**.
*   **Operational Quality Issues (VoC)**: Semantic clustering of 1-star reviews reveals that the primary driver of negative sentiment is **incorrect product color/attribute shipments** by third-party sellers (strongly associated with Portuguese terms `preto`/`rosa`/`azul` and action verbs `mandaram`/`pedi`/`errada` - meaning "sent wrong color requested").

**Key Strategic Recommendation**: Establish local fulfillment centers near the Rio de Janeiro metropolitan area to reduce lead times to under 10 days, and introduce mandatory Barcode Scan Validation in the Olist seller portal to eliminate packing errors.

---

## 2. BUSINESS BACKGROUND & OBJECTIVES

Olist is a major Brazilian e-commerce store-in-store integrator, connecting thousands of small businesses to the country’s largest online marketplaces. In a continental nation like Brazil, managing chặng cuối (last-mile) distribution is the single most critical factor for customer retention and operational profitability.

### Key Objectives:
1.  **Identify Logistics Bottlenecks**: Map geographic areas and time periods experiencing delays and high shipping costs.
2.  **Predict Weekly Order Volume**: Develop robust statistical and deep learning models to predict weekly transaction volumes for capacity planning.
3.  **Decode Customer Pain Points**: Analyze unstructured review comments using NLP to isolate root causes of dissatisfaction.

---

## 3. DATA PIPELINE ARCHITECTURE

The project employs a structured data integration architecture, simulating a distributed corporate data environment.

```text
[Raw CSV Orders] ───┐
[Raw JSON Customers] ├─> [ingestion.py] ──> [processing.py] ──> [Master Data CSV]
[Raw XML Products] ──┘         ^                    │
[Nager.Date API] ─────────────┘                    ├─> [analytics.py] ──> [KPIs CSV] ──> Power BI
                                                   └─> [nlp.py] ──> [Word2Vec] ──> Reports/Charts
```

*   **Ingestion (`src/ingestion.py`)**: Merges transactions (CSV), customer demographics (JSON), and product specifications (XML) to simulate enterprise data formats. It also connects to the **Nager.Date API** to fetch Brazilian national holidays from 2016 to 2018.
*   **Processing (`src/processing.py`)**:
    *   Constructs a single master table containing **110,189 delivered orders**.
    *   Computes logistics metrics: actual delivery lead time (`lead_time_days`), estimated lead time (`estimated_lead_time_days`), delivery delay (`days_diff_estimated`), and late delivery indicator flag (`is_late`).
    *   **Holidays in Transit**: Scans calendar ranges to count how many public holidays occurred during the active shipping period of each package (`holidays_in_transit`).
    *   **Weekly Business Aggregates**: Generates 7 dynamic weekly business metrics (such as active sellers, GMV, and average basket sizes) to enrich the dataset for multivariate deep learning models.

---

## 4. LOGISTICS OPERATIONAL PERFORMANCE ANALYSIS

### 4.1. Monthly Logistics Trend

From early 2017 to late 2018, Olist experienced rapid volume growth, scaling from **2,000 orders/month** to a peak of **7,500 orders/month** in November 2017 (Black Friday). 

Despite this steep surge in transactions, average delivery lead times remained well-controlled, fluctuating between **10 and 15 days**. This reflects the scalability of Olist's middle-mile and last-mile carrier partnerships.

![Monthly Trend](reports/figures/Chart_1_Monthly_Trend.png)

### 4.2. Geographic Performance & The Rio de Janeiro Bottleneck

Logistical efficiency varies significantly across Brazil’s states:
*   **São Paulo (SP)**: Dominates the platform with **~42.1% market share** (46,441 orders). SP operates as a benchmark: cheapest average freight (**15.11 BRL**), fastest deliveries (**8.26 days**), and the lowest late rate (**4.40%**).
*   **Rio de Janeiro (RJ)**: Represents a severe operational bottleneck. Despite generating 12.8% of platform volume, its delay rate is **11.62%** (nearly triple SP) and lead times stretch to **14.69 days**. Because RJ is geographically adjacent to SP, this delay points to local carrier congestion, sorting center delays, and cargo theft security risks.
*   **Remote Northern Regions (AL, RR, AM)**: Alagoas (AL) registers the highest delay rate (**20.84%**) and lead times up to **23.99 days**. Shipping costs to these remote regions average **40-43 BRL** (nearly triple SP’s rates).

![Late Delivery Rate by State](reports/figures/Chart_2_State_Late_Rate.png)

---

## 5. TIME-SERIES ANALYSIS & DEMAND FORECASTING

Weekly order data (88 observations beginning Jan 01, 2017) was subjected to rigorous statistical analysis before training forecasting models.

### 5.1. Stationarity Checking (ADF Test)

Time-series forecasting models require stationary inputs. The **Augmented Dickey-Fuller (ADF)** test showed:
*   **Original Series ($d=0$)**: $p\text{-value} = 0.2688 > 0.05$ -> **Non-stationary** due to strong upward growth.
*   **First-Order Differencing ($d=1$)**: $p\text{-value} = 3.2 \times 10^{-7} \le 0.05$ -> **Stationary**.
Consequently, downstream forecasting models are trained on first-order differenced data to avoid spurious regressions.

![ADF Test](reports/figures/Chart_4_Stationarity_Analysis.png)

### 5.2. Seasonality Proof (ACF/PACF)

*   The **PACF** (Partial Autocorrelation Function) spikes at Lag 1 (**0.807**), showing a strong autoregressive (AR) component where the current week's volume depends heavily on the previous week.
*   The **ACF** (Autocorrelation Function) exhibits local peaks at **Lags 4-5** (~1 month) and **Lag 13** (~1 quarter/13 weeks), mathematically proving the presence of monthly and quarterly seasonality.

![ACF and PACF Plots](reports/figures/Chart_7_Autocorrelation_Proof.png)

### 5.3. Forecasting Model Comparison

We performed an out-of-sample benchmarking test by splitting the historical data into a **75-week training set** and a **12-week test set**. Seven models were trained on the historical sequence and evaluated on their out-of-sample accuracy:

| Model | MAE | RMSE | MAPE (%) | Evaluation & Status |
| :--- | :---: | :---: | :---: | :--- |
| **Holt-Winters** | **367.26** | **423.74** | **22.75%** | 🏆 **Best Model (Selected for Production)** |
| Naive (Baseline) | 443.42 | 527.37 | 26.79% | Baseline reference |
| SARIMA | 364.75 | 448.59 | 29.20% | Classical statistical model, good seasonal capture |
| XGBoost | 511.66 | 603.42 | 30.98% | Advanced ML (lag features), struggles with noise |
| Random Forest Regressor | 515.94 | 600.12 | 31.40% | ML ensemble with lags |
| LightGBM | 544.52 | 636.95 | 33.37% | Light Gradient Boosting, overfits on small dataset |
| LSTM (Univariate) | *N/A* | *N/A* | *N/A* | Skipped (Tensorflow Python 3.14 environment constraint) |

*Results*: The Holt-Winters Exponential Smoothing model achieved the lowest MAPE of **22.75%**, demonstrating superior capability in capturing Olist's weekly e-commerce seasonality and short-term trends. Classical statistical models like Holt-Winters often outperform machine learning models (XGBoost, LightGBM) on smaller datasets (<100 observations) due to their sample efficiency and robust seasonal modeling. Holt-Winters is selected as the primary forecasting engine.

![Model Comparison](reports/figures/Chart_5_Forecast_Comparison.png)

### 5.4. Advanced Multivariate LSTM Forecasting

The advanced multivariate LSTM model (`scripts/run_forecasting.py`) incorporates 8 features (GMV, active sellers, price, freight, etc.) along with a `black_friday_peak` dummy variable to prevent outlier-induced weight distortion.

The model projects a stable demand outlook for the next 12 weeks, with order volumes leveling out between **628 and 1,275 orders/week** following the rapid growth phase of 2017.

![Advanced LSTM Forecast](reports/figures/Chart_6_LSTM_Forecast.png)

---

## 6. CUSTOMER SENTIMENT ANALYSIS (NLP MODEL)

We applied NLP to **100,000+ review comments** to extract qualitative drivers of customer satisfaction.

### 6.1. NLP Methodology
*   **Text Preprocessing**: Cleaned special characters, converted to lowercase, and removed Portuguese stopwords using NLTK.
*   **TF-IDF Extraction**: Extracted unigrams and bigrams from low-rated (1 star) vs. high-rated (5 stars) reviews.
*   **Word2Vec Embeddings**: Trained a 150-dimension word vector model to find terms with high Cosine Similarity to target themes.

### 6.2. Drivers of Customer Satisfaction (Positive Feedback)
Positive reviews (`positive_delivery` and `positive_product` seeds) are dominated by terms like: `good` (tốt), `excellent` (xuất sắc), `quality` (chất lượng), and `before` / `fast` (giao nhanh trước hạn). Meeting or beating estimated delivery dates and product condition are the primary drivers of 5-star ratings.

![Positive Feedback WordCloud](reports/figures/wordcloud_embedding_AGGREGATED_positive.png)

### 6.3. Drivers of Customer Dissatisfaction (Negative Feedback)
For negative reviews, the model identified Portuguese color terms: `preto`/`preta` (black), `rosa` (pink), `azul` (blue), `vermelho` (red), and `branco` (white) clustering alongside the words `mandaram` (they sent), `pedi` (I ordered), and `errada` (wrong).

*   **Key Finding (VoC)**: A major operational issue is that **sellers frequently ship products with incorrect color attributes** (particularly cables, chargers, and small replacement parts).
*   Additionally, terms like `delay`, `not arrived`, and `quebrado` (broken/damaged) highlight shipping delays and inadequate protective packaging for long-distance transit.

![Negative Feedback WordCloud](reports/figures/wordcloud_embedding_AGGREGATED_negative.png)

---

## 7. INTERACTIVE DASHBOARD SYSTEM

An interactive Power BI dashboard reporting system was developed to provide decision-makers with real-time operational metrics.

### 7.1. Operations Performance Dashboard
Monitors average lead times, delivery delays, and shipping costs, paired with a geographic heatmap to quickly identify bottlenecked states.

![Operations Dashboard](reports/dashboards/screenshots/operation_performence.png)

### 7.2. Customer Behavior Dashboard
Analyzes transaction volumes, regional buyer distribution, peak purchasing hours, top GMV categories, and review score distributions to track customer satisfaction.

![Customer Behaviour Dashboard](reports/dashboards/screenshots/customer_behaviour.png)

### 7.3. Strategic Growth Forecast Dashboard
Visualizes the 12-week forward demand projection, assisting logistics managers with warehouse staffing and carrier allocation.

![Strategic Growth Dashboard](reports/dashboards/screenshots/strategic_growth_forecast.png)

---

## 8. STRATEGIC RECOMMENDATIONS FOR OLIST EXECUTIVE BOARD

Based on logistics data, demand forecasts, and customer review NLP, we propose three strategic interventions:

### 1. Mitigate Last-Mile Congestion in Rio de Janeiro (RJ)
*   **Action**: Partner with regional carriers to establish a dedicated **Fulfillment Center (satellite warehouse)** near Rio de Janeiro for top-selling SKUs.
*   **Impact**: Reduce average RJ lead times from **14.69 days to under 10 days** and cut delay rates from **11.62% to under 5%** by bypassing interstate sorting centers in SP.

### 2. Implement Seller Quality Control & Packaging Standards
*   **Action**: Integrate mandatory **Barcode Scan Verification** into the Olist seller portal. Sellers must scan both the product barcode and the order sheet to confirm color and size attributes before generating a shipping label.
*   **Enforcement**: Penalize or lower search rankings for sellers who repeatedly trigger negative NLP flags for "wrong attributes shipped" or "damaged packaging" (`quebrado`).

### 3. Implement Demand-Driven Logistics Planning
*   **Action**: Use the **multivariate LSTM forecasting model** to align fulfillment warehouse staffing and carrier capacity with projected weekly order volumes.
*   **Real-time Alerts**: Link external events (e.g., transit strikes, extreme weather) to the checkout engine to dynamically adjust the estimated delivery dates shown to customers, preventing customer dissatisfaction when delays are unavoidable.
