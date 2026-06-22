# main.py
import os
import pandas as pd
from src.data import ingestion, processing
from src.models import analytics
from src.utils import visualization

# Path configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def main():
    print("====================================================")
    print("STARTING OLIST LOGISTICS ETL & BUSINESS PIPELINE")
    print("====================================================")

    # -------------------------------------------------------------------------
    # STEP 1: DATA INGESTION
    # -------------------------------------------------------------------------
    print("\n--- STEP 1: DATA INGESTION ---")
    
    # 1.1 Generate simulated JSON & XML datasets from CSV
    ingestion.convert_and_save_files(RAW_DATA_DIR)

    # 1.2 Fetch Brazil public holidays from API
    df_holidays = ingestion.fetch_holidays_api()

    # 1.3 Load raw datasets
    df_orders, df_customers, df_products = ingestion.load_raw_data(RAW_DATA_DIR)

    print(f"\nIngestion Summary:")
    print(f"  - Orders loaded (CSV): {df_orders.shape}")
    print(f"  - Customers loaded (JSON): {df_customers.shape}")
    print(f"  - Products loaded (XML): {df_products.shape}")
    print(f"  - Brazil Holidays (API): {df_holidays.shape}")

    # Save holidays CSV
    holidays_path = os.path.join(PROCESSED_DATA_DIR, 'brazil_holidays.csv')
    df_holidays.to_csv(holidays_path, index=False)
    print(f"    -> Saved holidays table to: {holidays_path}")

    # -------------------------------------------------------------------------
    # STEP 2: DATA INTEGRATION, CLEANING & FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    print("\n--- STEP 2: DATA INTEGRATION & PROCESSING ---")
    
    # 2.1 Clean date types
    df_orders, df_holidays = processing.clean_and_convert_types(df_orders, df_holidays)

    # 2.2 Load order items detail CSV
    items_path = os.path.join(RAW_DATA_DIR, 'olist_order_items_dataset.csv')
    if not os.path.exists(items_path):
        raise FileNotFoundError(f"Missing essential items dataset: {items_path}")
    df_items = pd.read_csv(items_path)
    print(f"  - Loaded order items details: {df_items.shape}")

    # 2.3 Merge datasets (Data Fusion)
    master_df = processing.merge_data(df_orders, df_items, df_customers, df_products)

    # 2.4 Handle missing data
    master_df = processing.handle_missing_data(master_df)

    # 2.5 Feature Engineering: Supply chain and transit holidays
    master_df = processing.create_logistics_features(master_df, df_holidays)

    # 2.6 Feature Engineering: Business aggregated features (weekly indicators)
    # [BUG FIX] Call create_aggregated_features so multivariate models have actual data columns!
    master_df = processing.create_aggregated_features(master_df)

    # 2.7 Drop unnecessary columns
    master_df = processing.drop_unnecessary_columns(master_df)

    # Save final Master Logistics Data CSV
    master_csv_path = os.path.join(PROCESSED_DATA_DIR, 'Master_Logistics_Data.csv')
    master_df.to_csv(master_csv_path, index=False)
    print(f"\n✅ Created clean Master dataset at: {master_csv_path}")

    # -------------------------------------------------------------------------
    # STEP 3: ADVANCED ANALYTICS & LOGISTICS KPI AGGREGATION
    # -------------------------------------------------------------------------
    print("\n--- STEP 3: LOGISTICS KPI AGGREGATION ---")
    
    # 3.1 Aggregate KPIs (State and Monthly trends)
    kpi_state, kpi_month = analytics.aggregate_kpis_for_dashboard(master_df)
    
    # Save KPI tables (lightweight, ideal for Power BI / Tableau connections)
    kpi_state_path = os.path.join(PROCESSED_DATA_DIR, 'KPI_by_State.csv')
    kpi_month_path = os.path.join(PROCESSED_DATA_DIR, 'KPI_by_Month.csv')
    kpi_state.to_csv(kpi_state_path, index=False)
    kpi_month.to_csv(kpi_month_path, index=False)
    print(f"    - Saved state KPIs to: {kpi_state_path}")
    print(f"    - Saved monthly KPIs to: {kpi_month_path}")

    # 3.2 Run Selected Hybrid Forecasting (Holt-Winters + XGBoost Residuals)
    forecast_df = analytics.forecast_orders_hybrid(master_df, periods=12)
    forecast_csv_path = os.path.join(PROCESSED_DATA_DIR, 'Forecast_Results.csv')
    forecast_df.to_csv(forecast_csv_path, index=False)
    print(f"    - Saved hybrid forecast timeline to: {forecast_csv_path}")

    # -------------------------------------------------------------------------
    # STEP 4: VISUALIZATION PLOTS GENERATION
    # -------------------------------------------------------------------------
    print("\n--- STEP 4: REPORT VISUALIZATIONS GENERATION ---")
    
    # 4.1 Plot monthly logistics trend
    visualization.plot_monthly_trend(kpi_month_path, FIGURES_DIR)

    # 4.2 Plot geographical late delivery performance
    visualization.plot_state_performance(kpi_state_path, FIGURES_DIR)

    # 4.3 Plot demand forecast chart
    visualization.plot_forecast(forecast_csv_path, FIGURES_DIR)

    print("\n====================================================")
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"  Processed tables: {PROCESSED_DATA_DIR}")
    print(f"  Generated charts: {FIGURES_DIR}")
    print("====================================================")

if __name__ == "__main__":
    main()
