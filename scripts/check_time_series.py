# scripts/check_time_series.py
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings

# Add parent directory to path to enable importing from src when run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

P_VALUE_THRESHOLD = 0.05

def prepare_timeseries_data(master_data_path):
    """
    Load weekly order timeseries data from Master Logistics dataset.
    """
    print(">>> [1] Preparing weekly order time series...")
    df = pd.read_csv(master_data_path, parse_dates=['order_purchase_timestamp'])
    df_ts = df.set_index('order_purchase_timestamp')
    weekly_orders = df_ts.resample('W')['order_id'].nunique().fillna(0)
    weekly_orders = weekly_orders[weekly_orders.index >= '2017-01-01']
    weekly_orders = weekly_orders.asfreq('W').fillna(0).astype(float)
    return weekly_orders

def test_stationarity(series, series_name=""):
    """
    Execute Augmented Dickey-Fuller (ADF) test.
    """
    print(f"\n--- ADF Stationarity Test for: '{series_name}' ---")
    result = adfuller(series.dropna())
    p_value = result[1]

    print(f'    ADF Statistic: {result[0]:.4f}')
    print(f'    p-value: {p_value:.10f}')

    if p_value <= P_VALUE_THRESHOLD:
        print(f'    -> Conclusion: STATIONARY (p-value <= {P_VALUE_THRESHOLD})')
    else:
        print(f'    -> Conclusion: NON-STATIONARY (p-value > {P_VALUE_THRESHOLD})')

    return p_value

def analyze_stationarity(weekly_orders):
    """
    Run full stationarity check including differencing up to d=2 and generate plot.
    """
    print("\n====== RUNNING STATIONARITY CHECKS ======")
    df_analysis = pd.DataFrame({'original': weekly_orders})
    df_analysis['diff_1'] = df_analysis['original'].diff()
    df_analysis['diff_2'] = df_analysis['diff_1'].diff()

    test_stationarity(df_analysis['original'], 'Original Series (d=0)')
    test_stationarity(df_analysis['diff_1'], 'First-Order Difference (d=1)')
    test_stationarity(df_analysis['diff_2'], 'Second-Order Difference (d=2)')

    # Export analysis csv
    csv_path = os.path.join(PROCESSED_DATA_DIR, 'Stationarity_Analysis.csv')
    df_analysis.to_csv(csv_path)
    print(f"\n>>> [ADF] Exported differencing analysis to: {csv_path}")

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Original
    axes[0].plot(df_analysis.index, df_analysis['original'], label='Original Series (d=0)')
    axes[0].set_title('Original Series (d=0) - Weekly Orders', fontsize=14)
    axes[0].legend()

    # Diff 1
    axes[1].plot(df_analysis.index, df_analysis['diff_1'], label='First-Order Differencing (d=1)', color='orange')
    axes[1].hlines(0, xmin=df_analysis.index.min(), xmax=df_analysis.index.max(), linestyles='--', color='gray')
    axes[1].set_title('First-Order Differencing (d=1)', fontsize=14)
    axes[1].legend()

    # Diff 2
    axes[2].plot(df_analysis.index, df_analysis['diff_2'], label='Second-Order Differencing (d=2)', color='green')
    axes[2].hlines(0, xmin=df_analysis.index.min(), xmax=df_analysis.index.max(), linestyles='--', color='gray')
    axes[2].set_title('Second-Order Differencing (d=2)', fontsize=14)
    axes[2].set_xlabel('Date')
    axes[2].legend()

    plt.tight_layout()
    chart_path = os.path.join(FIGURES_DIR, 'Chart_4_Stationarity_Analysis.png')
    plt.savefig(chart_path)
    plt.close()
    print(f">>> [ADF] Saved stationarity analysis plot at: {chart_path}")

def analyze_seasonality(series):
    """
    Compute and plot Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF).
    """
    print("\n====== RUNNING SEASONALITY CHECKS ======")
    n_lags = 26
    acf_values = acf(series, nlags=n_lags)
    pacf_values = pacf(series, nlags=n_lags)

    # Save outputs
    df_corr = pd.DataFrame({
        'Lag': np.arange(len(acf_values)),
        'ACF': acf_values,
        'PACF': pacf_values
    })
    csv_path = os.path.join(PROCESSED_DATA_DIR, 'Autocorrelation_Analysis_Results.csv')
    df_corr.to_csv(csv_path, index=False)
    print(f"\n>>> [ACF/PACF] Exported autocorrelation analysis to: {csv_path}")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # ACF
    plot_acf(series, lags=n_lags, ax=ax1, color='blue', vlines_kwargs={"colors": 'blue'})
    ax1.set_title('Autocorrelation Function (ACF) - Seasonality Proof', fontsize=14)
    ax1.set_xlabel('Lags (Weeks)')
    ax1.set_ylabel('Correlation')

    # PACF
    plot_pacf(series, lags=n_lags, ax=ax2, color='red', vlines_kwargs={"colors": 'red'})
    ax2.set_title('Partial Autocorrelation Function (PACF) - Autoregressive Order Identification', fontsize=14)
    ax2.set_xlabel('Lags (Weeks)')
    ax2.set_ylabel('Correlation Coefficient')

    plt.tight_layout()
    chart_path = os.path.join(FIGURES_DIR, 'Chart_7_Autocorrelation_Proof.png')
    plt.savefig(chart_path)
    plt.close()
    print(f">>> [ACF/PACF] Saved seasonality analysis plot at: {chart_path}")

def main():
    print("====================================================")
    print("STARTING TIME SERIES STATIONARITY & SEASONALITY CHECKS")
    print("====================================================")
    
    master_data_path = os.path.join(PROCESSED_DATA_DIR, 'Master_Logistics_Data.csv')
    if not os.path.exists(master_data_path):
        print(f"LỖI: File không tồn tại {master_data_path}. Hãy chạy `main.py` trước.")
        return

    # Prepare timeseries
    weekly_orders = prepare_timeseries_data(master_data_path)

    # ADF Test
    analyze_stationarity(weekly_orders)

    # ACF/PACF Check
    analyze_seasonality(weekly_orders)
    
    print("\n====================================================")
    print("TIME SERIES ANALYSIS COMPLETE")
    print("====================================================")

if __name__ == "__main__":
    main()
