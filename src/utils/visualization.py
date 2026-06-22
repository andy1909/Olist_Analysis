# src/visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def plot_monthly_trend(kpi_month_path, output_dir):
    """
    Plot monthly orders count (bar) vs average lead time (line).
    """
    print(">>> [Visualization] Plotting monthly logistics trend...")
    if not os.path.exists(kpi_month_path):
        print(f"    ! Warning: KPI file not found at {kpi_month_path}")
        return

    df = pd.read_csv(kpi_month_path)
    fig, ax1 = plt.subplots()

    # Left Y axis: total orders
    sns.barplot(data=df, x='order_month', y='total_orders', color='skyblue', alpha=0.6, ax=ax1)
    ax1.set_ylabel('Total Orders', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

    # Right Y axis: avg lead time
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x='order_month', y='avg_lead_time', color='red', marker='o', ax=ax2, linewidth=2)
    ax2.set_ylabel('Avg Lead Time (Days)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Logistics Trend: Orders Volume vs. Delivery Speed', fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'Chart_1_Monthly_Trend.png')
    plt.savefig(save_path)
    plt.close()
    print(f"    - Saved: {save_path}")

def plot_state_performance(kpi_state_path, output_dir):
    """
    Plot a bar chart showing the late delivery rate across the top 10 states.
    """
    print(">>> [Visualization] Plotting state-level delivery performance...")
    if not os.path.exists(kpi_state_path):
        print(f"    ! Warning: KPI file not found at {kpi_state_path}")
        return

    df = pd.read_csv(kpi_state_path)
    df_top = df.nlargest(10, 'total_orders')

    plt.figure(figsize=(12, 6))
    chart = sns.barplot(
        data=df_top,
        x='customer_state',
        y='late_rate_percent',
        palette='Reds'
    )

    for p in chart.patches:
        chart.annotate(f'{p.get_height():.1f}%',
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center',
                       xytext=(0, 9),
                       textcoords='offset points')

    plt.title('Top 10 States: Late Delivery Rate (%)', fontsize=16)
    plt.ylabel('Late Rate (%)')
    plt.xlabel('State')

    save_path = os.path.join(output_dir, 'Chart_2_State_Late_Rate.png')
    plt.savefig(save_path)
    plt.close()
    print(f"    - Saved: {save_path}")

def plot_forecast(forecast_path, output_dir):
    """
    Plot historical data against forecasted demand values.
    """
    print(">>> [Visualization] Plotting demand forecast charts...")
    if not os.path.exists(forecast_path):
        print(f"    ! Warning: Forecast file not found at {forecast_path}")
        return

    df = pd.read_csv(forecast_path)
    df['date'] = pd.to_datetime(df['date'])

    plt.figure(figsize=(14, 7))

    sns.lineplot(data=df[df['type']=='History'], x='date', y='order_count', label='Historical Data', color='gray')
    sns.lineplot(data=df[df['type']=='Forecast'], x='date', y='order_count', label='Forecast (Next 12 Weeks)', color='green', linewidth=3, linestyle='--')

    plt.title('Demand Forecasting: Order Volume Prediction', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Weekly Orders')
    plt.legend()

    save_path = os.path.join(output_dir, 'Chart_3_Forecast.png')
    plt.savefig(save_path)
    plt.close()
    print(f"    - Saved: {save_path}")
