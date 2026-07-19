# scripts/run_forecasting.py
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import MinMaxScaler

# Add parent directory to path to enable importing from src when run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Configuration
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Parameters
FORECAST_PERIODS = 12
N_STEPS = 8
EPOCHS = 100

# =============================================================================
# 1. UNIVARIATE MODELS & COMPARISONS
# =============================================================================
def prepare_univariate_data(master_path):
    print(">>> [Univariate] Preparing timeseries data...")
    df = pd.read_csv(master_path, parse_dates=['order_purchase_timestamp'])
    df_ts = df.set_index('order_purchase_timestamp')
    weekly_orders = df_ts.resample('W')['order_id'].nunique().fillna(0)
    weekly_orders = weekly_orders[weekly_orders.index >= '2017-01-01']
    weekly_orders = weekly_orders.asfreq('W').fillna(0).astype(float)
    
    # Slice the dataset to end on 2018-05-20 to avoid the strike and subsequent outliers
    weekly_orders = weekly_orders[weekly_orders.index <= '2018-05-20']
            
    return weekly_orders

def naive_forecast(series, periods):
    print(">>> [Univariate] Running Naive Forecast (Baseline)...")
    last_value = series.iloc[-1]
    return np.full(periods, last_value)

def holt_winters_forecast(series, periods):
    print(">>> [Univariate] Running Holt-Winters Forecast (Optimized)...")
    try:
        model = ExponentialSmoothing(
            series, seasonal_periods=13, trend='add', seasonal='add',
            damped_trend=False, initialization_method='estimated'
        ).fit(optimized=True)
        return model.forecast(periods)
    except Exception as e:
        print(f"    ! Holt-Winters failed: {e}. Falling back to series mean.")
        return np.full(periods, series.mean())

def univariate_lstm_forecast(series, periods, n_steps=10, epochs=50):
    print(">>> [Univariate] Running Univariate LSTM Forecast...")
    
    # 1. Stationarity differencing
    adf_result = adfuller(series)
    if adf_result[1] > 0.05:
        last_original_value = series.iloc[-1]
        series_diff = series.diff().dropna()
        is_differenced = True
    else:
        series_diff = series
        is_differenced = False

    # 2. Scaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(series_diff.values.reshape(-1, 1))

    # 3. Generator
    generator = TimeseriesGenerator(scaled_data, scaled_data, length=n_steps, batch_size=1)

    # 4. Model
    model = Sequential([
        LSTM(15, activation='relu', input_shape=(n_steps, 1)),
        Dropout(0.5),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(generator, epochs=epochs, verbose=0)

    # 5. Iterative forecasting
    forecast = []
    current_batch = scaled_data[-n_steps:].reshape((1, n_steps, 1))

    for _ in range(periods):
        current_pred = model.predict(current_batch, verbose=0)[0]
        forecast.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    # 6. Inverse transforms
    forecast_inversed = scaler.inverse_transform(forecast)
    if is_differenced:
        final_forecast = np.cumsum(forecast_inversed) + last_original_value
    else:
        final_forecast = forecast_inversed.flatten()

    return final_forecast

def create_lag_features(series, lags=[1, 2, 3]):
    df = pd.DataFrame(series)
    df.columns = ['y']
    for lag in lags:
        df[f'lag_{lag}'] = df['y'].shift(lag)
    df.dropna(inplace=True)
    X = df.drop(columns=['y'])
    y = df['y']
    return X, y

def random_forest_forecast(train_series, periods):
    print(">>> [Univariate] Running Random Forest Regressor Forecast (Optimized Hybrid, lags [1, 2, 3])...")
    try:
        # Fit optimized Holt-Winters baseline
        hw_model = ExponentialSmoothing(
            train_series, seasonal_periods=13, trend='add', seasonal='add',
            damped_trend=False, initialization_method='estimated'
        ).fit(optimized=True)
        hw_forecast = hw_model.forecast(periods)
        
        # Calculate residuals
        fitted_values = hw_model.fittedvalues
        residuals = train_series - fitted_values
        
        # Train Random Forest on residuals using lags 1, 2, 3
        lags_list = [1, 2, 3]
        X_train, y_train = create_lag_features(residuals, lags=lags_list)
        rf = RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_leaf=1, random_state=42)
        rf.fit(X_train.values, y_train.values)
        
        # Predict residuals recursively
        res_history = list(residuals.values)
        pred_res = []
        for _ in range(periods):
            features = [res_history[-lag] for lag in lags_list]
            pred = rf.predict([features])[0]
            pred_res.append(pred)
            res_history.append(pred)
            
        # Final forecast = Holt-Winters baseline + ML predicted residuals
        return hw_forecast.values + np.array(pred_res)
    except Exception as e:
        print(f"    ! RF hybrid failed: {e}. Falling back to Holt-Winters forecast.")
        return holt_winters_forecast(train_series, periods)

def sarima_forecast(train_series, periods):
    print(">>> [Univariate] Running SARIMA Forecast...")
    try:
        model = SARIMAX(train_series, order=(1,1,1), seasonal_order=(1,1,1,12), 
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        return model.forecast(periods)
    except Exception as e:
        print(f"    ! SARIMA failed: {e}. Falling back to series mean.")
        return np.full(periods, train_series.mean())

def xgboost_forecast(train_series, periods):
    print(">>> [Univariate] Running XGBoost Forecast (Optimized Hybrid, lags [1, 2, 3])...")
    if not HAS_XGBOOST:
        print("    ! XGBoost not available.")
        return np.full(periods, np.nan)
    try:
        # Fit optimized Holt-Winters baseline
        hw_model = ExponentialSmoothing(
            train_series, seasonal_periods=13, trend='add', seasonal='add',
            damped_trend=False, initialization_method='estimated'
        ).fit(optimized=True)
        hw_forecast = hw_model.forecast(periods)
        
        # Calculate residuals
        fitted_values = hw_model.fittedvalues
        residuals = train_series - fitted_values
        
        # Train XGBoost on residuals using lags 1, 2, 3
        lags_list = [1, 2, 3]
        X_train, y_train = create_lag_features(residuals, lags=lags_list)
        model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, subsample=1.0, random_state=42, objective='reg:squarederror')
        model.fit(X_train.values, y_train.values)
        
        # Predict residuals recursively
        res_history = list(residuals.values)
        pred_res = []
        for _ in range(periods):
            features = [res_history[-lag] for lag in lags_list]
            pred = model.predict(np.array([features]))[0]
            pred_res.append(pred)
            res_history.append(pred)
            
        # Final forecast = Holt-Winters baseline + ML predicted residuals
        return hw_forecast.values + np.array(pred_res)
    except Exception as e:
        print(f"    ! XGBoost hybrid failed: {e}. Falling back to Holt-Winters forecast.")
        return holt_winters_forecast(train_series, periods)

def lightgbm_forecast(train_series, periods):
    print(">>> [Univariate] Running LightGBM Forecast (Optimized Hybrid, lags [1, 2, 3])...")
    if not HAS_LIGHTGBM:
        print("    ! LightGBM not available.")
        return np.full(periods, np.nan)
    try:
        # Fit optimized Holt-Winters baseline
        hw_model = ExponentialSmoothing(
            train_series, seasonal_periods=13, trend='add', seasonal='add',
            damped_trend=False, initialization_method='estimated'
        ).fit(optimized=True)
        hw_forecast = hw_model.forecast(periods)
        
        # Calculate residuals
        fitted_values = hw_model.fittedvalues
        residuals = train_series - fitted_values
        
        # Train LightGBM on residuals using lags 1, 2, 3
        lags_list = [1, 2, 3]
        X_train, y_train = create_lag_features(residuals, lags=lags_list)
        model = lgb.LGBMRegressor(n_estimators=150, max_depth=-1, learning_rate=0.01, min_child_samples=5, random_state=42, verbose=-1)
        model.fit(X_train.values, y_train.values)
        
        # Predict residuals recursively
        res_history = list(residuals.values)
        pred_res = []
        for _ in range(periods):
            features = [res_history[-lag] for lag in lags_list]
            pred = model.predict(np.array([features]))[0]
            pred_res.append(pred)
            res_history.append(pred)
            
        # Final forecast = Holt-Winters baseline + ML predicted residuals
        return hw_forecast.values + np.array(pred_res)
    except Exception as e:
        print(f"    ! LightGBM hybrid failed: {e}. Falling back to Holt-Winters forecast.")
        return holt_winters_forecast(train_series, periods)

def calculate_accuracy_metrics(actual, predicted):
    # Filter out NaNs if any
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    if not np.any(mask):
        return np.nan, np.nan, np.nan
    act = actual[mask]
    pred = predicted[mask]
    mae = np.mean(np.abs(act - pred))
    rmse = np.sqrt(np.mean((act - pred)**2))
    mape = np.mean(np.abs((act - pred) / act)) * 100
    return mae, rmse, mape

def run_univariate_comparison(master_path):
    print("\n====== RUNNING UNIVARIATE MODEL BENCHMARKING (TRAIN/TEST SPLIT) ======")
    weekly_orders = prepare_univariate_data(master_path)
    
    # 12-week test window for validation
    train_series = weekly_orders.iloc[:-FORECAST_PERIODS]
    test_series = weekly_orders.iloc[-FORECAST_PERIODS:]
    print(f">>> [Univariate] Split details: Train size = {len(train_series)} weeks, Test size = {len(test_series)} weeks")

    # Run predictions on test window
    forecast_naive = naive_forecast(train_series, FORECAST_PERIODS)
    forecast_hw = holt_winters_forecast(train_series, FORECAST_PERIODS)
    forecast_sarima = sarima_forecast(train_series, FORECAST_PERIODS)
    forecast_rf = random_forest_forecast(train_series, FORECAST_PERIODS)
    forecast_xgb = xgboost_forecast(train_series, FORECAST_PERIODS)
    forecast_lgb = lightgbm_forecast(train_series, FORECAST_PERIODS)
    
    if HAS_TENSORFLOW:
        forecast_lstm = univariate_lstm_forecast(train_series, FORECAST_PERIODS, n_steps=8, epochs=EPOCHS)
    else:
        lstm_path = os.path.join(PROCESSED_DATA_DIR, 'LSTM_Forecast_Results.csv')
        if os.path.exists(lstm_path):
            print(">>> [Univariate] Loading pre-calculated LSTM predictions from file...")
            try:
                df_lstm_old = pd.read_csv(lstm_path)
                df_lstm_old.columns = ['date', 'Predicted_Order_Count']
                df_lstm_old['parsed_date'] = pd.to_datetime(df_lstm_old['date'], errors='coerce')
                df_lstm_old_temp = df_lstm_old.dropna(subset=['parsed_date']).set_index('parsed_date')
                lstm_aligned = df_lstm_old_temp.reindex(test_series.index)
                forecast_lstm = lstm_aligned['Predicted_Order_Count'].values
                
                # Fallback to positional alignment if no date overlap
                if np.sum(~np.isnan(forecast_lstm)) == 0:
                    print("    - No date overlap. Falling back to positional alignment for validation comparison.")
                    forecast_lstm = df_lstm_old['Predicted_Order_Count'].values[:FORECAST_PERIODS]
                else:
                    print(f"    - Aligned {np.sum(~np.isnan(forecast_lstm))} overlapping LSTM predictions by date.")
            except Exception as e:
                print(f"    ! Failed to load pre-calculated LSTM: {e}")
                forecast_lstm = np.full(FORECAST_PERIODS, np.nan)
        else:
            forecast_lstm = np.full(FORECAST_PERIODS, np.nan)
            print(">>> [Univariate] Skipping LSTM Forecast (no pre-calculated file found).")

    # Calculate metrics
    actuals = test_series.values
    metrics = {}
    metrics['Holt-Winters'] = calculate_accuracy_metrics(actuals, forecast_hw)
    metrics['SARIMA'] = calculate_accuracy_metrics(actuals, forecast_sarima)
    metrics['Random Forest'] = calculate_accuracy_metrics(actuals, forecast_rf)
    if HAS_XGBOOST:
        metrics['XGBoost'] = calculate_accuracy_metrics(actuals, forecast_xgb)
    if HAS_LIGHTGBM:
        metrics['LightGBM'] = calculate_accuracy_metrics(actuals, forecast_lgb)
    if HAS_TENSORFLOW or not np.all(np.isnan(forecast_lstm)):
        metrics['LSTM (Univariate)'] = calculate_accuracy_metrics(actuals, forecast_lstm)

    # Save metrics to CSV
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['MAE', 'RMSE', 'MAPE (%)'])
    metrics_df.index.name = 'Model'
    metrics_csv_path = os.path.join(PROCESSED_DATA_DIR, 'Model_Accuracy_Metrics.csv')
    metrics_df.to_csv(metrics_csv_path)
    
    print("\n====================================================")
    print("MODEL PERFORMANCE BENCHMARKING RESULTS (TEST SET)")
    print("====================================================")
    print(metrics_df.round(4).to_string())
    
    # Find best model
    valid_metrics = metrics_df.dropna(subset=['MAPE (%)'])
    if not valid_metrics.empty:
        best_model = valid_metrics['MAPE (%)'].idxmin()
        best_mape = valid_metrics.loc[best_model, 'MAPE (%)']
        print(f"\n🏆 Best model based on MAPE: {best_model} ({best_mape:.2f}%)")
    print("====================================================")

    # Save forecasts comparison on test set
    df_comparison = pd.DataFrame({
        'date': test_series.index,
        'Actual': actuals,
        'HoltWinters_Forecast': forecast_hw.values,
        'SARIMA_Forecast': forecast_sarima.values if hasattr(forecast_sarima, 'values') else forecast_sarima,
        'RandomForest_Forecast': forecast_rf,
        'XGBoost_Forecast': forecast_xgb,
        'LightGBM_Forecast': forecast_lgb,
        'LSTM_Forecast': forecast_lstm
    })
    
    comparison_csv_path = os.path.join(PROCESSED_DATA_DIR, 'Forecast_Comparison_Results.csv')
    df_comparison.to_csv(comparison_csv_path, index=False)
    print(f"\n>>> [Univariate] Exported test-set comparison CSV to: {comparison_csv_path}")

    # Plot Comparison on test set (Zoomed-in, plotting only the best models)
    import matplotlib.dates as mdates
    
    plt.figure(figsize=(14, 8))
    
    # Plot actual test set
    plt.plot(test_series.index, actuals, label='Actual Orders (Test Set)', color='#0f172a', linewidth=4, marker='o', markersize=8, zorder=5)
    
    mape_hw = metrics['Holt-Winters'][2]
    mape_rf = metrics['Random Forest'][2]
    
    plt.plot(test_series.index, forecast_hw, label=f'Holt-Winters (MAPE {mape_hw:.2f}%)', color='#059669', linestyle='-', linewidth=3, marker='s', markersize=6)
    plt.plot(test_series.index, forecast_rf, label=f'Random Forest Hybrid (MAPE {mape_rf:.2f}%)', color='#d97706', linestyle='--', linewidth=3, marker='^', markersize=7)
    
    if HAS_XGBOOST:
        mape_xgb = metrics['XGBoost'][2]
        plt.plot(test_series.index, forecast_xgb, label=f'XGBoost Hybrid (MAPE {mape_xgb:.2f}%)', color='#dc2626', linestyle=':', linewidth=3, marker='x', markersize=7)

    plt.title('Out-of-Sample Forecasting Model Benchmarking (Test Set Zoomed-in)', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=14, labelpad=12)
    plt.ylabel('Weekly Orders', fontsize=14, labelpad=12)
    
    # Format axes
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.SU))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
    plt.xticks(rotation=30, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    
    # Set y-axis limits dynamically with margin
    all_values = list(actuals) + list(forecast_hw) + list(forecast_rf)
    if HAS_XGBOOST:
        all_values += list(forecast_xgb)
    
    y_min, y_max = min(all_values), max(all_values)
    plt.ylim(y_min * 0.95, y_max * 1.05)
    
    plt.legend(frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=12, loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    chart_path = os.path.join(FIGURES_DIR, 'Chart_5_Forecast_Comparison.png')
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f">>> [Univariate] Saved clean zoomed model comparison chart at: {chart_path}")

# =============================================================================
# 2. ADVANCED MULTIVARIATE LSTM
# =============================================================================
def prepare_multivariate_lstm_data(master_path):
    print(">>> [Multivariate] Preparing advanced timeseries data (Log, Diff, Outliers)...")
    df = pd.read_csv(master_path, parse_dates=['order_purchase_timestamp'])
    df.set_index('order_purchase_timestamp', inplace=True)

    # Identify features
    feature_cols = [
        'order_id', 'weekly_active_sellers', 'weekly_active_customers', 'weekly_product_variety',
        'weekly_gmv', 'weekly_avg_basket_size', 'weekly_avg_price', 'weekly_avg_freight_value'
    ]
    existing_cols = [col for col in feature_cols if col in df.columns]
    
    # Fallback to order-level details if main features don't exist
    df_weekly = df[existing_cols].resample('W').first().fillna(0)
    order_counts = df['order_id'].resample('W').nunique()
    df_weekly['order_count'] = order_counts
    
    if 'order_id' in df_weekly.columns:
        df_weekly = df_weekly.drop(columns=['order_id'])
        
    df_weekly.rename(columns=lambda c: c.replace('weekly_', ''), inplace=True)
    df_weekly = df_weekly[df_weekly.index >= '2017-01-01']
    if len(df_weekly) > 0:
        df_weekly = df_weekly.iloc[:-1]

    # Black Friday Outlier Flag
    peak_date = df_weekly['order_count'].idxmax()
    df_weekly['black_friday_peak'] = 0
    df_weekly.loc[peak_date, 'black_friday_peak'] = 1

    # Log Transformation
    for col in df_weekly.columns:
        if col != 'black_friday_peak':
            df_weekly[col] = np.log1p(df_weekly[col])

    original_log_data = df_weekly.copy()

    # Differencing for Stationarity
    for col in df_weekly.columns:
        if col != 'black_friday_peak':
            df_weekly[col] = df_weekly[col].diff()

    df_weekly.dropna(inplace=True)
    print("    - Multivariate LSTM dataset features shape:", df_weekly.shape)
    return df_weekly, original_log_data

def run_multivariate_lstm(master_path):
    print("\n====== RUNNING MULTIVARIATE LSTM FORECASTING ======")
    if not HAS_TENSORFLOW:
        print("\n[WARNING] Tensorflow package is not installed.")
        print("Tensorflow is not supported on Python 3.14+ yet.")
        print("To run multivariate LSTM forecasting, please run this script in an environment with Python 3.8 - 3.12.")
        print("\n====================================================")
        print("MULTIVARIATE LSTM SKIPPED (ENVIRONMENT COMPATIBILITY)")
        print("====================================================")
        return

    df_processed, df_log_original = prepare_multivariate_lstm_data(master_path)

    # Train/Val/Test Split
    val_weeks = 12
    test_weeks = 12
    
    train_df = df_processed.iloc[:-(val_weeks + test_weeks)]
    val_df = df_processed.iloc[-(val_weeks + test_weeks):-test_weeks]
    test_df = df_processed.iloc[-test_weeks:]

    print(f">>> [Multivariate] Data Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    target_col = 'order_count'
    feature_cols = [col for col in df_processed.columns if col != target_col]

    # Normalize
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_scaled = scaler_X.fit_transform(train_df[feature_cols])
    y_train_scaled = scaler_y.fit_transform(train_df[[target_col]])
    X_val_scaled = scaler_X.transform(val_df[feature_cols])
    y_val_scaled = scaler_y.transform(val_df[[target_col]])

    # Timeseries Generator
    n_features = X_train_scaled.shape[1]
    train_generator = TimeseriesGenerator(X_train_scaled, y_train_scaled, length=N_STEPS, batch_size=1)
    val_generator = TimeseriesGenerator(X_val_scaled, y_val_scaled, length=N_STEPS, batch_size=1)

    # Model Architecture
    print(">>> [Multivariate] Training LSTM network...")
    model = Sequential([
        LSTM(75, activation='relu', input_shape=(N_STEPS, n_features), return_sequences=True),
        Dropout(0.3),
        LSTM(50, activation='relu'),
        Dropout(0.3),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=25, mode='min', verbose=0, restore_best_weights=True)
    model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS, callbacks=[early_stopping], verbose=0)

    # Forecast
    print(f">>> [Multivariate] Forecasting {FORECAST_PERIODS} weeks ahead...")
    full_processed_X = df_processed[feature_cols]
    scaled_full_X = scaler_X.transform(full_processed_X)

    future_predictions_scaled = []
    current_batch = scaled_full_X[-N_STEPS:].reshape((1, N_STEPS, n_features))

    for _ in range(FORECAST_PERIODS):
        current_pred_scaled = model.predict(current_batch, verbose=0)[0]
        future_predictions_scaled.append(current_pred_scaled)

        # Shift batch and pad other features with last values
        last_known_features_diff = current_batch[0, -1, :]
        new_row = last_known_features_diff.reshape(1, 1, n_features)
        current_batch = np.append(current_batch[:, 1:, :], new_row, axis=1)

    # Inverse transforms
    predictions_diff_log = scaler_y.inverse_transform(future_predictions_scaled)
    last_log_value = df_log_original['order_count'].iloc[-1]
    predictions_log = last_log_value + np.cumsum(predictions_diff_log)
    final_predictions = np.expm1(predictions_log)

    future_dates = pd.date_range(start=df_log_original.index[-1], periods=FORECAST_PERIODS + 1, freq='W')[1:]
    df_forecast = pd.DataFrame({'Predicted_Order_Count': final_predictions.flatten()}, index=future_dates)

    # Save results
    csv_path = os.path.join(PROCESSED_DATA_DIR, 'LSTM_Forecast_Results.csv')
    df_forecast.to_csv(csv_path)
    print(f">>> [Multivariate] Exported LSTM forecast to: {csv_path}")

    # Plot
    plt.figure(figsize=(18, 9))
    plt.plot(np.expm1(df_log_original['order_count']), label='Historical Data', color='gray')
    plt.plot(df_forecast.index, df_forecast['Predicted_Order_Count'], label=f'LSTM Forecast ({FORECAST_PERIODS} Weeks)', color='red', marker='x')

    plt.title('Advanced Multivariate LSTM Future Forecast', fontsize=20)
    plt.xlabel('Date')
    plt.ylabel('Weekly Orders')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)

    chart_path = os.path.join(FIGURES_DIR, 'Chart_6_LSTM_Forecast.png')
    plt.savefig(chart_path)
    plt.close()
    print(f">>> [Multivariate] Saved LSTM forecast chart at: {chart_path}")

def main():
    print("====================================================")
    print("STARTING DEMAND FORECASTING RUNNER (UNIVARIATE & MULTIVARIATE)")
    print("====================================================")
    
    master_path = os.path.join(PROCESSED_DATA_DIR, 'Master_Logistics_Data.csv')
    if not os.path.exists(master_path):
        print(f"ERROR: File not found at {master_path}. Please run `main.py` first.")
        return

    # Run univariate comparison
    run_univariate_comparison(master_path)

    # Run multivariate LSTM
    run_multivariate_lstm(master_path)
    
    print("\n====================================================")
    print("DEMAND FORECASTING COMPLETE")
    print("====================================================")

if __name__ == "__main__":
    main()
