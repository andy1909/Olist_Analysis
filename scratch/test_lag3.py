# scratch/test_lag3.py
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
master_path = os.path.join(BASE_DIR, 'data', 'processed', 'Master_Logistics_Data.csv')

# Load and prepare data
df = pd.read_csv(master_path, parse_dates=['order_purchase_timestamp'])
df_ts = df.set_index('order_purchase_timestamp')
weekly_orders = df_ts.resample('W')['order_id'].nunique().fillna(0)
weekly_orders = weekly_orders[weekly_orders.index >= '2017-01-01']
weekly_orders = weekly_orders.asfreq('W').fillna(0).astype(float)
weekly_orders = weekly_orders[weekly_orders.index <= '2018-05-20']

FORECAST_PERIODS = 12
train_series = weekly_orders.iloc[:-FORECAST_PERIODS]
test_series = weekly_orders.iloc[-FORECAST_PERIODS:]
actuals = test_series.values

def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

print(f"Data details: Train={len(train_series)} weeks, Test={len(test_series)} weeks")

# ==========================================================
# APPROACH 1: Lags [1, 2, 3] only (raw)
# ==========================================================
def run_approach_1():
    def create_features(series):
        df = pd.DataFrame(series)
        df.columns = ['y']
        for lag in [1, 2, 3]:
            df[f'lag_{lag}'] = df['y'].shift(lag)
        df.dropna(inplace=True)
        return df.drop(columns=['y']), df['y']

    X_train, y_train = create_features(train_series)
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train.values, y_train.values)
    history = list(train_series.values)
    pred_rf = []
    for _ in range(FORECAST_PERIODS):
        features = [history[-1], history[-2], history[-3]]
        pred = rf.predict([features])[0]
        pred_rf.append(pred)
        history.append(pred)
    
    # XGBoost
    model_xgb = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model_xgb.fit(X_train.values, y_train.values)
    history = list(train_series.values)
    pred_xgb = []
    for _ in range(FORECAST_PERIODS):
        features = [history[-1], history[-2], history[-3]]
        pred = model_xgb.predict(np.array([features]))[0]
        pred_xgb.append(pred)
        history.append(pred)

    # LightGBM
    model_lgb = lgb.LGBMRegressor(n_estimators=100, random_state=42, min_child_samples=5, verbose=-1)
    model_lgb.fit(X_train.values, y_train.values)
    history = list(train_series.values)
    pred_lgb = []
    for _ in range(FORECAST_PERIODS):
        features = [history[-1], history[-2], history[-3]]
        pred = model_lgb.predict(np.array([features]))[0]
        pred_lgb.append(pred)
        history.append(pred)
        
    print("\n--- APPROACH 1: Lags [1, 2, 3] only (Raw) ---")
    print(f"Random Forest MAPE: {calculate_mape(actuals, np.array(pred_rf)):.2f}%")
    print(f"XGBoost MAPE:       {calculate_mape(actuals, np.array(pred_xgb)):.2f}%")
    print(f"LightGBM MAPE:      {calculate_mape(actuals, np.array(pred_lgb)):.2f}%")
    return np.array(pred_rf), np.array(pred_xgb), np.array(pred_lgb)

# ==========================================================
# APPROACH 2: Lags [1, 2, 3] + Calendar Features (week, month)
# ==========================================================
def run_approach_2():
    def create_features(series):
        df = pd.DataFrame(series)
        df.columns = ['y']
        for lag in [1, 2, 3]:
            df[f'lag_{lag}'] = df['y'].shift(lag)
        df['week_of_year'] = df.index.isocalendar().week.astype(float)
        df['month'] = df.index.month.astype(float)
        df.dropna(inplace=True)
        return df.drop(columns=['y']), df['y']

    X_train, y_train = create_features(train_series)
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train.values, y_train.values)
    history = list(train_series.values)
    pred_rf = []
    test_dates = test_series.index
    for i in range(FORECAST_PERIODS):
        date = test_dates[i]
        week_val = float(date.isocalendar()[1])
        month_val = float(date.month)
        features = [history[-1], history[-2], history[-3], week_val, month_val]
        pred = rf.predict([features])[0]
        pred_rf.append(pred)
        history.append(pred)
    
    # XGBoost
    model_xgb = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model_xgb.fit(X_train.values, y_train.values)
    history = list(train_series.values)
    pred_xgb = []
    for i in range(FORECAST_PERIODS):
        date = test_dates[i]
        week_val = float(date.isocalendar()[1])
        month_val = float(date.month)
        features = [history[-1], history[-2], history[-3], week_val, month_val]
        pred = model_xgb.predict(np.array([features]))[0]
        pred_xgb.append(pred)
        history.append(pred)

    # LightGBM
    model_lgb = lgb.LGBMRegressor(n_estimators=100, random_state=42, min_child_samples=5, verbose=-1)
    model_lgb.fit(X_train.values, y_train.values)
    history = list(train_series.values)
    pred_lgb = []
    for i in range(FORECAST_PERIODS):
        date = test_dates[i]
        week_val = float(date.isocalendar()[1])
        month_val = float(date.month)
        features = [history[-1], history[-2], history[-3], week_val, month_val]
        pred = model_lgb.predict(np.array([features]))[0]
        pred_lgb.append(pred)
        history.append(pred)
        
    print("\n--- APPROACH 2: Lags [1, 2, 3] + Calendar Features ---")
    print(f"Random Forest MAPE: {calculate_mape(actuals, np.array(pred_rf)):.2f}%")
    print(f"XGBoost MAPE:       {calculate_mape(actuals, np.array(pred_xgb)):.2f}%")
    print(f"LightGBM MAPE:      {calculate_mape(actuals, np.array(pred_lgb)):.2f}%")
    return np.array(pred_rf), np.array(pred_xgb), np.array(pred_lgb)

# ==========================================================
# APPROACH 3: Residual Modeling (Holt-Winters residuals + Lags [1, 2, 3])
# ==========================================================
def run_approach_3():
    # 1. Fit Holt-Winters on training set
    hw_model = ExponentialSmoothing(
        train_series, seasonal_periods=12, trend='add', seasonal='add',
        damped_trend=True, initialization_method='estimated'
    ).fit(optimized=True)
    
    # 2. Get baseline forecast for test set
    hw_forecast = hw_model.forecast(FORECAST_PERIODS)
    
    # 3. Calculate residuals in training set
    fitted_values = hw_model.fittedvalues
    residuals = train_series - fitted_values
    
    # 4. Train ML models on residuals with lag 1, 2, 3
    def create_residual_features(res_series):
        df = pd.DataFrame(res_series)
        df.columns = ['res']
        for lag in [1, 2, 3]:
            df[f'lag_{lag}'] = df['res'].shift(lag)
        df.dropna(inplace=True)
        return df.drop(columns=['res']), df['res']
        
    X_train_res, y_train_res = create_residual_features(residuals)
    
    # Random Forest on residuals
    rf_res = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_res.fit(X_train_res.values, y_train_res.values)
    res_history = list(residuals.values)
    pred_rf_res = []
    for _ in range(FORECAST_PERIODS):
        features = [res_history[-1], res_history[-2], res_history[-3]]
        pred = rf_res.predict([features])[0]
        pred_rf_res.append(pred)
        res_history.append(pred)
    final_rf = hw_forecast.values + np.array(pred_rf_res)
    
    # XGBoost on residuals
    xgb_res = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_res.fit(X_train_res.values, y_train_res.values)
    res_history = list(residuals.values)
    pred_xgb_res = []
    for _ in range(FORECAST_PERIODS):
        features = [res_history[-1], res_history[-2], res_history[-3]]
        pred = xgb_res.predict(np.array([features]))[0]
        pred_xgb_res.append(pred)
        res_history.append(pred)
    final_xgb = hw_forecast.values + np.array(pred_xgb_res)

    # LightGBM on residuals
    lgb_res = lgb.LGBMRegressor(n_estimators=100, random_state=42, min_child_samples=5, verbose=-1)
    lgb_res.fit(X_train_res.values, y_train_res.values)
    res_history = list(residuals.values)
    pred_lgb_res = []
    for _ in range(FORECAST_PERIODS):
        features = [res_history[-1], res_history[-2], res_history[-3]]
        pred = lgb_res.predict(np.array([features]))[0]
        pred_lgb_res.append(pred)
        res_history.append(pred)
    final_lgb = hw_forecast.values + np.array(pred_lgb_res)

    print("\n--- APPROACH 3: Holt-Winters Residuals + Lags [1, 2, 3] ---")
    print(f"Random Forest residual MAPE: {calculate_mape(actuals, final_rf):.2f}%")
    print(f"XGBoost residual MAPE:       {calculate_mape(actuals, final_xgb):.2f}%")
    print(f"LightGBM residual MAPE:      {calculate_mape(actuals, final_lgb):.2f}%")
    return final_rf, final_xgb, final_lgb

p1 = run_approach_1()
p2 = run_approach_2()
p3 = run_approach_3()
