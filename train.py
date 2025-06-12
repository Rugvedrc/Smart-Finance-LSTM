import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from smartfinance import (
    process_user_transactions,
    categories,
    train_category_models,
    CONFIG
)
from project_utilities import validate_expense_csv
from sklearn.preprocessing import MinMaxScaler

def load_csv_and_validate(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    is_valid, message = validate_expense_csv(df)
    if not is_valid:
        raise ValueError(f"CSV validation failed: {message}")

    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].astype(str)
    df['Amount'] = df['Amount'].astype(float)

    transactions = df.to_dict('records')
    return transactions

def save_scalers(y, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i, category in enumerate(categories):
        y_cat = y[:, i].reshape(-1, 1)
        scaler = MinMaxScaler()
        scaler.fit(y_cat)
        scaler_path = os.path.join(save_dir, f"{category}_scaler.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"✅ Saved scaler: {scaler_path}")

def main():
    csv_file = "realistic_expense_data.csv"
    print(f"📁 Loading and validating: {csv_file}")
    transactions = load_csv_and_validate(csv_file)

    print(f"📊 Processing {len(transactions)} transactions into monthly data...")
    monthly_data = process_user_transactions(transactions)

    sequence_length = 3
    X, y = [], []
    for i in range(len(monthly_data) - sequence_length):
        X.append(monthly_data[i:i + sequence_length])
        y.append(monthly_data[i + sequence_length])
    if not X:
        raise ValueError("Not enough data to create sequences for training.")

    X = np.array(X)
    y = np.array(y)

    # Save processed arrays
    if CONFIG['X_path']:
        x_dir = os.path.dirname(CONFIG['X_path'])
        if x_dir:
            os.makedirs(x_dir, exist_ok=True)
        np.save(CONFIG['X_path'], X)
        print(f"💾 Saved: {CONFIG['X_path']}")
    if CONFIG['y_path']:
        y_dir = os.path.dirname(CONFIG['y_path'])
        if y_dir:
            os.makedirs(y_dir, exist_ok=True)
        np.save(CONFIG['y_path'], y)
        print(f"💾 Saved: {CONFIG['y_path']}")

    # Save per-category scalers
    print(f"⚖️ Saving scalers in: {CONFIG['scaler_dir']}")
    save_scalers(y, CONFIG['scaler_dir'])

    # Train models
    print(f"🚀 Starting training for {len(categories)} categories...")
    train_category_models(log_callback=lambda msg: print(f"[LOG] {msg}"))
    print("✅ All models trained and saved to:", CONFIG['model_dir'])

if __name__ == '__main__':
    main()
