# config.py
import os

CONFIG = {
    'model_dir': "category_models",
    'scaler_dir': "category_scalers",
    'global_scaler_path': "scaler.pkl",
    'X_path': "X.npy",
    'y_path': "y.npy",
    'budget_config_path': "budget_config.json",
    'user_data_dir': "userdata",
    'upload_dir': "uploads",
    'allowed_extensions': {'csv'},
    'prediction_confidence_threshold': 0.7,
    'lstm_anomaly_multiplier': 2.0,
    'min_historical_months': 3
}

os.makedirs(CONFIG['user_data_dir'], exist_ok=True)
os.makedirs(CONFIG['upload_dir'], exist_ok=True)


# file_storage.py
import json
import os

data_dir = "userdata"
os.makedirs(data_dir, exist_ok=True)

def safe_username(username):
    return ''.join(c for c in username if c.isalnum() or c in ('-', '_'))

def get_user_filepath(username):
    return os.path.join(data_dir, f"{safe_username(username)}.json")

def load_user_data(username):
    filepath = get_user_filepath(username)
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return {}
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"[ERROR] Invalid JSON in {filepath}")
        return {}


def save_user_data(username, data):
    filepath = get_user_filepath(username)
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save data for {username}: {e}")


def delete_user_data(username):
    filepath = get_user_filepath(username)
    if os.path.exists(filepath):
        os.remove(filepath)

def list_all_users():
    return [f[:-5] for f in os.listdir(data_dir) if f.endswith(".json")]


# csv_validator.py
import pandas as pd

def validate_expense_csv(df):
    required_columns = {'Date', 'Category', 'Amount'}
    if not required_columns.issubset(df.columns):
        return False, f"Missing columns: {required_columns - set(df.columns)}"
    return True, "Valid CSV"


# training_logger.py
from keras.callbacks import Callback

class TrainingLogger(Callback):
    def __init__(self):
        super().__init__()
        self.logs = {'loss': [], 'val_loss': []}

    def on_epoch_end(self, epoch, logs=None):
        self.logs['loss'].append(logs.get('loss'))
        self.logs['val_loss'].append(logs.get('val_loss'))

    def get_logs(self):
        return self.logs
