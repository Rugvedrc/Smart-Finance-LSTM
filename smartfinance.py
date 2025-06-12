import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import time
from keras import callbacks
from keras.models import load_model, Sequential
from keras.layers import GRU, SimpleRNN, Dense, LSTM, Dropout
from keras.optimizers import Adam
from keras import callbacks
from keras.models import load_model, Sequential
from keras.layers import GRU, SimpleRNN, Dense, LSTM, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib
matplotlib.use('Agg')  # MUST be before importing pyplot
import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping
import os
import json
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model, Sequential
from keras.layers import GRU, SimpleRNN, Dense, LSTM, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG = {
    'model_dir': "category_models",
    'scaler_dir': "category_scalers",
    'global_scaler_path': "scaler.pkl",
    'X_path': "X.npy",
    'y_path': "y.npy",
    'budget_config_path': "budget_config.json",
    'prediction_confidence_threshold': 0.7,
    'lstm_anomaly_multiplier': 2.0,
    'min_historical_months': 3
}

categories = [
    'Bills & Utilities', 'Education', 'Entertainment', 'food & Dining',
    'health & Medical', 'shopping', 'Travel & Transportation'
]

default_budget_limits = {
    "Bills & Utilities": {"limit": 15000, "volatility": 0.1},
    "Education": {"limit": 1000, "volatility": 0.3},
    "Entertainment": {"limit": 1200, "volatility": 0.5},
    "food & Dining": {"limit": 9000, "volatility": 0.2},
    "health & Medical": {"limit": 500, "volatility": 0.8},
    "shopping": {"limit": 2500, "volatility": 0.6},
    "Travel & Transportation": {"limit": 3000, "volatility": 0.4},
}

def invert_scaling(y_scaled, scaler, feature_index=None):
    if len(y_scaled.shape) == 1:
        y_scaled = y_scaled.reshape(-1, 1)
    if feature_index is not None:
        full_scale = np.zeros((y_scaled.shape[0], scaler.scale_.shape[0]))
        full_scale[:, feature_index] = y_scaled[:, 0]
        return scaler.inverse_transform(full_scale)[:, feature_index]
    return scaler.inverse_transform(y_scaled)

def load_scalers():
    try:
        scalers = {}
        if os.path.exists(CONFIG['scaler_dir']):
            for category in categories:
                scaler_path = os.path.join(CONFIG['scaler_dir'], f"{category}_scaler.pkl")
                if os.path.exists(scaler_path):
                    scalers[category] = joblib.load(scaler_path)
        
        if not scalers and os.path.exists(CONFIG['global_scaler_path']):
            global_scaler = joblib.load(CONFIG['global_scaler_path'])
            for category in categories:
                scalers[category] = global_scaler
        
        if not scalers:
            raise FileNotFoundError("No scalers found")
        
        return scalers, len(set(scalers.values())) == 1
    except Exception as e:
        logger.error(f"Error loading scalers: {e}")
        raise

def load_models():
    try:
        models = {}
        for category in categories:
            model_path = os.path.join(CONFIG['model_dir'], f"{category}_lstm.keras")
            if os.path.exists(model_path):
                models[category] = load_model(model_path)
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
        return models
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

def process_user_transactions(transactions):
    """Process user transactions into monthly spending data by category"""
    try:
        if not transactions:
            raise ValueError("No transaction data provided")
        
        df = pd.DataFrame(transactions)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df = df.dropna()
        
        # Group by year-month and category
        df['YearMonth'] = df['Date'].dt.to_period('M')
        monthly_spending = df.groupby(['YearMonth', 'Category'])['Amount'].sum().unstack(fill_value=0)
        
        # Ensure all categories are present
        for category in categories:
            if category not in monthly_spending.columns:
                monthly_spending[category] = 0
        
        # Reorder columns to match categories list
        monthly_spending = monthly_spending[categories]
        
        logger.info(f"Processed {len(df)} transactions into {len(monthly_spending)} monthly records")
        return monthly_spending.values  # Return as numpy array
        
    except Exception as e:
        logger.error(f"Error processing user transactions: {e}")
        raise

def analyze_user_spending_patterns(monthly_data):
    """Analyze user's spending patterns for better predictions"""
    patterns = {}
    
    for i, category in enumerate(categories):
        category_data = monthly_data[:, i]
        
        if len(category_data) >= 2:
            # Calculate trends
            trend = np.polyfit(range(len(category_data)), category_data, 1)[0]
            
            # Calculate seasonality (basic)
            monthly_avg = np.mean(category_data)
            monthly_std = np.std(category_data)
            
            # Identify spending spikes
            spikes = category_data > (monthly_avg + 2 * monthly_std)
            
            patterns[category] = {
                'trend': trend,
                'avg': monthly_avg,
                'std': monthly_std,
                'volatility': monthly_std / (monthly_avg + 1),
                'has_spikes': np.any(spikes),
                'recent_trend': category_data[-1] - category_data[-2] if len(category_data) >= 2 else 0
            }
        else:
            patterns[category] = {
                'trend': 0,
                'avg': category_data[0] if len(category_data) > 0 else 0,
                'std': 0,
                'volatility': 0,
                'has_spikes': False,
                'recent_trend': 0
            }
    
    return patterns

def create_user_sequences(monthly_data, sequence_length=3):
    """Create input sequences from user's monthly spending data"""
    try:
        if len(monthly_data) < sequence_length:
            # If not enough data, pad with zeros or use available data
            if len(monthly_data) == 0:
                return np.zeros((1, sequence_length, len(categories)))
            
            # Pad with the last available month's data
            padding_needed = sequence_length - len(monthly_data)
            last_month = monthly_data[-1] if len(monthly_data) > 0 else np.zeros(len(categories))
            padded_data = np.vstack([np.tile(last_month, (padding_needed, 1)), monthly_data])
            return padded_data[-sequence_length:].reshape(1, sequence_length, len(categories))
        
        # Use the last sequence_length months
        latest_sequence = monthly_data[-sequence_length:]
        return latest_sequence.reshape(1, sequence_length, len(categories))
        
    except Exception as e:
        logger.error(f"Error creating user sequences: {e}")
        raise

def generate_dynamic_budget(user_transactions):
    """Generate budget suggestions based on user's spending patterns"""
    try:
        monthly_data = process_user_transactions(user_transactions)
        if len(monthly_data) == 0:
            return default_budget_limits
        
        dynamic_budget = {}
        for i, category in enumerate(categories):
            category_spending = monthly_data[:, i]
            if len(category_spending) > 0:
                # Set budget as 110% of average spending + 1 std deviation
                avg_spending = np.mean(category_spending)
                std_spending = np.std(category_spending)
                suggested_limit = avg_spending + std_spending
                
                # Ensure minimum budget
                suggested_limit = max(suggested_limit, 500)
                
                volatility = std_spending / (avg_spending + 1) if avg_spending > 0 else 0.5
                
                dynamic_budget[category] = {
                    "limit": round(suggested_limit, 2),
                    "volatility": round(min(volatility, 1.0), 2)
                }
            else:
                dynamic_budget[category] = default_budget_limits[category]
        
        return dynamic_budget
    except Exception as e:
        logger.error(f"Error generating dynamic budget: {e}")
        return default_budget_limits

def generate_user_predictions(models, scalers, user_transactions, is_global_scaler):
    """Generate predictions based on user's actual transaction data with trend analysis"""
    try:
        # Process user data into monthly spending
        monthly_data = process_user_transactions(user_transactions)
        
        if len(monthly_data) == 0:
            return {category: 500.0 for category in categories}, {category: 0.3 for category in categories}
        
        predictions = {}
        confidences = {}
        
        for i, category in enumerate(categories):
            category_spending = monthly_data[:, i]
            
            # Calculate trend-based prediction
            if len(category_spending) >= 3:
                # Use weighted average with recent months having more weight
                weights = np.exp(np.linspace(-1, 0, len(category_spending)))
                weights = weights / weights.sum()
                trend_prediction = np.average(category_spending, weights=weights)
                
                # Calculate growth rate from last 3 months
                recent_months = category_spending[-3:]
                if len(recent_months) >= 2 and recent_months[0] > 0:
                    growth_rate = (recent_months[-1] - recent_months[0]) / recent_months[0]
                    growth_rate = np.clip(growth_rate, -0.5, 0.5)  # Limit extreme changes
                    trend_prediction *= (1 + growth_rate * 0.3)  # Apply 30% of growth rate
                
                # Calculate seasonal adjustment (if we have enough data)
                if len(category_spending) >= 6:
                    current_month = len(category_spending) % 12
                    seasonal_factor = 1.0
                    
                    # Simple seasonal adjustments for common patterns
                    if category == 'Entertainment':
                        # Higher in December/January, summer months
                        if current_month in [11, 0, 5, 6, 7]:  # Dec, Jan, Jun, Jul, Aug
                            seasonal_factor = 1.2
                    elif category == 'Travel & Transportation':
                        # Higher in summer and holidays
                        if current_month in [5, 6, 7, 11, 0]:
                            seasonal_factor = 1.3
                    elif category == 'shopping':
                        # Higher during festival seasons
                        if current_month in [10, 11, 0]:  # Nov, Dec, Jan
                            seasonal_factor = 1.4
                    
                    trend_prediction *= seasonal_factor
                
                # Confidence based on spending consistency
                recent_std = np.std(category_spending[-min(3, len(category_spending)):])
                recent_mean = np.mean(category_spending[-min(3, len(category_spending)):])
                
                if recent_mean > 0:
                    cv = recent_std / recent_mean  # Coefficient of variation
                    confidence = max(0.3, 1.0 - cv)  # Higher consistency = higher confidence
                else:
                    confidence = 0.3
                
            elif len(category_spending) >= 1:
                # Limited data - use simple average with growth
                recent_avg = np.mean(category_spending[-2:]) if len(category_spending) >= 2 else category_spending[-1]
                last_month = category_spending[-1]
                
                # Simple growth prediction
                if len(category_spending) >= 2 and category_spending[-2] > 0:
                    growth = (last_month - category_spending[-2]) / category_spending[-2]
                    growth = np.clip(growth, -0.3, 0.3)
                    trend_prediction = last_month * (1 + growth * 0.5)
                else:
                    trend_prediction = recent_avg
                
                confidence = 0.4  # Lower confidence due to limited data
            else:
                trend_prediction = 100.0  # Minimal default
                confidence = 0.2
            
            # Apply user-specific volatility adjustments
            if len(category_spending) >= 2:
                user_volatility = np.std(category_spending) / (np.mean(category_spending) + 1)
                volatility_adjustment = 1 + (user_volatility * 0.1)  # Small adjustment for volatility
                trend_prediction *= volatility_adjustment
            
            # Ensure non-negative predictions
            trend_prediction = max(0, trend_prediction)
            
            # Apply some randomness for realism (small amount)
            if trend_prediction > 0:
                noise_factor = 1 + np.random.normal(0, 0.05)  # 5% noise
                trend_prediction *= abs(noise_factor)
            
            predictions[category] = trend_prediction
            confidences[category] = np.clip(confidence, 0.2, 0.95)
        
        logger.info(f"Generated user-specific predictions for {len(categories)} categories")
        return predictions, confidences
        
    except Exception as e:
        logger.error(f"Error generating user predictions: {e}")
        # Return user-data-influenced fallback
        try:
            monthly_data = process_user_transactions(user_transactions)
            if len(monthly_data) > 0:
                fallback_predictions = {}
                for i, category in enumerate(categories):
                    avg_spending = np.mean(monthly_data[:, i]) if len(monthly_data) > 0 else 200
                    fallback_predictions[category] = max(avg_spending * 1.1, 100)  # 10% increase as prediction
                return fallback_predictions, {category: 0.4 for category in categories}
        except:
            pass
        
        return {category: 300.0 for category in categories}, {category: 0.3 for category in categories}
def load_components():
    try:
        models = load_models()
        scalers, is_global_scaler = load_scalers()
        X_input = np.load(CONFIG['X_path'])
        y_true = np.load(CONFIG['y_path'])
        return models, scalers, X_input, y_true, is_global_scaler
    except Exception as e:
        logger.error(f"Error loading components: {e}")
        raise

def load_budget_config():
    try:
        if os.path.exists(CONFIG['budget_config_path']):
            with open(CONFIG['budget_config_path'], 'r') as f:
                return json.load(f)
        return default_budget_limits
    except Exception as e:
        logger.warning(f"Error loading budget config: {e}")
        return default_budget_limits

def validate_data(X_input, y_true):
    if X_input.shape[0] < CONFIG['min_historical_months']:
        raise ValueError(f"Insufficient historical data: {X_input.shape[0]} months")
    if X_input.shape[2] != len(categories):
        raise ValueError(f"Category mismatch: {X_input.shape[2]} vs {len(categories)}")
    if np.any(np.isnan(X_input)) or np.any(np.isnan(y_true)):
        raise ValueError("Data contains NaN values")
    if np.any(X_input < 0) or np.any(y_true < 0):
        logger.warning("Negative values detected")
    return True

def generate_predictions(models, scalers, X_input, is_global_scaler):
    try:
        timesteps = X_input.shape[1]
        predictions = {}
        confidences = {}

        for i, category in enumerate(categories):
            X_cat = X_input[:, :, i].reshape(X_input.shape[0], timesteps, 1)
            latest_sequence = X_cat[-1].reshape(1, timesteps, 1)

            predictions_list = []
            for _ in range(10):
                pred_scaled = models[category].predict(latest_sequence, verbose=0)
                predictions_list.append(pred_scaled[0, 0])

            mean_pred_scaled = np.mean(predictions_list)
            std_pred_scaled = np.std(predictions_list)
            scaler = scalers[category]

            if is_global_scaler:
                dummy_pred = np.zeros((1, len(categories)))
                dummy_pred[0, i] = mean_pred_scaled
                y_pred_category = scaler.inverse_transform(dummy_pred)[0, i]
            else:
                y_pred_category = scaler.inverse_transform([[mean_pred_scaled]])[0, 0]

            y_pred_category = max(y_pred_category, 0)
            confidence = 1 / (1 + abs(std_pred_scaled / y_pred_category)) if y_pred_category != 0 else 0.5
            confidence = np.clip(confidence, 0.1, 1.0)

            predictions[category] = y_pred_category
            confidences[category] = confidence

        return predictions, confidences
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        raise

def lstm_anomaly_detection(y_true, predictions, confidences, scalers, is_global_scaler=False):
    try:
        if len(y_true) < 2:
            return []

        last_month_data = y_true[-1] if len(y_true.shape) > 1 else y_true

        if len(y_true.shape) > 1:
            if is_global_scaler:
                scaler = list(scalers.values())[0]
                last_month_actual = scaler.inverse_transform(last_month_data.reshape(1, -1)).flatten()
            else:
                last_month_actual = np.zeros(len(categories))
                for i, category in enumerate(categories):
                    scaler = scalers[category]
                    last_month_actual[i] = scaler.inverse_transform([[last_month_data[i]]])[0, 0]
        else:
            last_month_actual = last_month_data

        last_month_actual = np.maximum(last_month_actual, 0)
        alerts = []

        for i, category in enumerate(categories):
            if i >= len(last_month_actual):
                continue

            actual_last = last_month_actual[i]
            predicted = predictions[category]
            conf = confidences[category]

            if actual_last < 1 and predicted < 1:
                continue

            residual = abs(predicted - actual_last)

            if len(y_true) >= 3 and len(y_true.shape) > 1:
                historical_data = []
                for j in range(3):
                    h_val = y_true[-(j+1), i]
                    if is_global_scaler:
                        scaler = list(scalers.values())[0]
                        h_dummy = np.zeros((1, len(categories)))
                        h_dummy[0, i] = h_val
                        h_inverse = scaler.inverse_transform(h_dummy)[0, i]
                    else:
                        scaler = scalers[category]
                        h_inverse = scaler.inverse_transform([[h_val]])[0, 0]
                    historical_data.append(max(0, h_inverse))

                historical_std = np.std(historical_data) if len(historical_data) > 1 else max(actual_last * 0.2, 100)
            else:
                historical_std = max(actual_last * 0.2, 100)

            if historical_std > 0:
                z_score = residual / historical_std
                confidence_factor = (1 - conf) if conf < 1 else 0.1
                anomaly_score = z_score * (1 + confidence_factor * CONFIG['lstm_anomaly_multiplier'])
            else:
                anomaly_score = 0

            budget_config = load_budget_config()
            volatility_factor = budget_config.get(category, {}).get('volatility', 0.3)
            threshold = 2.0 / (volatility_factor + 0.1)

            if anomaly_score > threshold and (residual > 100 or abs(predicted - actual_last) > max(actual_last * 0.3, 100)):
                severity = "HIGH" if anomaly_score > threshold * 1.5 else "MEDIUM"
                change_pct = ((predicted - actual_last) / max(actual_last, 1) * 100)
                direction = "increase" if change_pct > 0 else "decrease"

                alert = f"""🚨 {severity} LSTM Anomaly Detected in "{category}":
Predicted: ₹{predicted:.2f} | Last Month: ₹{actual_last:.2f} | Change: {change_pct:+.2f}%
Model Confidence: {conf:.2f} | Anomaly Score: {anomaly_score:.2f}
Pattern Analysis: Unusual {direction} detected by category-specific LSTM model.
Recommendation: Review recent transactions in this category for unexpected expenses or changes."""
                
                alerts.append({
                    'category': category,
                    'alert': alert,
                    'severity': severity,
                    'anomaly_score': anomaly_score,
                    'confidence': conf
                })

        return alerts
    except Exception as e:
        logger.error(f"Error in LSTM anomaly detection: {e}")
        return []

def check_budget(predictions, budgets):
    try:
        alerts = {}
        for category, predicted_amount in predictions.items():
            if category in budgets:
                budget_limit = budgets[category].get('limit', budgets[category]) if isinstance(budgets[category], dict) else budgets[category]
                if predicted_amount > budget_limit:
                    overage_pct = ((predicted_amount - budget_limit) / budget_limit) * 100
                    severity = "CRITICAL" if overage_pct > 50 else "HIGH" if overage_pct > 20 else "MEDIUM"
                    alerts[category] = {
                        'predicted': predicted_amount,
                        'budget': budget_limit,
                        'overage_pct': overage_pct,
                        'severity': severity
                    }
        return alerts
    except Exception as e:
        logger.error(f"Error in budget checking: {e}")
        return {}

def financial_health_score(predictions, budgets, last_month, confidences, anomaly_threshold=30):
    try:
        budget_violations = 0
        anomaly_count = 0
        total_categories = len(predictions)
        confidence_penalty = 0

        for i, category in enumerate(categories):
            pred_val = predictions[category]
            budget_info = budgets.get(category, {})
            budget_val = budget_info.get('limit', budget_info) if isinstance(budget_info, dict) else budget_info
            last_val = last_month[i] if i < len(last_month) else 0
            conf = confidences[category]

            if pred_val > budget_val:
                overage = (pred_val - budget_val) / budget_val
                budget_violations += min(overage, 1.0)

            if last_val != 0:
                pct_change = abs((pred_val - last_val) / last_val) * 100
                volatility = budget_info.get('volatility', 0.3) if isinstance(budget_info, dict) else 0.3
                adjusted_threshold = anomaly_threshold * (1 + volatility)
                if pct_change > adjusted_threshold:
                    anomaly_count += min(pct_change / adjusted_threshold / 2, 1.0)

            if conf < CONFIG['prediction_confidence_threshold']:
                confidence_penalty += (CONFIG['prediction_confidence_threshold'] - conf)

        score = 100
        score -= (budget_violations / total_categories) * 40
        score -= (anomaly_count / total_categories) * 30
        score -= (confidence_penalty / total_categories) * 30
        return max(0, min(100, round(score, 2)))
    except Exception as e:
        logger.error(f"Error calculating financial health score: {e}")
        return 50.0

def generate_recommendations(predictions, budgets, lstm_anomalies, budget_alerts, health_score):
    try:
        recommendations = []

        if budget_alerts:
            high_violations = [cat for cat, info in budget_alerts.items() if info['severity'] in ['HIGH', 'CRITICAL']]
            if high_violations:
                recommendations.append(f"🎯 Priority: Reduce spending in {', '.join(high_violations)} to stay within budget.")

        high_anomalies = [alert for alert in lstm_anomalies if alert['severity'] == 'HIGH']
        if high_anomalies:
            categories_list = [alert['category'] for alert in high_anomalies]
            recommendations.append(f"🔍 Investigate: Unusual patterns detected in {', '.join(categories_list)} by individual AI models.")

        if health_score < 50:
            recommendations.append("🚨 Urgent: Consider financial counseling or debt management strategies.")
        elif health_score < 70:
            recommendations.append("⚠️ Caution: Review and optimize your spending categories.")
        else:
            recommendations.append("✅ Good: Maintain current spending discipline.")

        for category, amount in predictions.items():
            if category == 'Entertainment' and amount > budgets.get(category, {}).get('limit', float('inf')):
                recommendations.append("🎬 Tip: Consider free entertainment alternatives or set weekly limits.")
            elif category == 'food & Dining' and amount > budgets.get(category, {}).get('limit', float('inf')):
                recommendations.append("🍽️ Tip: Try meal planning and cooking at home to reduce dining expenses.")

        return recommendations
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return ["Unable to generate recommendations due to system error."]

def compare_models():
    try:
        lstm_model = load_model('best_expense_lstm_model.keras')
        models, scalers, X_input, y_true, is_global_scaler = load_components()
        
        X_train, X_test, y_train, y_test = train_test_split(X_input, y_true, test_size=0.2, random_state=42)
        scaler = list(scalers.values())[0]

        y_pred_lstm_scaled = lstm_model.predict(X_test)
        y_pred_lstm = invert_scaling(y_pred_lstm_scaled, scaler)
        y_test_orig = invert_scaling(y_test, scaler)

        metrics = {}
        model_preds = {}

        for mse, mae, rmse, r2 in [(mean_squared_error, mean_absolute_error, lambda x, y: np.sqrt(mean_squared_error(x, y)), r2_score)]:
            metrics['LSTM'] = {
                'MSE': mse(y_test, y_pred_lstm_scaled),
                'MAE': mae(y_test, y_pred_lstm_scaled),
                'RMSE': rmse(y_test, y_pred_lstm_scaled),
                'R2': r2(y_test, y_pred_lstm_scaled)
            }

        for model_name, layer_type in [('GRU', GRU), ('SimpleRNN', SimpleRNN)]:
            model = Sequential([
                layer_type(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
                Dense(y_train.shape[1])
            ])
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
            y_pred_scaled = model.predict(X_test)
            y_pred = invert_scaling(y_pred_scaled, scaler)
            model_preds[model_name] = y_pred

            metrics[model_name] = {
                'MSE': mean_squared_error(y_test_orig, y_pred),
                'MAE': mean_absolute_error(y_test_orig, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test_orig, y_pred)),
                'R2': r2_score(y_test_orig, y_pred)
            }

        X_train_rf = X_train.reshape(X_train.shape[0], -1)
        X_test_rf = X_test.reshape(X_test.shape[0], -1)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_rf, y_train)
        y_pred_rf = rf_model.predict(X_test_rf)
        model_preds['Random Forest'] = y_pred_rf

        metrics['Random Forest'] = {
            'MSE': mean_squared_error(y_test_orig, y_pred_rf),
            'MAE': mean_absolute_error(y_test_orig, y_pred_rf),
            'RMSE': np.sqrt(mean_squared_error(y_test_orig, y_pred_rf)),
            'R2': r2_score(y_test_orig, y_pred_rf)
        }

        for model_name, model_metrics in metrics.items():
            print(f"{model_name} Metrics:")
            for metric, value in model_metrics.items():
                print(f"  {metric}: {value:.5f}")
            print()

        metrics_df = pd.DataFrame(metrics).T
        metrics_df.plot(kind='bar', figsize=(12, 6), title='Model Performance Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=0)
        plt.show()

        category_index = 0
        for model_name, y_pred in [('LSTM', y_pred_lstm), ('Random Forest', y_pred_rf)]:
            plt.figure(figsize=(12, 5))
            plt.plot(y_test_orig[:, category_index], label='Actual')
            plt.plot(y_pred[:, category_index], label=f'{model_name} Predicted')
            plt.title(f'Actual vs Predicted Expenses - Category Index {category_index} ({model_name})')
            plt.legend()
            plt.show()

    except Exception as e:
        logger.error(f"Error in model comparison: {e}")

def build_model(units, learning_rate, input_shape, output_size):
    model = Sequential()
    model.add(LSTM(units, activation='tanh', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units // 2, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(output_size, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def grid_search(X_train, y_train, X_val, y_val, input_shape, output_size, log_callback=None):
    from keras.callbacks import EarlyStopping

    lstm_unit_options = [32, 64, 128]
    batch_size_options = [4, 8, 16]
    learning_rate_options = [0.001, 0.0005]

    best_val_loss = np.inf
    best_model = None
    best_params = None

    for units in lstm_unit_options:
        for batch_size in batch_size_options:
            for lr in learning_rate_options:
                if log_callback:
                    log_callback(f"🔧 Trying: units={units}, batch_size={batch_size}, lr={lr}")
                # --- DEBUG: Print full y_train for shopping ---
                import json
                if 'shopping' in locals() or 'shopping' in globals():
                    log_callback(f"Full y_train for shopping: {json.dumps(y_train.tolist())}")
                try:
                    log_callback("🏗️ Building model and starting .fit()…")
                    model = build_model(units, lr, input_shape, output_size)
                    log_callback(f"🏃 Calling model.fit(): epochs=1, batch={batch_size}")
                    early_stop = EarlyStopping(patience=1, restore_best_weights=True)
                    live_plot = LiveLossPlotCallback()  # 👈 new callback
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=20,
                        batch_size=batch_size,
                        verbose=0,
                        callbacks=[EarlyStopping(patience=1, restore_best_weights=True)]
                    )
                    if 'loss' in history.history and 'val_loss' in history.history:
                        save_loss_chart(history.history['loss'], history.history['val_loss'])
                                            
                    log_callback("🏁 model.fit() returned")
                    val_loss = min(history.history['val_loss'])
                    if log_callback:
                        log_callback(f"📉 Result: val_loss={val_loss:.6f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params = (units, batch_size, lr)
                        best_model = model
                except Exception as e:
                    if log_callback:
                        log_callback(f"❌ Model error: {str(e)[:50]}...")

    return best_model, best_params

def grid_search_with_timeout(X_train, y_train, X_val, y_val, input_shape, output_size, timeout=60, log_callback=None):
    from keras.callbacks import EarlyStopping
    import numpy as np

    if log_callback:
        log_callback(f"X_train shape: {getattr(X_train, 'shape', None)} dtype: {getattr(X_train, 'dtype', None)}")
        log_callback(f"y_train shape: {getattr(y_train, 'shape', None)} dtype: {getattr(y_train, 'dtype', None)}")
        if hasattr(X_train, 'shape') and X_train.shape[0] > 0:
            log_callback(f"X_train sample: {X_train[0]}")
        if hasattr(y_train, 'shape') and y_train.shape[0] > 0:
            log_callback(f"y_train sample: {y_train[0]}")

    if X_train is None or y_train is None or len(X_train) == 0 or len(y_train) == 0:
        if log_callback:
            log_callback("❌ Training data is empty. Skipping model training.")
        return None, None

    if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)):
        if log_callback:
            log_callback("❌ Training data contains NaN values. Skipping model training.")
        return None, None

    if X_train.shape[1:] != input_shape:
        if log_callback:
            log_callback(f"❌ X_train shape mismatch: {X_train.shape[1:]} vs {input_shape}")
        return None, None

    # Hyperparameters to test
    lstm_unit_options = [32]
    batch_size_options = [4]
    learning_rate_options = [0.001]

    best_val_loss = float('inf')
    best_params = None
    best_model = None

    for units in lstm_unit_options:
        for batch_size in batch_size_options:
            for lr in learning_rate_options:
                if log_callback:
                    log_callback(f"🔧 Trying: units={units}, batch_size={batch_size}, lr={lr}")
                try:
                    # Log and check y_train distribution
                    unique_vals = np.unique(y_train)
                    log_callback(f"🔬 Unique y_train values: {unique_vals}")

                    # Check for near-constant values
                    if np.allclose(y_train, y_train[0], rtol=1e-5, atol=1e-5):
                        log_callback("⚠️ Skipping training - y_train has constant values")
                        return None, None

                    # Check for skewed data with 0s and high outliers
                    if y_train.min() == 0 and np.mean(y_train) > 3000:
                        log_callback("⚠️ Skipping training - y_train contains zero alongside high values")
                        return None, None

                    # Log shape before training
                    log_callback(f"🧪 Training shape: X_train={X_train.shape}, y_train={y_train.shape}, X_val={X_val.shape}, y_val={y_val.shape}")

                    # Build and train model
                    model = build_model(units, lr, input_shape, output_size)
                    early_stop = EarlyStopping(patience=1, restore_best_weights=True)
                    live_plot = LiveLossPlotCallback()  # 👈 new callback
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=20,
                        batch_size=batch_size,
                        verbose=0,
                        callbacks=[EarlyStopping(patience=1, restore_best_weights=True)]
                    )
                    if 'loss' in history.history and 'val_loss' in history.history:
                        save_loss_chart(history.history['loss'], history.history['val_loss'])

                    val_loss = min(history.history['val_loss'])
                    log_callback(f"📉 Result: val_loss={val_loss:.6f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params = (units, batch_size, lr)
                        best_model = model

                except Exception as e:
                    if log_callback:
                        log_callback(f"❌ Model error: {str(e)[:50]}...")

    if best_model is not None:
        return best_model, best_params
    else:
        if log_callback:
            log_callback("⚠️ No valid model found. Skipping model save.")
        return None, None


def train_category_models(log_callback=print):
    try:
        MODEL_DIR = CONFIG['model_dir']
        os.makedirs(MODEL_DIR, exist_ok=True)

        X_all = np.load(CONFIG['X_path'])
        Y_all = np.load(CONFIG['y_path'])
        timesteps = X_all.shape[1]

        for i, cat in enumerate(categories):
            X_cat = X_all[:, :, i].reshape(-1, timesteps, 1)
            y_cat = Y_all[:, i].reshape(-1, 1)

            if np.allclose(y_cat, y_cat[0]):
                log_callback(f"⚠️ Skipping {cat} - y_train values are all the same ({y_cat[0][0]})")
                continue

            X_train, X_val, y_train, y_val = train_test_split(X_cat, y_cat, test_size=0.2, random_state=42)
            input_shape = (timesteps, 1)
            output_size = 1

            best_model, best_params = grid_search_with_timeout(
                X_train, y_train, X_val, y_val,
                input_shape, output_size,
                timeout=60,
                log_callback=log_callback
            )

            if best_model is not None:
                model_path = os.path.join(MODEL_DIR, f"{cat}_lstm.keras")
                best_model.save(model_path)
                log_callback(f"✅ Saved model for {cat}: Units={best_params[0]}, Batch={best_params[1]}, LR={best_params[2]}")
            else:
                log_callback(f"❌ {cat} model was not trained (skipped due to poor training or constant data)")

    except Exception as e:
        logger.error(f"Error training category models: {e}")

def _fit_model_worker(X_train, y_train, X_val, y_val, input_shape, output_size, units, lr, batch_size, epochs, return_dict):
    from keras.callbacks import EarlyStopping
    from smartfinance import build_model
    try:
        model = build_model(units, lr, input_shape, output_size)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[EarlyStopping(patience=1, restore_best_weights=True)]
        )
        val_loss = min(history.history['val_loss'])
        return_dict['val_loss'] = val_loss
        return_dict['success'] = True
    except Exception as e:
        return_dict['error'] = str(e)
        return_dict['success'] = False

# Add this function to your smartfinance.py file

def load_user_budget(username):
    """Load user-specific budget configuration"""
    try:
        user_budget_file = f"user_budgets/{username}_budget.json"
        if os.path.exists(user_budget_file):
            with open(user_budget_file, 'r') as f:
                return json.load(f)
        
        # Fallback to user data file
        from project_utilities import load_user_data
        user_data = load_user_data(username)
        if 'budget' in user_data:
            return user_data['budget']
        
        return default_budget_limits
    except Exception as e:
        logger.warning(f"Error loading user budget: {e}")
        return default_budget_limits

# Update this function in smartfinance.py
def process_user_transactions(transactions):
    """Process user transactions into monthly spending data by category - Updated to handle income"""
    try:
        if not transactions:
            raise ValueError("No transaction data provided")
        
        df = pd.DataFrame(transactions)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df = df.dropna()
        
        # Separate expenses (positive amounts) from income (negative amounts)
        expenses_df = df[df['Amount'] >= 0].copy()
        
        if len(expenses_df) == 0:
            logger.warning("No expense transactions found")
            return np.zeros((1, len(categories)))
        
        # Group by year-month and category for expenses only
        expenses_df['YearMonth'] = expenses_df['Date'].dt.to_period('M')
        monthly_spending = expenses_df.groupby(['YearMonth', 'Category'])['Amount'].sum().unstack(fill_value=0)
        
        # Ensure all categories are present
        for category in categories:
            if category not in monthly_spending.columns:
                monthly_spending[category] = 0
        
        # Reorder columns to match categories list
        monthly_spending = monthly_spending[categories]
        
        logger.info(f"Processed {len(expenses_df)} expense transactions into {len(monthly_spending)} monthly records")
        return monthly_spending.values
        
    except Exception as e:
        logger.error(f"Error processing user transactions: {e}")
        return np.zeros((6, len(categories)))  # fallback safe dummy array


# Add these updated categories to include Income
categories_with_income = categories + ['Income']


def save_loss_chart(train_loss, val_loss, save_path='static/loss_chart.png'):
    os.makedirs('static', exist_ok=True)

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_loss) + 1)

    plt.plot(epochs, train_loss, label='Training Loss', color='blue', linewidth=2)
    plt.plot(epochs, val_loss, label='Validation Loss', color='red', linestyle='--', linewidth=2)

    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()







class LiveLossPlotCallback(callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')

        if train_loss is not None and val_loss is not None:
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            save_loss_chart(self.train_losses, self.val_losses)