from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
import threading
import time
import signal
import logging
import numpy as np
from datetime import datetime
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt
import os
from datetime import date as dt_date
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
# Fix: Only import joblib, keras, pandas, matplotlib if available
try:
    import joblib
except ImportError:
    joblib = None
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None
try:
    from keras.models import load_model, Sequential
except ImportError:
    load_model = None
    Sequential = None
try:
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    MinMaxScaler = None

from smartfinance import default_budget_limits
from project_utilities import load_user_data, save_user_data, validate_expense_csv, safe_username
from smartfinance import (
    load_components, 
    lstm_anomaly_detection, financial_health_score, check_budget, 
    generate_recommendations, load_budget_config, categories, 
    train_category_models, compare_models, CONFIG,
    generate_user_predictions,load_user_budget 
)
# Global variable to track training status
training_status = {
    'status': 'idle',
    'current_category': 0,
    'message': '',
    'log': [],
    'results': [],
    'loss_data': {},
    'error': None
}
app = Flask(__name__)
app.secret_key = 'abcdef1234567890'  # Change this to a secure key in production

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        demo_mode = request.form.get('demo_mode') == 'on'
        
        if demo_mode:
            session['username'] = 'demo'
            session['demo_mode'] = True
            return redirect(url_for('dashboard'))
        
        if username:
            session['username'] = safe_username(username)
            session['demo_mode'] = False
            return redirect(url_for('dashboard'))
        
        flash('Please enter a username')
    
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    username = session['username']
    user_data = load_user_data(username)
    
    summary = {}
    if user_data.get('transactions'):
        df = pd.DataFrame(user_data['transactions'])
        for category in categories:
            cat_data = df[df['Category'] == category]
            summary[category] = {
                'count': len(cat_data),
                'total': cat_data['Amount'].sum() if len(cat_data) > 0 else 0,
                'avg': cat_data['Amount'].mean() if len(cat_data) > 0 else 0
            }
    
    return render_template('dashboard.html', summary=summary, demo_mode=session.get('demo_mode', False))

# Add these routes to your app.py file

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'username' not in session or session.get('demo_mode'):
        flash('Upload not available in demo mode')
        return redirect(url_for('dashboard'))
    
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('dashboard'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('dashboard'))
    
    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)
            is_valid, message = validate_expense_csv(df)
            
            if not is_valid:
                flash(f'Invalid CSV: {message}')
                return redirect(url_for('dashboard'))
            
            df['Date'] = pd.to_datetime(df['Date'])
            df['Date'] = df['Date'].astype(str)
            df['Amount'] = df['Amount'].astype(float)
            new_transactions = df.to_dict('records')

            username = session['username']
            user_data = load_user_data(username)
            
            # MERGE WITH EXISTING DATA instead of replacing
            existing_transactions = user_data.get('transactions', [])
            
            # Convert existing transactions to set of tuples for duplicate checking
            existing_set = set()
            for t in existing_transactions:
                existing_set.add((t['Date'], t['Category'], float(t['Amount'])))
            
            # Add only new transactions (avoid duplicates)
            merged_transactions = existing_transactions.copy()
            new_count = 0
            
            for new_trans in new_transactions:
                trans_tuple = (new_trans['Date'], new_trans['Category'], float(new_trans['Amount']))
                if trans_tuple not in existing_set:
                    merged_transactions.append(new_trans)
                    new_count += 1
            
            # Sort by date
            merged_transactions.sort(key=lambda x: x['Date'])
            
            user_data['transactions'] = merged_transactions
            user_data['last_updated'] = datetime.now().isoformat()
            save_user_data(username, user_data)
            
            flash(f'Successfully merged {new_count} new transactions (total: {len(merged_transactions)})')
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
    
    return redirect(url_for('dashboard'))
# Add these routes to your app.py file

@app.route('/add_transaction', methods=['GET', 'POST'])
def add_transaction():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            date = request.form.get('date')
            category = request.form.get('category')
            amount = float(request.form.get('amount'))
            transaction_type = request.form.get('type', 'expense')  # expense or income
            description = request.form.get('description', '')
            
            if not date or not category or amount <= 0:
                flash('Please fill all required fields with valid values')
                return redirect(url_for('add_transaction'))
            
            # For income, use negative amount to distinguish from expenses
            if transaction_type == 'income':
                amount = -abs(amount)  # Make it negative for income
                category = 'Income'  # Use special Income category
            
            new_transaction = {
                'Date': date,
                'Category': category,
                'Amount': amount,
                'Description': description,
                'Type': transaction_type
            }
            
            username = session['username']
            user_data = load_user_data(username)
            existing_transactions = user_data.get('transactions', [])
            existing_transactions.append(new_transaction)
            
            # Sort by date
            existing_transactions.sort(key=lambda x: x['Date'])
            
            user_data['transactions'] = existing_transactions
            user_data['last_updated'] = datetime.now().isoformat()
            save_user_data(username, user_data)
            
            flash(f'Successfully added {transaction_type}: ₹{abs(amount)} in {category}')
            return redirect(url_for('dashboard'))
            
        except ValueError:
            flash('Invalid amount entered')
        except Exception as e:
            flash(f'Error adding transaction: {str(e)}')
    
    return render_template('add_transaction.html', categories=categories, today=dt_date.today().strftime('%Y-%m-%d'))

@app.route('/set_budget', methods=['GET', 'POST'])
def set_budget():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            username = session['username']
            user_data = load_user_data(username)
            
            # Get budget data from form
            budget_data = {}
            for category in categories:
                limit = request.form.get(f'budget_{category}')
                volatility = request.form.get(f'volatility_{category}', '0.3')
                
                if limit:
                    budget_data[category] = {
                        'limit': float(limit),
                        'volatility': float(volatility)
                    }
            
            # Save budget to user data
            user_data['budget'] = budget_data
            user_data['budget_updated'] = datetime.now().isoformat()
            save_user_data(username, user_data)
            
            # Also save to global budget config for the user
            budget_file = f"user_budgets/{username}_budget.json"
            os.makedirs('user_budgets', exist_ok=True)
            with open(budget_file, 'w') as f:
                json.dump(budget_data, f, indent=2)
            
            flash('Budget settings saved successfully!')
            return redirect(url_for('dashboard'))
            
        except ValueError:
            flash('Please enter valid numeric values for budget limits')
        except Exception as e:
            flash(f'Error saving budget: {str(e)}')
    
    # Load current budget settings
    username = session['username']
    user_data = load_user_data(username)
    current_budget = user_data.get('budget', default_budget_limits)
    
    return render_template('set_budget.html', categories=categories, current_budget=current_budget)

@app.route('/view_transactions')
def view_transactions():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    username = session['username']
    user_data = load_user_data(username)
    transactions = user_data.get('transactions', [])
    
    # Sort by date (newest first)
    transactions.sort(key=lambda x: x['Date'], reverse=True)
    
    # Separate income and expenses
    expenses = [t for t in transactions if t.get('Amount', 0) >= 0]
    income = [t for t in transactions if t.get('Amount', 0) < 0]
    
    # Calculate totals
    total_expenses = sum(t['Amount'] for t in expenses)
    total_income = sum(abs(t['Amount']) for t in income)
    net_balance = total_income - total_expenses
    
    return render_template('view_transactions.html', 
                         transactions=transactions,
                         expenses=expenses,
                         income=income,
                         total_expenses=total_expenses,
                         total_income=total_income,
                         net_balance=net_balance)

@app.route('/delete_transaction/<int:index>', methods=['POST'])
def delete_transaction(index):
    if 'username' not in session or session.get('demo_mode'):
        flash('Delete not available in demo mode')
        return redirect(url_for('view_transactions'))
    
    try:
        username = session['username']
        user_data = load_user_data(username)
        transactions = user_data.get('transactions', [])
        
        if 0 <= index < len(transactions):
            deleted_trans = transactions.pop(index)
            user_data['transactions'] = transactions
            user_data['last_updated'] = datetime.now().isoformat()
            save_user_data(username, user_data)
            flash(f'Deleted transaction: ₹{abs(deleted_trans["Amount"])} from {deleted_trans["Category"]}')
        else:
            flash('Invalid transaction index')
            
    except Exception as e:
        flash(f'Error deleting transaction: {str(e)}')
    
    return redirect(url_for('view_transactions'))
@app.route('/load_demo')
def load_demo():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    try:
        demo_path = 'sample_expense_data.csv'
        if not os.path.exists(demo_path):
            demo_data = []
            dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
            for date in dates[:100]:
                for category in categories[:3]:
                    amount = np.random.uniform(100, 1000)
                    demo_data.append({
                        'Date': date.strftime('%Y-%m-%d'),
                        'Category': category,
                        'Amount': round(amount, 2)
                    })
            
            os.makedirs('demo', exist_ok=True)
            pd.DataFrame(demo_data).to_csv(demo_path, index=False)
        
        df = pd.read_csv(demo_path)
        transactions = df.to_dict('records')
        
        username = session['username']
        user_data = load_user_data(username)
        user_data['transactions'] = transactions
        user_data['last_updated'] = datetime.now().isoformat()
        save_user_data(username, user_data)
        
        flash(f'Loaded {len(transactions)} demo transactions')
        
    except Exception as e:
        flash(f'Error loading demo data: {str(e)}')
    
    return redirect(url_for('dashboard'))

# Add this route to your app.py
@app.route('/predict_loading')
def predict_loading():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('loading.html')

@app.route('/predict_process')
def predict_process():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    try:
        models, scalers, X_input, y_true, is_global_scaler = load_components()

        username = session['username']
        user_data = load_user_data(username)
        user_transactions = user_data.get('transactions', [])

        predictions, confidences = generate_user_predictions(models, scalers, user_transactions, is_global_scaler)


        df = pd.DataFrame(user_transactions)
        df['Date'] = pd.to_datetime(df['Date'])
        df['YearMonth'] = df['Date'].dt.to_period('M')
        monthly_spending = df.groupby(['YearMonth', 'Category'])['Amount'].sum().unstack(fill_value=0)

        for category in categories:
            if category not in monthly_spending.columns:
                monthly_spending[category] = 0

        monthly_spending = monthly_spending[categories]
        y_last_month = monthly_spending.iloc[-1].values if len(monthly_spending) > 0 else np.zeros(len(categories))

        
        lstm_anomalies = lstm_anomaly_detection(y_true, predictions, confidences, scalers, is_global_scaler)
        budget_limits = load_budget_config()
        budget_alerts = check_budget(predictions, budget_limits)
        health_score = financial_health_score(predictions, budget_limits, y_last_month, confidences)
        recommendations = generate_recommendations(predictions, budget_limits, lstm_anomalies, budget_alerts, health_score)
        
        results = {
            'predictions': predictions,
            'confidences': confidences,
            'lstm_anomalies': lstm_anomaly_detection(y_true, predictions, confidences, scalers, is_global_scaler),
            'budget_alerts': budget_alerts,
            'health_score': health_score,
            'recommendations': recommendations,
            'total_predicted': sum(predictions.values())
        }
        
        return render_template('results.html', results=results)
        
    except Exception as e:
        flash(f'Prediction error: {str(e)}')
        return redirect(url_for('dashboard'))

@app.route('/predict')
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    try:
        # Load user's transaction data
        username = session['username']
        user_data = load_user_data(username)
        user_transactions = user_data.get('transactions', [])
        
        if not user_transactions:
            flash('No transaction data found. Please upload your expense data first.')
            return redirect(url_for('dashboard'))
        
        # Load pre-trained models and scalers (for fallback only)
        models, scalers, _, _, is_global_scaler = load_components()
        
        # Generate predictions based on user's actual data
        predictions, confidences = generate_user_predictions(models, scalers, user_transactions, is_global_scaler)
        
        # Process user data for analysis
        monthly_data_df = pd.DataFrame(user_transactions)
        monthly_data_df['Date'] = pd.to_datetime(monthly_data_df['Date'])
        monthly_data_df['YearMonth'] = monthly_data_df['Date'].dt.to_period('M')
        
        # Get monthly spending by category
        monthly_spending = monthly_data_df.groupby(['YearMonth', 'Category'])['Amount'].sum().unstack(fill_value=0)
        
        # Ensure all categories are present
        for category in categories:
            if category not in monthly_spending.columns:
                monthly_spending[category] = 0
        
        monthly_spending = monthly_spending[categories]
        
        # Get last month's data for comparison
        if len(monthly_spending) > 0:
            y_last_month = monthly_spending.iloc[-1].values
        else:
            y_last_month = np.zeros(len(categories))
        
        # Generate user-specific analysis
        budget_limits = load_budget_config()
        budget_alerts = check_budget(predictions, budget_limits)
        health_score = financial_health_score(predictions, budget_limits, y_last_month, confidences)
        
        # Enhanced anomaly detection based on user's data
        lstm_anomalies = []
        if len(monthly_spending) >= 2:
            for i, category in enumerate(categories):
                if category in monthly_spending.columns:
                    recent_spending = monthly_spending[category].iloc[-3:].values  # Last 3 months
                    predicted = predictions[category]
                    
                    if len(recent_spending) >= 2 and np.mean(recent_spending) > 0:
                        avg_recent = np.mean(recent_spending)
                        std_recent = np.std(recent_spending)
                        
                        # Check if prediction is significantly different from recent pattern
                        if std_recent > 0:
                            z_score = abs(predicted - avg_recent) / std_recent
                            if z_score > 2:  # Significant deviation
                                change_pct = ((predicted - avg_recent) / avg_recent) * 100
                                severity = "HIGH" if z_score > 3 else "MEDIUM"
                                direction = "increase" if predicted > avg_recent else "decrease"
                                
                                alert_text = f"""🚨 {severity} Spending Pattern Change in "{category}":
Predicted: ₹{predicted:.2f} | Recent Average: ₹{avg_recent:.2f} | Change: {change_pct:+.2f}%
Based on your last {len(recent_spending)} months of data, this represents a significant {direction}."""
                                
                                lstm_anomalies.append({
                                    'category': category,
                                    'alert': alert_text,
                                    'severity': severity,
                                    'anomaly_score': z_score,
                                    'confidence': confidences[category]
                                })
        
        recommendations = generate_recommendations(predictions, budget_limits, lstm_anomalies, budget_alerts, health_score)
        
        results = {
            'predictions': predictions,
            'confidences': confidences,
            'lstm_anomalies': lstm_anomaly_detection(y_true, predictions, confidences, scalers, is_global_scaler),
            'budget_alerts': budget_alerts,
            'health_score': health_score,
            'recommendations': recommendations,
            'total_predicted': sum(predictions.values())
        }
        
        return render_template('results.html', results=results)
        
    except Exception as e:
        flash(f'Prediction error: {str(e)}')
        return redirect(url_for('dashboard'))

@app.route('/developer')
def developer():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    return render_template('developer.html')

@app.route('/train_models', methods=['POST'])
def train_models():
    if 'username' not in session or session.get('demo_mode'):
        return jsonify({'error': 'Training not available in demo mode'})
    
    try:
        train_category_models()
        return jsonify({'success': 'Models trained successfully'})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/clear_data', methods=['POST'])
def clear_data():
    if 'username' not in session or session.get('demo_mode'):
        flash('Clear not available in demo mode')
        return redirect(url_for('dashboard'))
    
    username = session['username']
    save_user_data(username, {})
    flash('Data cleared successfully')
    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully')
    return redirect(url_for('landing'))

import threading
import time
from datetime import datetime

# Add thread lock at the top with other global variables
training_lock = threading.Lock()

@app.route('/train_models_advanced', methods=['POST'])
def train_models_advanced():
    if 'username' not in session or session.get('demo_mode'):
        return jsonify({'error': 'Training not available in demo mode'})
    
    global training_status
    
    # Prevent multiple simultaneous training requests
    with training_lock:
        if training_status['status'] == 'training':
            return jsonify({'error': 'Training already in progress. Please wait for completion.'})
        
        # Reset training status
        training_status = {
            'status': 'training',
            'current_category': 0,
            'message': 'Initializing training...',
            'log': [],
            'results': [],
            'loss_data': {},
            'error': None,
            'start_time': time.time()
        }
    
    try:
        params = request.json
        username = session['username']
        user_data = load_user_data(username)
        
        if not user_data.get('transactions'):
            training_status['status'] = 'error'
            training_status['error'] = 'No transaction data found'
            return jsonify({'error': 'No transaction data found. Please upload data first.'})
        
        # Start training in background thread with timeout
        thread = threading.Thread(target=background_training_with_timeout, args=(params, user_data['transactions']))
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True, 'message': 'Training started successfully'})
        
    except Exception as e:
        training_status['status'] = 'error'
        training_status['error'] = str(e)
        return jsonify({'error': str(e)})

@app.route('/training_status')
def get_training_status():
    return jsonify(training_status)

def add_training_log(message, type='info'):
    global training_status
    timestamp = datetime.now().strftime("%H:%M:%S")
    training_status['log'].append({
        'message': f"[{timestamp}] {message}",
        'type': type
    })
    # Keep only last 50 log entries
    if len(training_status['log']) > 50:
        training_status['log'] = training_status['log'][-50:]

def background_training_with_timeout(params, user_transactions):
    global training_status
    import time
    TRAINING_TIMEOUT = 120  # 2 minutes timeout for safety
    start_time = time.time()
    
    try:
        from smartfinance import (
            categories, process_user_transactions, 
            build_model, grid_search_with_timeout, CONFIG
        )
        import os
        os.makedirs(CONFIG['model_dir'], exist_ok=True)
        os.makedirs(CONFIG['scaler_dir'], exist_ok=True)

        add_training_log("🚀 Starting advanced model training with timeout protection...")
        add_training_log(f"Mode: {params['mode']} | Timeout: {TRAINING_TIMEOUT}s")
        training_status['message'] = 'Processing user transaction data...'

        # Prepare monthly dataset
        monthly_data = process_user_transactions(user_transactions)

        if time.time() - start_time > TRAINING_TIMEOUT:
            add_training_log("⏰ Training timeout after data processing", 'error')
            training_status['status'] = 'timeout'
            return

        if len(monthly_data) < 6:
            add_training_log("❌ Not enough data for training", 'error')
            training_status['status'] = 'error'
            return

        add_training_log(f"Processed {len(monthly_data)} months of data")
        sequence_length = 3
        results = []

        for i, category in enumerate(categories):
            if time.time() - start_time > TRAINING_TIMEOUT:
                add_training_log(f"⏰ Training timeout at category {category}", 'error')
                training_status['status'] = 'timeout'
                break

            add_training_log(f"🎯 Training {category} model...")

            try:
                # Prepare sequences for LSTM
                X = []
                y = []
                for j in range(len(monthly_data) - sequence_length):
                    X.append(monthly_data[j:j + sequence_length])
                    y.append(monthly_data[j + sequence_length][i])
                X = np.array(X)
                y = np.array(y).reshape(-1, 1)

                if len(X) == 0:
                    add_training_log(f"⚠️ Not enough data for {category}", 'warn')
                    results.append({'category': category, 'status': 'Skipped'})
                    continue

                # Train/Val split
                split = int(0.8 * len(X))
                X_train, X_val = X[:split], X[split:]
                y_train, y_val = y[:split], y[split:]

                input_shape = (sequence_length, len(categories))
                output_size = 1

                model, params = grid_search_with_timeout(
                    X_train, y_train, X_val, y_val,
                    input_shape, output_size,
                    timeout=60,
                    log_callback=add_training_log
                )

                if model is not None:
                    model_path = os.path.join(CONFIG['model_dir'], f"{category}_lstm.keras")
                    model.save(model_path)
                    add_training_log(f"✅ {category} model trained and saved.")
                    results.append({
                        'category': category,
                        'status': 'Completed',
                        'params': params
                    })
                else:
                    add_training_log(f"❌ {category} model was not trained (skipped due to constant data or poor training).")
                    results.append({'category': category, 'status': 'Skipped'})

            except Exception as e:
                add_training_log(f"❌ {category} model error: {str(e)[:50]}", 'error')
                results.append({'category': category, 'status': 'Error', 'error': str(e)})

        with training_lock:
            training_status['status'] = 'done'
            training_status['results'] = results

        completed_count = sum(1 for r in results if r['status'] == 'Completed')
        add_training_log(f"🎉 Training finished! {completed_count}/{len(categories)} models trained")

    except TimeoutError as e:
        with training_lock:
            training_status['status'] = 'timeout'
        add_training_log(f"⏰ Training timeout: {str(e)}", 'error')

    except Exception as e:
        with training_lock:
            training_status['status'] = 'error'
        add_training_log(f"💥 Training failed: {str(e)}", 'error')


@app.route('/train_models_page')
def train_models_page():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('training.html')


@app.route('/compare_models')
def compare_models():
    try:
        # Example dummy data — replace this with real model comparison logic
        data = {
            'success': True,
            'accuracy_data': {
                'labels': ['Food', 'Transport', 'Entertainment', 'Shopping', 'Bills'],
                'current': [0.89, 0.92, 0.87, 0.91, 0.88],
                'previous': [0.85, 0.88, 0.83, 0.87, 0.84]
            },
            'loss_data': {
                'labels': ['Food', 'Transport', 'Entertainment', 'Shopping', 'Bills'],
                'current': [0.12, 0.08, 0.15, 0.09, 0.13],
                'previous': [0.18, 0.15, 0.22, 0.16, 0.19]
            },
            'metrics': {
                'overall_accuracy': { 'current': 89.6, 'previous': 85.4 },
                'avg_loss': { 'current': 0.114, 'previous': 0.176 },
                'training_time': { 'current': 45, 'previous': 52 },
                'model_size': { 'current': 2.3, 'previous': 2.1 }
            }
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)