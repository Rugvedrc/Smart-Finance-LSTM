# Smart-Finance-LSTM

A Flask-based personal finance web application that uses **LSTM (Long Short-Term Memory)** neural networks to predict future monthly expenses by category, detect spending anomalies, and provide personalised financial recommendations.

---

## Features

- **Expense Prediction** — Per-category next-month forecasts powered by trained LSTM models with trend analysis and seasonal adjustments.
- **Anomaly Detection** — Identifies unusual spending patterns by comparing predictions against recent history using z-score analysis.
- **Financial Health Score** — Composite score based on predicted spending vs. budget limits and prediction confidence.
- **Budget Management** — Set custom monthly budget limits and volatility thresholds for each category.
- **Recommendations** — Actionable suggestions generated from predictions, budget alerts, and anomaly signals.
- **Transaction Management** — Add individual transactions, bulk-upload CSV files (with duplicate merging), and delete entries.
- **Multi-user Support** — Each user's transactions and budget are stored in separate JSON files under `userdata/`.
- **Demo Mode** — Instantly explore the app using pre-loaded sample expense data without creating an account.
- **Model Comparison** — Training pipeline compares LSTM, GRU, SimpleRNN, and Random Forest models per category and saves the best performer.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | Flask 3.x + Jinja2 |
| Deep learning | TensorFlow / Keras 2.13 (LSTM, GRU, SimpleRNN) |
| Machine learning | scikit-learn (MinMaxScaler, RandomForestRegressor) |
| Data processing | pandas, NumPy |
| Visualisation | Matplotlib, Plotly, Seaborn |
| Session storage | Flask-Session |
| Serialisation | joblib (scalers), Keras `.keras` (models) |

---

## Expense Categories

The application tracks spending across **7 categories**:

- Bills & Utilities
- Education
- Entertainment
- food & Dining
- health & Medical
- shopping
- Travel & Transportation

---

## Project Structure

```
Smart-Finance-LSTM/
├── app.py                      # Flask application & all route handlers
├── smartfinance.py             # Core ML logic: models, predictions, anomaly detection, scoring
├── train.py                    # Training pipeline entry point
├── project_utilities.py        # Config, file storage, CSV validator, training logger
├── requirements.txt            # Python dependencies
│
├── best_expense_lstm_model.keras  # Pre-trained global LSTM model
├── category_models/            # Per-category trained model files (*_lstm.keras)
├── category_scalers/           # Per-category MinMaxScaler files (*_scaler.pkl)
│
├── realistic_expense_data.csv  # Full training dataset
├── sample_expense_data.csv     # Demo / quick-start dataset
│
├── templates/                  # Jinja2 HTML templates
│   ├── landing.html
│   ├── login.html
│   ├── dashboard.html
│   ├── add_transaction.html
│   ├── view_transactions.html
│   ├── set_budget.html
│   ├── results.html
│   ├── training.html
│   ├── loading.html
│   └── developer.html
└── static/
    └── loss_chart.png          # Training loss chart (generated during training)
```

---

## Installation

### Prerequisites

- Python 3.9 or later
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Rugvedrc/Smart-Finance-LSTM.git
cd Smart-Finance-LSTM

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Train the Models (optional — pre-trained models are included)

```bash
python train.py
```

This will:
1. Load and validate `realistic_expense_data.csv`.
2. Aggregate transactions into monthly per-category sequences.
3. Train LSTM, GRU, SimpleRNN, and Random Forest models for each category.
4. Save the best model per category to `category_models/` and scalers to `category_scalers/`.

### 2. Run the Web Application

```bash
python app.py
```

Open your browser at `http://127.0.0.1:5000`.

### 3. Quick Start with Demo Mode

On the login page, toggle **Demo Mode** to load pre-built sample data and explore all features without any setup.

---

## CSV Upload Format

To import your own transaction history, prepare a CSV file with the following columns:

| Column | Type | Example |
|---|---|---|
| `Date` | Date string (YYYY-MM-DD) | `2024-03-15` |
| `Category` | One of the 7 categories | `food & Dining` |
| `Amount` | Numeric | `850.00` |

Duplicate transactions (same Date + Category + Amount) are automatically skipped during import.

---

## How Predictions Work

1. **Data Aggregation** — Raw transactions are grouped into monthly spending totals per category.
2. **Sequence Creation** — A sliding window of 3 consecutive months forms each input sequence.
3. **Scaling** — Each category's values are normalised with its own `MinMaxScaler`.
4. **LSTM Inference** — The trained model predicts spending for the next month.
5. **Trend Adjustment** — A weighted exponential average and recent growth rate are applied on top of the model output.
6. **Seasonal Adjustment** — Hard-coded seasonal multipliers are applied for categories like Entertainment, Travel, and Shopping.
7. **Confidence Score** — Derived from the coefficient of variation of recent months; more consistent spending yields higher confidence.

---

## Application Routes

| Route | Description |
|---|---|
| `/` | Landing page |
| `/login` | Login / demo mode entry |
| `/dashboard` | Spending summary by category |
| `/add_transaction` | Add a single transaction |
| `/upload_csv` | Bulk import transactions from CSV |
| `/view_transactions` | List, filter, and delete transactions |
| `/set_budget` | Set per-category budget limits |
| `/predict` | Run prediction and view full analysis |
| `/predict_loading` | Loading screen before prediction |
| `/predict_process` | Internal prediction processing endpoint |
| `/load_demo` | Load sample demo data |
| `/developer` | Developer / model info page |

---

## Configuration

Key settings are defined in `project_utilities.py` and `smartfinance.py` under the `CONFIG` dictionary:

| Key | Default | Description |
|---|---|---|
| `model_dir` | `category_models` | Directory for saved per-category models |
| `scaler_dir` | `category_scalers` | Directory for saved per-category scalers |
| `prediction_confidence_threshold` | `0.7` | Minimum confidence to trust a prediction |
| `lstm_anomaly_multiplier` | `2.0` | Standard-deviation multiplier for anomaly threshold |
| `min_historical_months` | `3` | Minimum months of data required for prediction |

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in the repository.

