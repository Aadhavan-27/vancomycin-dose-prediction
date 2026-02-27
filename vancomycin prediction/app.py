"""
VancoPred - Vancomycin Trough Concentration Predictor
Flask Backend — integrates CAPSTONE_FINAL_1.ipynb (RF + XGBoost)
               and CAPSTONE_FINAL_2.ipynb (RF + Traditional PK comparison)

Run:
    pip install flask scikit-learn xgboost numpy pandas
    python app.py
Then open http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import os

warnings.filterwarnings('ignore')

# ── Try importing XGBoost (optional — falls back to RF-only if missing) ──────
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not installed. Running RF-only mode.")
    print("    Install with: pip install xgboost")

np.random.seed(42)

app = Flask(__name__)

# =============================================================================
# 1. DATA GENERATION  (exact copy from both notebooks)
# =============================================================================
def generate_synthetic_vancomycin_data(n_samples=500):
    data = {
        'age':                  np.random.randint(18, 90, n_samples),
        'weight':               np.random.normal(70, 15, n_samples),
        'height':               np.random.normal(170, 10, n_samples),
        'gender':               np.random.choice([0, 1], n_samples),  # 0=Female, 1=Male
        'serum_creatinine':     np.random.uniform(0.5, 3.0, n_samples),
        'dose':                 np.random.choice([500, 750, 1000, 1250, 1500], n_samples),
        'dosing_interval':      np.random.choice([8, 12, 24], n_samples),
        'duration_of_infusion': np.random.choice([1, 1.5, 2], n_samples),
        'time_to_trough':       np.random.uniform(0.5, 2, n_samples),
    }
    df = pd.DataFrame(data)

    # Derived features (same as notebooks)
    df['bmi']        = df['weight'] / ((df['height'] / 100) ** 2)
    df['daily_dose'] = (df['dose'] * 24) / df['dosing_interval']

    # Cockcroft-Gault CrCl
    df['crcl'] = ((140 - df['age']) * df['weight']) / (72 * df['serum_creatinine'])
    df.loc[df['gender'] == 0, 'crcl'] *= 0.85  # female adjustment

    # Synthetic trough  (exact formula from notebooks)
    trough = (
        5
        + (df['dose'] / 100) * 0.8
        - (df['weight'] / 10) * 0.3
        - (df['crcl'] / 10) * 0.4
        + (df['age'] / 10) * 0.2
        + np.random.normal(0, 2, n_samples)
    )
    df['trough_concentration'] = np.clip(trough, 3, 35)
    return df

# =============================================================================
# 2. FEATURE COLUMNS (same order as notebooks — order MUST match scaler)
# =============================================================================
FEATURE_COLUMNS = [
    'age', 'weight', 'height', 'gender', 'serum_creatinine',
    'dose', 'dosing_interval', 'duration_of_infusion',
    'time_to_trough', 'bmi', 'daily_dose', 'crcl'
]

# =============================================================================
# 3. TRAIN MODELS  (runs once at startup)
# =============================================================================
print("=" * 60)
print("  VancoPred — Training Models on Startup")
print("=" * 60)

df = generate_synthetic_vancomycin_data(500)
X  = df[FEATURE_COLUMNS]
y  = df['trough_concentration']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Random Forest  (from CAPSTONE_FINAL_1 + CAPSTONE_FINAL_2) ────────────────
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)           # RF uses raw (unscaled) features per notebook
y_pred_rf = rf.predict(X_test)

rf_metrics = {
    'train_r2':   round(r2_score(y_train, rf.predict(X_train)), 4),
    'test_r2':    round(r2_score(y_test,  y_pred_rf), 4),
    'test_rmse':  round(float(np.sqrt(mean_squared_error(y_test, y_pred_rf))), 4),
    'test_mae':   round(float(mean_absolute_error(y_test, y_pred_rf)), 4),
}
print(f"\n✅ Random Forest   → Test R²: {rf_metrics['test_r2']}  MAE: {rf_metrics['test_mae']}")

# ── XGBoost  (from CAPSTONE_FINAL_1) ─────────────────────────────────────────
xgb_model  = None
xgb_metrics = {}
if XGBOOST_AVAILABLE:
    xgb_model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    xgb_metrics = {
        'train_r2':  round(r2_score(y_train, xgb_model.predict(X_train)), 4),
        'test_r2':   round(r2_score(y_test,  y_pred_xgb), 4),
        'test_rmse': round(float(np.sqrt(mean_squared_error(y_test, y_pred_xgb))), 4),
        'test_mae':  round(float(mean_absolute_error(y_test, y_pred_xgb)), 4),
    }
    print(f"✅ XGBoost         → Test R²: {xgb_metrics['test_r2']}  MAE: {xgb_metrics['test_mae']}")
else:
    print("⚠️  XGBoost skipped — using RF-only ensemble")

# ── Traditional PK Baseline  (from CAPSTONE_FINAL_2) ─────────────────────────
y_pred_pk = (X_test['dose'] * 0.015) - (X_test['crcl'] * 0.03) + 12
pk_metrics = {
    'test_r2':   round(r2_score(y_test, y_pred_pk), 4),
    'test_mae':  round(float(mean_absolute_error(y_test, y_pred_pk)), 4),
}
print(f"📐 Traditional PK  → Test R²: {pk_metrics['test_r2']}  MAE: {pk_metrics['test_mae']}")

# ── Feature importance ────────────────────────────────────────────────────────
rf_importance = [
    {'feature': f, 'importance': round(float(imp), 4)}
    for f, imp in sorted(
        zip(FEATURE_COLUMNS, rf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
]

print("\n✅ Models ready. Starting server...\n")

# =============================================================================
# 4. HELPER — build patient DataFrame from request values
# =============================================================================
def build_patient_df(data):
    age    = float(data['age'])
    weight = float(data['weight'])
    height = float(data['height'])
    gender = float(data['gender'])
    scr    = float(data['serum_creatinine'])
    dose   = float(data['dose'])
    interval = float(data['dosing_interval'])
    duration  = float(data['duration_of_infusion'])
    ttt       = float(data['time_to_trough'])

    # Derived (same as notebook)
    bmi        = weight / ((height / 100) ** 2)
    daily_dose = (dose * 24) / interval
    crcl       = ((140 - age) * weight) / (72 * scr)
    if gender == 0:
        crcl *= 0.85

    return pd.DataFrame([{
        'age': age, 'weight': weight, 'height': height,
        'gender': gender, 'serum_creatinine': scr,
        'dose': dose, 'dosing_interval': interval,
        'duration_of_infusion': duration, 'time_to_trough': ttt,
        'bmi': bmi, 'daily_dose': daily_dose, 'crcl': crcl,
    }])[FEATURE_COLUMNS], {
        'bmi': round(bmi, 2),
        'daily_dose': round(daily_dose, 1),
        'crcl': round(crcl, 1),
        'dose_per_kg': round(dose / weight, 1),
    }

# =============================================================================
# 5. ROUTES
# =============================================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        patient_df, derived = build_patient_df(data)

        # ── Model predictions ──
        pred_rf = float(rf.predict(patient_df)[0])

        if xgb_model is not None:
            pred_xgb = float(xgb_model.predict(patient_df)[0])
            ensemble  = (pred_rf + pred_xgb) / 2
        else:
            pred_xgb = None
            ensemble  = pred_rf

        # ── Traditional PK baseline ──
        crcl = derived['crcl']
        dose = float(data['dose'])
        pred_pk = (dose * 0.015) - (crcl * 0.03) + 12

        # ── Clinical interpretation ──
        interpretation = classify_trough(ensemble)

        return jsonify({
            'success': True,
            'predictions': {
                'random_forest': round(pred_rf, 2),
                'xgboost':       round(pred_xgb, 2) if pred_xgb is not None else None,
                'ensemble':      round(ensemble, 2),
                'traditional_pk': round(pred_pk, 2),
            },
            'derived_params': derived,
            'interpretation': interpretation,
            'xgboost_available': XGBOOST_AVAILABLE,
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/model-info')
def model_info():
    """Return model performance metrics and feature importance."""
    return jsonify({
        'random_forest': rf_metrics,
        'xgboost':       xgb_metrics if XGBOOST_AVAILABLE else None,
        'traditional_pk': pk_metrics,
        'feature_importance': rf_importance,
        'xgboost_available': XGBOOST_AVAILABLE,
        'training_samples': len(X_train),
        'test_samples':     len(X_test),
    })


# =============================================================================
# 6. CLINICAL INTERPRETATION
# =============================================================================
def classify_trough(val):
    if val < 10:
        return {
            'status': 'SUB-THERAPEUTIC',
            'color': '#8888ff',
            'emoji': '⬇️',
            'recommendation': (
                'Trough is below the therapeutic range (10–20 mg/L). '
                'Consider increasing the dose or shortening the dosing interval. '
                'Ensure adequate hydration and recheck SCr.'
            ),
            'urgency': 'moderate',
        }
    elif val <= 20:
        return {
            'status': 'THERAPEUTIC',
            'color': '#00ff9d',
            'emoji': '✅',
            'recommendation': (
                'Trough is within the therapeutic window (10–20 mg/L). '
                'Current regimen is appropriate. Continue standard monitoring. '
                'Recheck trough at next scheduled dose.'
            ),
            'urgency': 'none',
        }
    elif val <= 25:
        return {
            'status': 'HIGH — CAUTION',
            'color': '#ffd166',
            'emoji': '⚠️',
            'recommendation': (
                'Trough is elevated (20–25 mg/L). Risk of nephrotoxicity is increased. '
                'Consider extending the dosing interval or reducing dose by 10–15%. '
                'Monitor renal function (SCr, BUN) closely every 48h.'
            ),
            'urgency': 'moderate',
        }
    else:
        return {
            'status': 'TOXIC — DANGER',
            'color': '#ff6b6b',
            'emoji': '🚨',
            'recommendation': (
                'Trough is in the toxic range (>25 mg/L). Immediate action required. '
                'Hold the next dose and reassess. Urgently monitor renal function. '
                'Consider nephrology consult. Do not resume until trough < 15 mg/L.'
            ),
            'urgency': 'high',
        }


# =============================================================================
# 7. RUN
# =============================================================================
if __name__ == '__main__':
    app.run(debug=True, port=5000)
