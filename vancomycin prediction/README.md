# VancoPred — Vancomycin Trough Concentration Predictor

Full-stack ML web app integrating your two capstone notebooks into a live clinical decision support dashboard.

## Project Structure

```
vancopred/
├── app.py               ← Flask backend (ML pipeline from both notebooks)
├── requirements.txt
└── templates/
    └── index.html       ← Frontend (talks to Flask via /predict)
```

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Flask server
python app.py

# 3. Open in browser
http://localhost:5000
```

## What the Backend Does

`app.py` contains the exact ML pipeline from your notebooks:

### From CAPSTONE_FINAL_1.ipynb
- Random Forest (200 trees, max_depth=10)
- XGBoost (200 estimators, lr=0.05)
- StandardScaler
- Feature importance analysis

### From CAPSTONE_FINAL_2.ipynb
- Random Forest vs Traditional PK comparison
- Traditional PK baseline: `(dose × 0.015) − (CrCl × 0.03) + 12`

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the frontend |
| `/predict` | POST | Runs all 3 models, returns predictions + clinical interpretation |
| `/model-info` | GET | Returns metrics + feature importance (for Model Info tab) |

### Request Body for `/predict`
```json
{
  "age": 65,
  "weight": 75,
  "height": 170,
  "gender": 1,
  "serum_creatinine": 1.2,
  "dose": 1000,
  "dosing_interval": 12,
  "duration_of_infusion": 1.5,
  "time_to_trough": 1.0
}
```

### Response
```json
{
  "success": true,
  "predictions": {
    "random_forest": 12.45,
    "xgboost": 11.98,
    "ensemble": 12.22,
    "traditional_pk": 10.80
  },
  "derived_params": {
    "bmi": 25.95,
    "crcl": 65.1,
    "daily_dose": 2000.0,
    "dose_per_kg": 13.3
  },
  "interpretation": {
    "status": "THERAPEUTIC",
    "color": "#00ff9d",
    "recommendation": "Trough is within therapeutic window..."
  }
}
```

## Better Approaches (for Panel Discussion)

1. **NONMEM/Monolix** — gold standard population PK, FDA-accepted
2. **PK-ML Hybrid** — mechanistic PK equations + ML on residuals
3. **Bayesian MAP forecasting** — update population prior with observed troughs (how InsightRx/DoseMeRx work)
4. **Real data + SHAP** — replace synthetic data, add explainability

> ⚠️ Current models are trained on synthetic data — R² is high because the model re-learns the formula. With real data, expect lower but clinically meaningful performance.
