# FraudShield AI — Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FraudShield AI Platform                       │
│                                                                       │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────────┐  │
│  │   Data Layer │    │   ML Pipeline    │    │  Streamlit UI     │  │
│  │              │    │                  │    │  (6 pages)        │  │
│  │  Synthetic   │──▶ │  Preprocessor    │──▶ │                   │  │
│  │  Generator   │    │  LR / RF         │    │  Executive Summary│  │
│  │              │    │  XGBoost         │    │  Fraud Analytics  │  │
│  │  SQLite DB   │◀── │  LightGBM        │    │  Risk Monitoring  │  │
│  │  (txns,preds,│    │  Best Model ✓    │    │  Model Perf.      │  │
│  │   alerts)    │    │                  │    │  Real-Time Detect │  │
│  └──────────────┘    │  SHAP XAI        │    │  History          │  │
│                      └──────────────────┘    └───────────────────┘  │
│                                                        │             │
│  ┌──────────────────────────────────────────────────── │ ──────────┐ │
│  │  Risk Engine              PDF Reports               ▼          │ │
│  │  Low / Medium / High  →   fpdf2 generator  ←  Download Button  │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Descriptions

### Data Layer (`utils/data_generator.py`, `utils/database.py`)
- **Synthetic Generator**: Creates realistic fraud datasets using `faker` + numpy probability distributions. Fraud cases have higher amounts, more daily transactions, more Online/ATM usage.
- **SQLite Database**: Three tables — `transactions`, `predictions`, `alerts`. Auto-creates High Risk alerts. Supports resolve/close workflow.

### ML Pipeline (`utils/preprocessor.py`, `models/train.py`)
- **Feature Engineering**: Extracts Hour, DayOfWeek from timestamps; computes Amount-to-Balance ratio.
- **Label Encoding**: Per-column LabelEncoder with unseen-label handling.
- **SMOTE**: Synthetic Minority Oversampling on training set only (no data leakage).
- **Models**: 4 classifiers evaluated on ROC-AUC; best saved as `models/best_model.pkl`.

### Risk Engine (`utils/risk_engine.py`)
- Converts raw `predict_proba` output (0–1) to 0–100 risk score.
- Three tiers with colour codes for UI rendering.

### Explainability (`shap`)
- `TreeExplainer` for tree-based models (RF, XGBoost, LightGBM).
- Per-prediction waterfall chart on the Real-Time Detection page.
- Global mean |SHAP| feature importance on Model Performance page.

### PDF Reports (`utils/report_generator.py`)
- `fpdf2`-based dark-themed reports.
- Per-transaction: full transaction details + risk assessment + recommendation.
- Executive summary: platform KPIs + model performance table.

## Data Flow — Real-Time Prediction

```
User Input (form)
  │
  ▼
pd.DataFrame (1 row)
  │
  ▼
FraudPreprocessor.transform()
  ├─ engineer_features()  → Hour, DayOfWeek, Amount/Balance ratio
  └─ encode_categoricals() → LabelEncoder (fitted at train time)
  │
  ▼
StandardScaler.transform()
  │
  ▼
model.predict_proba()[:, 1]  → fraud_probability (float 0-1)
  │
  ▼
risk_engine.assess_risk()   → RiskAssessment(score, level, color, recommendation)
  │
  ├─▶ SQLite insert (transaction + prediction + optional alert)
  ├─▶ Plotly gauge chart
  ├─▶ SHAP waterfall chart
  └─▶ PDF report generation + download button
```
