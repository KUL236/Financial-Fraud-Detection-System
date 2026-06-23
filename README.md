# 🛡️ FraudShield AI
### Financial Fraud Detection & Risk Analytics Platform

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> A production-ready AI platform for real-time financial fraud detection, risk scoring, and analytics — built as a full-stack internship-level project.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🤖 Multi-Model ML | Logistic Regression, Random Forest, XGBoost, LightGBM with auto-selection |
| ⚡ Real-Time Detection | Submit any transaction and receive instant fraud probability + risk score |
| 🎯 Risk Engine | 3-tier scoring (Low / Medium / High) with 0–100 risk score |
| 🔍 Explainable AI | SHAP feature importance and per-prediction waterfall charts |
| 📊 6-Page Dashboard | Executive Summary, Fraud Analytics, Risk Monitoring, Model Performance, Detection, History |
| 🗄️ SQLite Database | Full persistence of transactions, predictions, and alerts |
| 📄 PDF Reports | Downloadable per-transaction and executive summary reports |
| 🐳 Docker Ready | One-command container deployment |

---

## 📁 Project Structure

```
fraudshield-ai/
├── app.py                    # Main Streamlit application (6 pages)
├── requirements.txt
├── Dockerfile
├── Procfile                  # Heroku / Render deployment
├── runtime.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml           # Dark theme configuration
├── models/
│   ├── train.py              # Full ML training pipeline
│   ├── best_model.pkl        # Auto-generated after training
│   ├── preprocessor.pkl      # Feature engineering artifacts
│   └── model_metrics.json    # Saved evaluation metrics
├── data/
│   ├── synthetic_fraud_dataset.csv   # Auto-generated dataset
│   └── fraudshield.db        # SQLite database
├── reports/                  # Generated PDF reports
├── utils/
│   ├── __init__.py
│   ├── data_generator.py     # Synthetic dataset generator
│   ├── preprocessor.py       # Feature engineering & encoding
│   ├── risk_engine.py        # Fraud risk scoring engine
│   ├── database.py           # SQLite ORM layer
│   └── report_generator.py   # PDF report generation
├── notebooks/
│   └── FraudShield_EDA.ipynb
└── docs/
    ├── Architecture.md
    └── Deployment.md
```

---

## ⚙️ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/fraudshield-ai.git
cd fraudshield-ai
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train Models

```bash
python models/train.py
```

This will:
- Generate a 15,000-row synthetic fraud dataset
- Train 4 models (LR, RF, XGBoost, LightGBM) with SMOTE oversampling
- Auto-select the best model by ROC-AUC
- Save `models/best_model.pkl`, `models/preprocessor.pkl`, `models/model_metrics.json`

### 3. Run the Dashboard

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🐳 Docker Deployment

```bash
docker build -t fraudshield-ai .
docker run -p 8501:8501 fraudshield-ai
```

---

## ☁️ Deploy to Render / Railway

1. Push to GitHub
2. Connect repo on [render.com](https://render.com)
3. Set **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
4. Done — Render will auto-install from `requirements.txt`

---

## 🧠 ML Pipeline

```
Raw Data  →  Feature Engineering  →  Label Encoding  →  Standard Scaling
    ↓
SMOTE Oversampling  →  Train/Test Split (80/20)
    ↓
┌─────────────────────────────────────────────────┐
│  Logistic Regression  │  Random Forest           │
│  XGBoost              │  LightGBM                │
└─────────────────────────────────────────────────┘
    ↓
ROC-AUC Comparison  →  Best Model Selection  →  SHAP Explainability
```

### Features Used

| Feature | Type |
|---|---|
| Transaction_Amount | Numeric |
| Account_Balance | Numeric |
| Daily_Transaction_Count | Numeric |
| Card_Age | Numeric |
| Previous_Fraudulent_Activity | Binary |
| Amount_to_Balance_Ratio | Engineered |
| Hour, DayOfWeek | Temporal |
| Transaction_Type_Encoded | Categorical |
| Device_Type_Encoded | Categorical |
| Location_Encoded | Categorical |
| Merchant_Category_Encoded | Categorical |
| Card_Type_Encoded | Categorical |

---

## 📊 Dashboard Pages

| Page | Description |
|---|---|
| 📊 Executive Summary | KPI cards, fraud distribution, location heatmap |
| 🔍 Fraud Analytics | Time trends, amount analysis, device/merchant breakdowns, heatmaps |
| ⚡ Risk Monitoring | Live alerts, open case management, risk distribution |
| 🤖 Model Performance | Comparison table, radar chart, ROC curves, confusion matrix, SHAP |
| 🎯 Real-Time Detection | Transaction form → instant risk score + gauge + SHAP + PDF download |
| 📋 Prediction History | Full log with CSV export |

---

## 📄 License

MIT License — free for educational and commercial use.

---

## 👤 Author

Built as a FinTech AI internship project.  
Stack: Python · Streamlit · scikit-learn · XGBoost · LightGBM · SHAP · SQLite · Plotly · fpdf2
