"""
FraudShield AI - Model Training Pipeline
Trains Logistic Regression, Random Forest, XGBoost, LightGBM.
Auto-selects best model and saves to models/best_model.pkl.
"""
import os
import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data_generator import generate_fraud_dataset, save_dataset
from utils.preprocessor import FraudPreprocessor

MODEL_DIR = os.path.join(os.path.dirname(__file__))
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_fraud_dataset.csv")


def load_or_generate_data(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        print(f"Loading existing dataset: {path}")
        return pd.read_csv(path)
    print("Generating synthetic dataset...")
    df = generate_fraud_dataset(n_samples=15000)
    save_dataset(df, path)
    return df


def evaluate_model(model, X_test, y_test, name: str) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "y_prob": y_prob.tolist(),
        "y_test": y_test.tolist(),
    }
    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    return metrics


def train_all_models(X_train, X_test, y_train, y_test) -> list[dict]:
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42, C=0.1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=10, use_label_encoder=False,
            eval_metric="logloss", random_state=42, verbosity=0
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            class_weight="balanced", random_state=42, verbose=-1
        ),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, name)
        metrics["trained_model"] = model
        results.append(metrics)
        trained_models[name] = model

    return results


def select_best_model(results: list[dict]) -> dict:
    """Select best model by ROC-AUC score."""
    best = max(results, key=lambda r: r["roc_auc"])
    print(f"\n🏆 Best Model: {best['model']} (ROC-AUC: {best['roc_auc']:.4f})")
    return best


def save_artifacts(best_result: dict, all_results: list[dict], preprocessor: FraudPreprocessor):
    """Save best model, preprocessor, and metrics."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save best model
    best_model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump(best_result["trained_model"], best_model_path)
    print(f"✅ Best model saved: {best_model_path}")

    # Save preprocessor
    preprocessor.save(os.path.join(MODEL_DIR, "preprocessor.pkl"))
    print(f"✅ Preprocessor saved")

    # Save metrics (without model object)
    metrics_to_save = []
    for r in all_results:
        m = {k: v for k, v in r.items() if k != "trained_model"}
        metrics_to_save.append(m)

    metrics_path = os.path.join(MODEL_DIR, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "trained_at": datetime.now().isoformat(),
            "best_model": best_result["model"],
            "models": metrics_to_save
        }, f, indent=2)
    print(f"✅ Metrics saved: {metrics_path}")


def main():
    print("\n" + "="*60)
    print("  FraudShield AI — Model Training Pipeline")
    print("="*60)

    # 1. Load data
    df = load_or_generate_data(DATA_PATH)
    print(f"\nDataset shape: {df.shape}")
    print(f"Fraud rate: {df['Fraud_Label'].mean():.2%}")

    # 2. Preprocess
    preprocessor = FraudPreprocessor()
    X, y = preprocessor.fit_transform(df)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\nTrain size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,}")

    # 4. SMOTE oversampling on training set
    print("\nApplying SMOTE oversampling...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE → fraud: {y_train_bal.sum():,} | legit: {(y_train_bal == 0).sum():,}")

    # 5. Train all models
    results = train_all_models(X_train_bal, X_test, y_train_bal, y_test)

    # 6. Select best
    best = select_best_model(results)

    # 7. Save
    save_artifacts(best, results, preprocessor)

    print("\n✅ Training complete!")
    return results, best


if __name__ == "__main__":
    main()
