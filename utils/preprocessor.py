"""
FraudShield AI - Data Preprocessor
Handles feature engineering, encoding, and scaling.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os


FEATURE_COLUMNS = [
    "Transaction_Amount", "Account_Balance", "Daily_Transaction_Count", "Card_Age",
    "Previous_Fraudulent_Activity", "Transaction_Type_Encoded", "Device_Type_Encoded",
    "Location_Encoded", "Merchant_Category_Encoded", "Card_Type_Encoded",
    "Hour", "DayOfWeek", "Amount_to_Balance_Ratio"
]

CATEGORICAL_COLS = ["Transaction_Type", "Device_Type", "Location", "Merchant_Category", "Card_Type"]


class FraudPreprocessor:
    def __init__(self):
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        df = df.copy()
        df = self._engineer_features(df)
        df = self._encode_categoricals(df, fit=True)
        X = df[FEATURE_COLUMNS]
        y = df["Fraud_Label"]
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        return pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS), y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._engineer_features(df)
        df = self._encode_categoricals(df, fit=False)
        X = df[FEATURE_COLUMNS]
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["Hour"] = df["Date"].dt.hour.fillna(12).astype(int)
            df["DayOfWeek"] = df["Date"].dt.dayofweek.fillna(0).astype(int)
        else:
            df["Hour"] = 12
            df["DayOfWeek"] = 0

        df["Amount_to_Balance_Ratio"] = (
            df["Transaction_Amount"] / (df["Account_Balance"] + 1)
        ).clip(0, 100)
        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        for col in CATEGORICAL_COLS:
            encoded_col = f"{col}_Encoded"
            if col not in df.columns:
                df[encoded_col] = 0
                continue
            if fit:
                le = LabelEncoder()
                df[encoded_col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    # Handle unseen labels gracefully
                    known = set(le.classes_)
                    df[col] = df[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
                    df[encoded_col] = le.transform(df[col])
                else:
                    df[encoded_col] = 0
        return df

    def save(self, path: str = "models/preprocessor.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = "models/preprocessor.pkl") -> "FraudPreprocessor":
        return joblib.load(path)
