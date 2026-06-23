"""
FraudShield AI - Synthetic Fraud Dataset Generator
Generates a realistic synthetic fraud dataset matching the original notebook schema.
"""
import numpy as np
import pandas as pd
from faker import Faker
import os

fake = Faker()
np.random.seed(42)

TRANSACTION_TYPES = ["Online", "POS", "ATM", "Wire Transfer", "Mobile"]
DEVICE_TYPES = ["Mobile", "Laptop", "Tablet", "Desktop"]
LOCATIONS = ["New York", "London", "Mumbai", "Tokyo", "Sydney", "Paris", "Berlin", "Dubai", "Singapore", "Toronto"]
MERCHANT_CATEGORIES = ["Retail", "Food & Beverage", "Travel", "Electronics", "Healthcare", "Entertainment", "Utilities", "Finance"]
CARD_TYPES = ["Visa", "Mastercard", "Amex", "Discover"]


def generate_fraud_dataset(n_samples: int = 10000, fraud_ratio: float = 0.08) -> pd.DataFrame:
    """
    Generate a synthetic fraud dataset.
    
    Args:
        n_samples: Total number of transactions
        fraud_ratio: Fraction of fraudulent transactions
        
    Returns:
        pd.DataFrame with fraud dataset
    """
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    records = []

    # Legitimate transactions
    for _ in range(n_legit):
        amount = np.random.lognormal(mean=4.5, sigma=1.0)
        amount = np.clip(amount, 5, 2000)
        records.append({
            "Transaction_ID": fake.uuid4()[:12].upper(),
            "User_ID": f"USER_{np.random.randint(1000, 9999)}",
            "Date": fake.date_time_between(start_date="-1y", end_date="now").strftime("%Y-%m-%d %H:%M:%S"),
            "Transaction_Amount": round(amount, 2),
            "Transaction_Type": np.random.choice(TRANSACTION_TYPES, p=[0.35, 0.30, 0.15, 0.10, 0.10]),
            "Account_Balance": round(np.random.uniform(500, 100000), 2),
            "Device_Type": np.random.choice(DEVICE_TYPES, p=[0.45, 0.30, 0.15, 0.10]),
            "Location": np.random.choice(LOCATIONS),
            "Merchant_Category": np.random.choice(MERCHANT_CATEGORIES),
            "Previous_Fraudulent_Activity": np.random.choice([0, 1], p=[0.95, 0.05]),
            "Daily_Transaction_Count": np.random.randint(1, 15),
            "Card_Age": np.random.randint(1, 120),
            "Card_Type": np.random.choice(CARD_TYPES),
            "Fraud_Label": 0
        })

    # Fraudulent transactions
    for _ in range(n_fraud):
        amount = np.random.lognormal(mean=6.5, sigma=1.5)
        amount = np.clip(amount, 100, 15000)
        records.append({
            "Transaction_ID": fake.uuid4()[:12].upper(),
            "User_ID": f"USER_{np.random.randint(1000, 9999)}",
            "Date": fake.date_time_between(start_date="-1y", end_date="now").strftime("%Y-%m-%d %H:%M:%S"),
            "Transaction_Amount": round(amount, 2),
            "Transaction_Type": np.random.choice(TRANSACTION_TYPES, p=[0.50, 0.15, 0.20, 0.10, 0.05]),
            "Account_Balance": round(np.random.uniform(100, 20000), 2),
            "Device_Type": np.random.choice(DEVICE_TYPES, p=[0.60, 0.20, 0.10, 0.10]),
            "Location": np.random.choice(LOCATIONS),
            "Merchant_Category": np.random.choice(MERCHANT_CATEGORIES),
            "Previous_Fraudulent_Activity": np.random.choice([0, 1], p=[0.60, 0.40]),
            "Daily_Transaction_Count": np.random.randint(5, 50),
            "Card_Age": np.random.randint(1, 36),
            "Card_Type": np.random.choice(CARD_TYPES),
            "Fraud_Label": 1
        })

    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def save_dataset(df: pd.DataFrame, path: str = "data/synthetic_fraud_dataset.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Dataset saved: {path} | Shape: {df.shape} | Fraud rate: {df['Fraud_Label'].mean():.2%}")


if __name__ == "__main__":
    df = generate_fraud_dataset(n_samples=15000)
    save_dataset(df, "data/synthetic_fraud_dataset.csv")
