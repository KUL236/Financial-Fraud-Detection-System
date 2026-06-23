"""
FraudShield AI - SQLite Database Layer
Stores transactions, predictions, alerts, and prediction history.
"""
import sqlite3
import os
from datetime import datetime
from typing import Optional
import pandas as pd

DB_PATH = os.getenv("FRAUDSHIELD_DB", "data/fraudshield.db")


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they do not exist."""
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT UNIQUE,
            user_id TEXT,
            transaction_amount REAL,
            transaction_type TEXT,
            account_balance REAL,
            device_type TEXT,
            location TEXT,
            merchant_category TEXT,
            card_type TEXT,
            daily_transaction_count INTEGER,
            card_age INTEGER,
            previous_fraudulent_activity INTEGER,
            timestamp TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT,
            fraud_probability REAL,
            risk_score INTEGER,
            risk_level TEXT,
            model_used TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT,
            risk_level TEXT,
            message TEXT,
            is_resolved INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def insert_transaction(txn: dict) -> None:
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT OR IGNORE INTO transactions
        (transaction_id, user_id, transaction_amount, transaction_type,
         account_balance, device_type, location, merchant_category,
         card_type, daily_transaction_count, card_age,
         previous_fraudulent_activity, timestamp)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        txn.get("Transaction_ID"), txn.get("User_ID"),
        txn.get("Transaction_Amount"), txn.get("Transaction_Type"),
        txn.get("Account_Balance"), txn.get("Device_Type"),
        txn.get("Location"), txn.get("Merchant_Category"),
        txn.get("Card_Type"), txn.get("Daily_Transaction_Count"),
        txn.get("Card_Age"), txn.get("Previous_Fraudulent_Activity"),
        txn.get("Date", datetime.now().isoformat())
    ))
    conn.commit()
    conn.close()


def insert_prediction(prediction: dict) -> None:
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO predictions (transaction_id, fraud_probability, risk_score, risk_level, model_used)
        VALUES (?,?,?,?,?)
    """, (
        prediction["transaction_id"], prediction["fraud_probability"],
        prediction["risk_score"], prediction["risk_level"],
        prediction.get("model_used", "best_model")
    ))
    # Auto-create alert for High risk
    if prediction["risk_level"] == "High":
        c.execute("""
            INSERT INTO alerts (transaction_id, risk_level, message)
            VALUES (?,?,?)
        """, (
            prediction["transaction_id"], "High",
            f"HIGH FRAUD RISK detected for transaction {prediction['transaction_id']} "
            f"(score: {prediction['risk_score']}/100)"
        ))
    conn.commit()
    conn.close()


def get_prediction_history(limit: int = 200) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT p.*, t.transaction_amount, t.location, t.transaction_type
        FROM predictions p
        LEFT JOIN transactions t ON p.transaction_id = t.transaction_id
        ORDER BY p.created_at DESC
        LIMIT ?
    """, conn, params=(limit,))
    conn.close()
    return df


def get_alerts(resolved: Optional[bool] = None, limit: int = 50) -> pd.DataFrame:
    conn = get_connection()
    query = "SELECT * FROM alerts"
    params = []
    if resolved is not None:
        query += " WHERE is_resolved = ?"
        params.append(int(resolved))
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def resolve_alert(alert_id: int) -> None:
    conn = get_connection()
    conn.execute("UPDATE alerts SET is_resolved = 1 WHERE id = ?", (alert_id,))
    conn.commit()
    conn.close()


def get_summary_stats() -> dict:
    conn = get_connection()
    c = conn.cursor()
    stats = {}
    c.execute("SELECT COUNT(*) FROM transactions")
    stats["total_transactions"] = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM predictions WHERE risk_level = 'High'")
    stats["high_risk_count"] = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM predictions WHERE risk_level = 'Medium'")
    stats["medium_risk_count"] = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM predictions WHERE risk_level = 'Low'")
    stats["low_risk_count"] = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM alerts WHERE is_resolved = 0")
    stats["open_alerts"] = c.fetchone()[0]
    conn.close()
    return stats
