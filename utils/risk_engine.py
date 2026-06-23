"""
FraudShield AI - Fraud Risk Engine
Classifies transactions into Low / Medium / High risk tiers
and generates risk scores (0–100).
"""
from dataclasses import dataclass


@dataclass
class RiskAssessment:
    fraud_probability: float
    risk_score: int          # 0-100
    risk_level: str          # Low | Medium | High
    risk_color: str          # Hex colour for UI
    risk_emoji: str
    recommendation: str


def assess_risk(fraud_probability: float) -> RiskAssessment:
    """
    Convert a raw fraud probability (0-1) into a structured risk assessment.
    
    Thresholds:
        Low    : prob < 0.35  → score 0-34
        Medium : 0.35 ≤ prob < 0.65 → score 35-64
        High   : prob ≥ 0.65  → score 65-100
    """
    score = int(fraud_probability * 100)
    score = max(0, min(100, score))

    if fraud_probability < 0.35:
        return RiskAssessment(
            fraud_probability=fraud_probability,
            risk_score=score,
            risk_level="Low",
            risk_color="#00C851",
            risk_emoji="🟢",
            recommendation="Transaction appears legitimate. No immediate action required.",
        )
    elif fraud_probability < 0.65:
        return RiskAssessment(
            fraud_probability=fraud_probability,
            risk_score=score,
            risk_level="Medium",
            risk_color="#FF8800",
            risk_emoji="🟡",
            recommendation="Elevated risk detected. Consider secondary verification (OTP / call-back).",
        )
    else:
        return RiskAssessment(
            fraud_probability=fraud_probability,
            risk_score=score,
            risk_level="High",
            risk_color="#FF4444",
            risk_emoji="🔴",
            recommendation="HIGH FRAUD RISK. Block transaction and trigger immediate investigation.",
        )


def batch_assess(probabilities) -> list[RiskAssessment]:
    return [assess_risk(p) for p in probabilities]
