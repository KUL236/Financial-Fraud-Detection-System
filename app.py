"""
FraudShield AI – Financial Fraud Detection & Risk Analytics Platform
Main Streamlit Application
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from utils.database import init_db, insert_transaction, insert_prediction, get_prediction_history, get_alerts, get_summary_stats, resolve_alert
from utils.risk_engine import assess_risk
from utils.report_generator import generate_transaction_report, generate_summary_report
from utils.data_generator import generate_fraud_dataset

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
#  GLOBAL CSS  (FinTech dark theme)
# ─────────────────────────────────────────
st.markdown("""
<style>
  /* ---------- Base ---------- */
  .stApp { background: #0f172a; color: #e2e8f0; }
  section[data-testid="stSidebar"] { background: #1e293b !important; }
  section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

  /* ---------- Header bar ---------- */
  .fraud-header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-left: 4px solid #63b3ed;
    padding: 18px 24px; border-radius: 8px;
    margin-bottom: 20px;
  }
  .fraud-header h1 { color: #63b3ed; margin: 0; font-size: 1.9rem; }
  .fraud-header p  { color: #94a3b8; margin: 4px 0 0 0; font-size: 0.9rem; }

  /* ---------- Metric cards ---------- */
  .metric-card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 10px; padding: 20px 16px;
    text-align: center; transition: transform .15s;
  }
  .metric-card:hover { transform: translateY(-3px); border-color: #63b3ed; }
  .metric-card .value { font-size: 2rem; font-weight: 700; color: #63b3ed; }
  .metric-card .label { font-size: 0.8rem; color: #94a3b8; margin-top: 4px; }

  /* ---------- Risk badges ---------- */
  .risk-low    { background:#064e3b; color:#6ee7b7; border:1px solid #059669; padding:4px 10px; border-radius:20px; font-weight:600; }
  .risk-medium { background:#451a03; color:#fcd34d; border:1px solid #d97706; padding:4px 10px; border-radius:20px; font-weight:600; }
  .risk-high   { background:#450a0a; color:#fca5a5; border:1px solid #dc2626; padding:4px 10px; border-radius:20px; font-weight:600; }

  /* ---------- Form / inputs ---------- */
  .stTextInput>div>div>input,
  .stNumberInput>div>div>input,
  .stSelectbox>div>div>div { background:#1e293b !important; color:#e2e8f0 !important; border:1px solid #334155 !important; }

  /* ---------- Buttons ---------- */
  .stButton>button {
    background: linear-gradient(135deg,#2563eb,#1d4ed8);
    color: white; border: none; border-radius: 8px;
    padding: 8px 20px; font-weight: 600;
  }
  .stButton>button:hover { background: linear-gradient(135deg,#1d4ed8,#1e40af); }

  /* ---------- Tabs ---------- */
  .stTabs [data-baseweb="tab"] { color:#94a3b8; }
  .stTabs [aria-selected="true"] { color:#63b3ed !important; border-bottom-color:#63b3ed !important; }

  /* ---------- DataFrames ---------- */
  .dataframe { background:#1e293b !important; color:#e2e8f0 !important; }

  /* ---------- Alert boxes ---------- */
  .alert-box {
    background:#450a0a; border:1px solid #dc2626;
    border-radius:8px; padding:12px 16px; margin:6px 0;
  }
  .info-box {
    background:#1e3a5f; border:1px solid #2563eb;
    border-radius:8px; padding:12px 16px; margin:6px 0;
  }
  hr { border-color: #334155; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  INIT DB + SESSION STATE
# ─────────────────────────────────────────
init_db()

if "model" not in st.session_state:
    st.session_state.model = None
if "preprocessor" not in st.session_state:
    st.session_state.preprocessor = None
if "model_metrics" not in st.session_state:
    st.session_state.model_metrics = None
if "dataset" not in st.session_state:
    st.session_state.dataset = None


@st.cache_resource
def load_model_artifacts():
    model_path = "models/best_model.pkl"
    prep_path  = "models/preprocessor.pkl"
    metrics_path = "models/model_metrics.json"

    model = None
    preprocessor = None
    metrics = None

    if os.path.exists(model_path):
        model = joblib.load(model_path)
    if os.path.exists(prep_path):
        preprocessor = joblib.load(prep_path)
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    return model, preprocessor, metrics


@st.cache_data
def load_dataset():
    path = "data/synthetic_fraud_dataset.csv"
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["Date"])
    df = generate_fraud_dataset(5000)
    os.makedirs("data", exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _plotly_dark():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1e293b",
        font_color="#e2e8f0",
        xaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
        yaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
    )


# ─────────────────────────────────────────
#  SIDEBAR NAVIGATION
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:10px 0 20px'>
      <span style='font-size:2.4rem'>🛡️</span><br>
      <span style='color:#63b3ed;font-weight:700;font-size:1.1rem'>FraudShield AI</span><br>
      <span style='color:#94a3b8;font-size:0.75rem'>v1.0 · Production</span>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["📊 Executive Summary", "🔍 Fraud Analytics", "⚡ Risk Monitoring",
         "🤖 Model Performance", "🎯 Real-Time Detection", "📋 Prediction History"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("<p style='color:#64748b;font-size:0.75rem'>Model Status</p>", unsafe_allow_html=True)
    model, preprocessor, metrics = load_model_artifacts()
    if model:
        st.success("✅ Model Loaded")
        if metrics:
            st.caption(f"Best: **{metrics.get('best_model','N/A')}**")
    else:
        st.warning("⚠️ No model found")
        st.caption("Run models/train.py first")

    st.markdown("---")
    if st.button("🔄 Retrain Models", use_container_width=True):
        with st.spinner("Training models (this may take a minute)…"):
            try:
                from models.train import main as train_main
                load_dataset()  # ensure data exists
                train_main()
                st.cache_resource.clear()
                st.cache_data.clear()
                st.success("Training complete!")
                st.rerun()
            except Exception as e:
                st.error(f"Training error: {e}")


# ─────────────────────────────────────────
#  HELPER: SHAP explanation (lazy import)
# ─────────────────────────────────────────
def get_shap_values(model, X_sample):
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        return shap_values, explainer.expected_value
    except Exception:
        return None, None


# ═══════════════════════════════════════════════════════════════
#  PAGE 1 — EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════
if page == "📊 Executive Summary":
    st.markdown("""
    <div class='fraud-header'>
      <h1>📊 Executive Summary</h1>
      <p>High-level KPIs and fraud landscape overview</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_dataset()
    db_stats = get_summary_stats()

    total = len(df)
    fraud_count = int(df["Fraud_Label"].sum())
    fraud_rate = df["Fraud_Label"].mean()
    avg_fraud_amount = df[df["Fraud_Label"] == 1]["Transaction_Amount"].mean()
    avg_legit_amount = df[df["Fraud_Label"] == 0]["Transaction_Amount"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, label, color in [
        (c1, f"{total:,}", "Total Transactions", "#63b3ed"),
        (c2, f"{fraud_count:,}", "Fraud Cases", "#f87171"),
        (c3, f"{fraud_rate:.2%}", "Fraud Rate", "#fbbf24"),
        (c4, f"${avg_fraud_amount:,.0f}", "Avg Fraud Amount", "#f87171"),
        (c5, f"${avg_legit_amount:,.0f}", "Avg Legit Amount", "#6ee7b7"),
    ]:
        col.markdown(f"""
        <div class='metric-card'>
          <div class='value' style='color:{color}'>{val}</div>
          <div class='label'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Fraud vs Legitimate Transactions")
        fig = px.pie(
            values=[total - fraud_count, fraud_count],
            names=["Legitimate", "Fraudulent"],
            color_discrete_sequence=["#22c55e", "#ef4444"],
            hole=0.55,
        )
        fig.update_layout(**_plotly_dark(), showlegend=True, height=320)
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Fraud by Transaction Type")
        grp = df.groupby("Transaction_Type")["Fraud_Label"].agg(["sum", "count"])
        grp["rate"] = grp["sum"] / grp["count"]
        fig2 = px.bar(
            grp.reset_index(), x="Transaction_Type", y="rate",
            color="rate", color_continuous_scale=["#22c55e", "#fbbf24", "#ef4444"],
            labels={"rate": "Fraud Rate", "Transaction_Type": "Type"},
        )
        fig2.update_layout(**_plotly_dark(), height=320)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Transaction Amount Distribution")
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=df[df["Fraud_Label"] == 0]["Transaction_Amount"],
            name="Legitimate", marker_color="#22c55e", opacity=0.7, nbinsx=50
        ))
        fig3.add_trace(go.Histogram(
            x=df[df["Fraud_Label"] == 1]["Transaction_Amount"],
            name="Fraudulent", marker_color="#ef4444", opacity=0.7, nbinsx=50
        ))
        fig3.update_layout(**_plotly_dark(), barmode="overlay", height=320,
                           xaxis_title="Amount ($)", yaxis_title="Count")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Fraud by Location")
        loc = df.groupby("Location")["Fraud_Label"].mean().sort_values(ascending=True)
        fig4 = px.bar(
            loc.reset_index(), x="Fraud_Label", y="Location", orientation="h",
            color="Fraud_Label", color_continuous_scale=["#22c55e", "#ef4444"],
            labels={"Fraud_Label": "Fraud Rate", "Location": ""},
        )
        fig4.update_layout(**_plotly_dark(), height=320)
        st.plotly_chart(fig4, use_container_width=True)

    # Download summary report
    if metrics:
        st.markdown("---")
        if st.button("📥 Download Executive PDF Report"):
            path = generate_summary_report(db_stats, {
                m["model"]: f"ROC-AUC {m['roc_auc']}"
                for m in metrics.get("models", [])
            })
            with open(path, "rb") as f:
                st.download_button("⬇️ Download PDF", f, file_name=os.path.basename(path), mime="application/pdf")


# ═══════════════════════════════════════════════════════════════
#  PAGE 2 — FRAUD ANALYTICS
# ═══════════════════════════════════════════════════════════════
elif page == "🔍 Fraud Analytics":
    st.markdown("""
    <div class='fraud-header'>
      <h1>🔍 Fraud Analytics</h1>
      <p>Deep-dive visualisations across time, device, merchant & geography</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_dataset()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Hour"] = df["Date"].dt.hour
    df["DayOfWeek"] = df["Date"].dt.day_name()

    tab1, tab2, tab3, tab4 = st.tabs(["⏰ Time Analysis", "💳 Amount Analysis", "📱 Device & Merchant", "🌡️ Heatmaps"])

    with tab1:
        st.subheader("Fraud by Hour of Day")
        hour_grp = df.groupby(["Hour", "Fraud_Label"]).size().reset_index(name="count")
        fig = px.line(
            hour_grp, x="Hour", y="count", color="Fraud_Label",
            color_discrete_map={0: "#22c55e", 1: "#ef4444"},
            labels={"count": "Transactions", "Fraud_Label": "Fraud"},
            markers=True
        )
        fig.update_layout(**_plotly_dark(), height=360)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Daily Transaction Trend (30-day window)")
        df["DateOnly"] = df["Date"].dt.date
        daily = df.groupby(["DateOnly", "Fraud_Label"]).size().reset_index(name="count")
        daily_sorted = daily.sort_values("DateOnly").tail(60)
        fig2 = px.area(
            daily_sorted, x="DateOnly", y="count", color="Fraud_Label",
            color_discrete_map={0: "#22c55e", 1: "#ef4444"},
        )
        fig2.update_layout(**_plotly_dark(), height=300)
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("Fraud Amount vs Legitimate Amount")
        fig3 = go.Figure()
        fig3.add_trace(go.Box(
            y=df[df["Fraud_Label"] == 0]["Transaction_Amount"],
            name="Legitimate", marker_color="#22c55e"
        ))
        fig3.add_trace(go.Box(
            y=df[df["Fraud_Label"] == 1]["Transaction_Amount"],
            name="Fraudulent", marker_color="#ef4444"
        ))
        fig3.update_layout(**_plotly_dark(), height=380, yaxis_title="Amount ($)")
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Amount Buckets vs Fraud Rate")
        df["amount_bucket"] = pd.cut(df["Transaction_Amount"],
            bins=[0, 50, 200, 500, 1000, 5000, 99999],
            labels=["<$50", "$50-200", "$200-500", "$500-1K", "$1K-5K", ">$5K"])
        bucket = df.groupby("amount_bucket", observed=True)["Fraud_Label"].mean().reset_index()
        fig4 = px.bar(bucket, x="amount_bucket", y="Fraud_Label",
                      color="Fraud_Label", color_continuous_scale=["#22c55e", "#ef4444"],
                      labels={"Fraud_Label": "Fraud Rate", "amount_bucket": "Amount Range"})
        fig4.update_layout(**_plotly_dark(), height=300)
        st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Fraud by Device Type")
            dev = df.groupby("Device_Type")["Fraud_Label"].mean().reset_index()
            fig5 = px.bar(dev, x="Device_Type", y="Fraud_Label",
                          color="Fraud_Label", color_continuous_scale=["#22c55e", "#ef4444"],
                          labels={"Fraud_Label": "Fraud Rate"})
            fig5.update_layout(**_plotly_dark(), height=320)
            st.plotly_chart(fig5, use_container_width=True)

        with c2:
            st.subheader("Fraud by Merchant Category")
            merch = df.groupby("Merchant_Category")["Fraud_Label"].mean().sort_values(ascending=True).reset_index()
            fig6 = px.bar(merch, x="Fraud_Label", y="Merchant_Category", orientation="h",
                          color="Fraud_Label", color_continuous_scale=["#22c55e", "#ef4444"],
                          labels={"Fraud_Label": "Fraud Rate", "Merchant_Category": ""})
            fig6.update_layout(**_plotly_dark(), height=320)
            st.plotly_chart(fig6, use_container_width=True)

        st.subheader("Card Type Distribution Among Fraud Cases")
        card = df[df["Fraud_Label"] == 1].groupby("Card_Type").size().reset_index(name="count")
        fig7 = px.pie(card, values="count", names="Card_Type",
                      color_discrete_sequence=px.colors.sequential.Blues_r, hole=0.4)
        fig7.update_layout(**_plotly_dark(), height=320)
        st.plotly_chart(fig7, use_container_width=True)

    with tab4:
        st.subheader("Fraud Correlation Heatmap")
        import seaborn as sns, matplotlib.pyplot as plt
        num_cols = ["Transaction_Amount", "Account_Balance", "Daily_Transaction_Count",
                    "Card_Age", "Previous_Fraudulent_Activity", "Fraud_Label"]
        corr = df[num_cols].corr()
        fig8, ax = plt.subplots(figsize=(8, 5))
        fig8.patch.set_facecolor("#1e293b")
        ax.set_facecolor("#1e293b")
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
        ax.tick_params(colors="#e2e8f0")
        for text in ax.texts:
            text.set_color("#e2e8f0")
        st.pyplot(fig8)

        st.subheader("Hour × Day Fraud Heatmap")
        df["DayOfWeekNum"] = df["Date"].dt.dayofweek
        pivot = df[df["Fraud_Label"] == 1].pivot_table(
            index="DayOfWeekNum", columns="Hour", values="Fraud_Label",
            aggfunc="count", fill_value=0
        )
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        fig9 = px.imshow(pivot, labels=dict(x="Hour", y="Day", color="Fraud Count"),
                         y=[days[i] for i in pivot.index],
                         color_continuous_scale="Reds")
        fig9.update_layout(**_plotly_dark(), height=320)
        st.plotly_chart(fig9, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE 3 — RISK MONITORING
# ═══════════════════════════════════════════════════════════════
elif page == "⚡ Risk Monitoring":
    st.markdown("""
    <div class='fraud-header'>
      <h1>⚡ Risk Monitoring</h1>
      <p>Live alerts dashboard and open case management</p>
    </div>
    """, unsafe_allow_html=True)

    db_stats = get_summary_stats()

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label, color in [
        (c1, db_stats.get("total_transactions", 0), "Predictions Made", "#63b3ed"),
        (c2, db_stats.get("high_risk_count", 0), "High Risk", "#f87171"),
        (c3, db_stats.get("medium_risk_count", 0), "Medium Risk", "#fbbf24"),
        (c4, db_stats.get("open_alerts", 0), "Open Alerts", "#f87171"),
    ]:
        col.markdown(f"""
        <div class='metric-card'>
          <div class='value' style='color:{color}'>{val:,}</div>
          <div class='label'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🚨 Open Alerts")
    alerts = get_alerts(resolved=False)
    if alerts.empty:
        st.markdown("<div class='info-box'>✅ No open alerts at this time.</div>", unsafe_allow_html=True)
    else:
        for _, row in alerts.iterrows():
            col1, col2 = st.columns([5, 1])
            col1.markdown(f"""
            <div class='alert-box'>
              🔴 <b>Alert #{row['id']}</b> — TXN: {row['transaction_id']}<br>
              <small>{row['message']}<br>📅 {row['created_at']}</small>
            </div>
            """, unsafe_allow_html=True)
            if col2.button("✅ Resolve", key=f"resolve_{row['id']}"):
                resolve_alert(int(row["id"]))
                st.rerun()

    st.markdown("### 📊 Risk Distribution (All Predictions)")
    history = get_prediction_history(500)
    if not history.empty:
        risk_counts = history["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        fig = px.bar(risk_counts, x="Risk Level", y="Count",
                     color="Risk Level",
                     color_discrete_map={"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"})
        fig.update_layout(**_plotly_dark(), height=300)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run some predictions on the Real-Time Detection page to populate charts.")


# ═══════════════════════════════════════════════════════════════
#  PAGE 4 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown("""
    <div class='fraud-header'>
      <h1>🤖 Model Performance</h1>
      <p>Comparative evaluation of all trained models</p>
    </div>
    """, unsafe_allow_html=True)

    _, _, metrics = load_model_artifacts()

    if not metrics:
        st.warning("No trained models found. Use the **Retrain Models** button in the sidebar.")
        st.stop()

    models_data = metrics.get("models", [])
    best_name = metrics.get("best_model", "")

    # Comparison table
    st.subheader("📋 Model Comparison")
    comp_rows = []
    for m in models_data:
        star = " 🏆" if m["model"] == best_name else ""
        comp_rows.append({
            "Model": m["model"] + star,
            "Accuracy": f"{m['accuracy']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "F1 Score": f"{m['f1_score']:.4f}",
            "ROC-AUC": f"{m['roc_auc']:.4f}",
        })
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

    # Radar chart
    st.subheader("📡 Model Metrics Radar Chart")
    cats = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    fig_radar = go.Figure()
    colors = ["#63b3ed", "#fbbf24", "#6ee7b7", "#f87171"]
    for i, m in enumerate(models_data):
        vals = [m[k] for k in keys]
        vals.append(vals[0])
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=cats + [cats[0]],
            fill="toself", name=m["model"],
            line_color=colors[i % len(colors)], opacity=0.7
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#1e293b",
            radialaxis=dict(visible=True, range=[0, 1], color="#94a3b8"),
            angularaxis=dict(color="#94a3b8")
        ),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
        legend=dict(bgcolor="#1e293b", bordercolor="#334155"),
        height=420
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ROC curves
    st.subheader("📈 ROC Curves")
    from sklearn.metrics import roc_curve
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                  name="Random Baseline", line=dict(color="#64748b", dash="dash")))
    for i, m in enumerate(models_data):
        if "y_prob" in m and "y_test" in m:
            fpr, tpr, _ = roc_curve(m["y_test"], m["y_prob"])
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{m['model']} (AUC={m['roc_auc']:.3f})",
                line=dict(color=colors[i % len(colors)], width=2)
            ))
    fig_roc.update_layout(
        **_plotly_dark(), height=400,
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        legend=dict(bgcolor="#1e293b", bordercolor="#334155")
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # Confusion matrix for best model
    st.subheader(f"🧮 Confusion Matrix — {best_name}")
    best_metrics = next((m for m in models_data if m["model"] == best_name), None)
    if best_metrics and "confusion_matrix" in best_metrics:
        cm = best_metrics["confusion_matrix"]
        fig_cm = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Legitimate", "Fraud"], y=["Legitimate", "Fraud"],
            color_continuous_scale=["#1e293b", "#2563eb", "#ef4444"]
        )
        fig_cm.update_layout(**_plotly_dark(), height=350)
        st.plotly_chart(fig_cm, use_container_width=True)

    # Feature importance (SHAP or tree-based)
    st.subheader("🔍 Feature Importance")
    model, preprocessor, _ = load_model_artifacts()
    if model and preprocessor:
        try:
            df_raw = load_dataset()
            sample = df_raw.sample(min(300, len(df_raw)), random_state=42)
            X_sample = preprocessor.transform(sample)
            shap_vals, _ = get_shap_values(model, X_sample)
            if shap_vals is not None:
                importance = np.abs(shap_vals).mean(0)
                feat_names = X_sample.columns.tolist()
                fi_df = pd.DataFrame({"feature": feat_names, "importance": importance})
                fi_df = fi_df.sort_values("importance", ascending=True)
                fig_fi = px.bar(fi_df, x="importance", y="feature", orientation="h",
                                color="importance", color_continuous_scale=["#1e293b", "#63b3ed"],
                                labels={"importance": "Mean |SHAP|", "feature": ""})
                fig_fi.update_layout(**_plotly_dark(), height=400)
                st.plotly_chart(fig_fi, use_container_width=True)
                st.caption("SHAP (SHapley Additive exPlanations) mean absolute values — higher = more influential")
            else:
                # Fallback: tree feature importance
                if hasattr(model, "feature_importances_"):
                    fi_df = pd.DataFrame({
                        "feature": preprocessor.transform(sample).columns.tolist(),
                        "importance": model.feature_importances_
                    }).sort_values("importance", ascending=True)
                    fig_fi = px.bar(fi_df, x="importance", y="feature", orientation="h",
                                    color="importance", color_continuous_scale=["#1e293b", "#63b3ed"])
                    fig_fi.update_layout(**_plotly_dark(), height=400)
                    st.plotly_chart(fig_fi, use_container_width=True)
        except Exception as e:
            st.info(f"Feature importance unavailable: {e}")


# ═══════════════════════════════════════════════════════════════
#  PAGE 5 — REAL-TIME DETECTION
# ═══════════════════════════════════════════════════════════════
elif page == "🎯 Real-Time Detection":
    st.markdown("""
    <div class='fraud-header'>
      <h1>🎯 Real-Time Detection</h1>
      <p>Submit a transaction and get an instant fraud risk assessment</p>
    </div>
    """, unsafe_allow_html=True)

    model, preprocessor, _ = load_model_artifacts()

    if not model or not preprocessor:
        st.error("⚠️ No trained model found. Click **Retrain Models** in the sidebar first.")
        st.stop()

    with st.form("detection_form"):
        st.subheader("Transaction Details")
        c1, c2, c3 = st.columns(3)

        with c1:
            txn_amount = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=50000.0, value=250.0, step=10.0)
            account_balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=200000.0, value=5000.0, step=100.0)
            daily_txn_count = st.number_input("Daily Transaction Count", min_value=1, max_value=100, value=3)

        with c2:
            transaction_type = st.selectbox("Transaction Type", ["Online", "POS", "ATM", "Wire Transfer", "Mobile"])
            device_type = st.selectbox("Device Type", ["Mobile", "Laptop", "Tablet", "Desktop"])
            location = st.selectbox("Location", ["New York", "London", "Mumbai", "Tokyo", "Sydney", "Paris", "Berlin", "Dubai", "Singapore", "Toronto"])

        with c3:
            merchant_category = st.selectbox("Merchant Category", ["Retail", "Food & Beverage", "Travel", "Electronics", "Healthcare", "Entertainment", "Utilities", "Finance"])
            card_type = st.selectbox("Card Type", ["Visa", "Mastercard", "Amex", "Discover"])
            card_age = st.number_input("Card Age (months)", min_value=1, max_value=240, value=24)
            prior_fraud = st.selectbox("Previous Fraudulent Activity", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        submitted = st.form_submit_button("🔍 Analyze Transaction", use_container_width=True)

    if submitted:
        import uuid
        txn_id = f"TXN-{uuid.uuid4().hex[:8].upper()}"
        txn_data = {
            "Transaction_ID": txn_id,
            "User_ID": f"USER-{uuid.uuid4().hex[:6].upper()}",
            "Transaction_Amount": txn_amount,
            "Transaction_Type": transaction_type,
            "Account_Balance": account_balance,
            "Device_Type": device_type,
            "Location": location,
            "Merchant_Category": merchant_category,
            "Card_Type": card_type,
            "Daily_Transaction_Count": daily_txn_count,
            "Card_Age": card_age,
            "Previous_Fraudulent_Activity": prior_fraud,
            "Date": datetime.now().isoformat(),
        }

        # Predict
        input_df = pd.DataFrame([txn_data])
        X = preprocessor.transform(input_df)
        prob = float(model.predict_proba(X)[0][1])
        assessment = assess_risk(prob)

        # Persist
        insert_transaction(txn_data)
        insert_prediction({
            "transaction_id": txn_id,
            "fraud_probability": prob,
            "risk_score": assessment.risk_score,
            "risk_level": assessment.risk_level,
            "model_used": "best_model"
        })

        # Display result
        st.markdown("---")
        st.subheader("🎯 Assessment Result")

        risk_css = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}[assessment.risk_level]
        st.markdown(f"""
        <div style='text-align:center;padding:20px;background:#1e293b;border-radius:12px;border:1px solid #334155;margin:10px 0'>
          <div style='font-size:3rem'>{assessment.risk_emoji}</div>
          <div class='{risk_css}' style='font-size:1.4rem;display:inline-block;margin:10px 0'>{assessment.risk_level.upper()} RISK</div>
          <br>
          <span style='font-size:2rem;font-weight:700;color:#e2e8f0'>Risk Score: {assessment.risk_score}/100</span><br>
          <span style='color:#94a3b8'>Fraud Probability: {prob:.2%}</span>
        </div>
        """, unsafe_allow_html=True)

        # Risk gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=assessment.risk_score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Risk Score", "font": {"color": "#e2e8f0"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#e2e8f0"},
                "bar": {"color": assessment.risk_color},
                "bgcolor": "#1e293b",
                "steps": [
                    {"range": [0, 35], "color": "#064e3b"},
                    {"range": [35, 65], "color": "#451a03"},
                    {"range": [65, 100], "color": "#450a0a"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "thickness": 0.75, "value": assessment.risk_score}
            }
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", height=280)
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
        <div class='info-box'>
          <b>📋 Recommendation:</b><br>{assessment.recommendation}
        </div>
        """, unsafe_allow_html=True)

        # SHAP waterfall (if available)
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X)
            if isinstance(shap_vals, list):
                sv = shap_vals[1][0]
                base = float(explainer.expected_value[1])
            else:
                sv = shap_vals[0]
                base = float(explainer.expected_value)
            feat_names = X.columns.tolist()
            shap_df = pd.DataFrame({"Feature": feat_names, "SHAP": sv}).sort_values("SHAP", key=abs, ascending=False).head(10)
            fig_shap = px.bar(shap_df, x="SHAP", y="Feature", orientation="h",
                              color="SHAP", color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
                              title="Top 10 Contributing Features (SHAP)",
                              labels={"SHAP": "SHAP Value", "Feature": ""})
            fig_shap.update_layout(**_plotly_dark(), height=380)
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception:
            pass

        # PDF download
        pdf_path = generate_transaction_report(txn_data, assessment)
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Download Fraud Report (PDF)", f, file_name=os.path.basename(pdf_path), mime="application/pdf")


# ═══════════════════════════════════════════════════════════════
#  PAGE 6 — PREDICTION HISTORY
# ═══════════════════════════════════════════════════════════════
elif page == "📋 Prediction History":
    st.markdown("""
    <div class='fraud-header'>
      <h1>📋 Prediction History</h1>
      <p>Full log of all analyzed transactions</p>
    </div>
    """, unsafe_allow_html=True)

    history = get_prediction_history(200)

    if history.empty:
        st.info("No predictions yet. Use the Real-Time Detection page to analyze transactions.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Predictions", len(history))
        c2.metric("High Risk", int((history["risk_level"] == "High").sum()))
        c3.metric("Avg Risk Score", f"{history['risk_score'].mean():.1f}")

        # Filter
        col1, col2 = st.columns(2)
        risk_filter = col1.multiselect("Filter by Risk Level", ["Low", "Medium", "High"],
                                        default=["Low", "Medium", "High"])
        history_filtered = history[history["risk_level"].isin(risk_filter)]

        # Color-coded table
        def color_risk(val):
            colors = {"Low": "color: #6ee7b7", "Medium": "color: #fcd34d", "High": "color: #fca5a5"}
            return colors.get(val, "")

        display_cols = ["transaction_id", "risk_level", "risk_score", "fraud_probability",
                        "transaction_amount", "location", "transaction_type", "created_at"]
        available_cols = [c for c in display_cols if c in history_filtered.columns]
        styled = history_filtered[available_cols].style.applymap(color_risk, subset=["risk_level"])
        st.dataframe(styled, use_container_width=True)

        # Risk over time
        if "created_at" in history_filtered.columns:
            history_filtered = history_filtered.copy()
            history_filtered["created_at"] = pd.to_datetime(history_filtered["created_at"])
            fig = px.scatter(
                history_filtered, x="created_at", y="risk_score",
                color="risk_level",
                color_discrete_map={"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"},
                title="Risk Score Over Time",
                labels={"created_at": "Time", "risk_score": "Risk Score"}
            )
            fig.update_layout(**_plotly_dark(), height=320)
            st.plotly_chart(fig, use_container_width=True)

        # CSV export
        csv = history_filtered.to_csv(index=False)
        st.download_button("📥 Export CSV", csv, file_name="fraud_prediction_history.csv", mime="text/csv")
