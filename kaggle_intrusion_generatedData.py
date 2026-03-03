import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shap
import warnings
warnings.filterwarnings("ignore")

# Page Config
st.set_page_config(page_title="ML IDS Research", layout="wide")
st.title("ML Intrusion Detection System")
st.write("🔬 **Research Topic:** Explainable AI (XAI) for Cybersecurity in Telecom")

# Upload CSV
uploaded_file = st.file_uploader("Upload Network Logs (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # --------------------------
    # Clean useless columns
    # --------------------------
    drop_cols = []
    for c in df.columns:
        if c.lower() in ['unnamed: 0', 'session_id', 'id', 'index']:
            drop_cols.append(c)
    df = df.drop(columns=drop_cols, errors='ignore')

    # --------------------------
    # Auto find label column
    # --------------------------
    target_candidates = ['label', 'outcome', 'class', 'attack_type', 'target']
    target_col = None
    for cand in target_candidates:
        if cand in df.columns:
            target_col = cand
            break
    if target_col is None:
        target_col = df.columns[-1]

    # --------------------------
    # Auto-encode ALL text columns (fixes 'DES' error)
    # --------------------------
    for col in df.columns:
        if col != target_col:
            if df[col].dtype == object:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

    df = df.dropna(subset=[target_col])

    # --------------------------
    # Features & Label
    # --------------------------
    X = df.drop(columns=[target_col])
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # --------------------------
    # Train Test Split
    # --------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # --------------------------
    # Model
    # --------------------------
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    st.success("✅ Model Trained Successfully with Feature Names!")

    # --------------------------
    # Evaluation
    # --------------------------
    y_pred = model.predict(X_test)
    st.subheader("Model Performance Metrics")
    st.text(classification_report(y_test, y_pred, zero_division=0))

    # -------------------------------------------------------------------------
    # Phase 1: GLOBAL SHAP
    # -------------------------------------------------------------------------
    st.header("Phase 1: Global Feature Importance")
    explainer = shap.TreeExplainer(model)

    sample_size = min(100, len(X_test))
    sample_X = X_test.iloc[:sample_size]
    shap_values_global = explainer.shap_values(sample_X)

    fig_global, ax_global = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values_global,
        sample_X,
        plot_type="bar",
        class_names=model.classes_,
        show=False
    )
    st.pyplot(fig_global)

    # -------------------------------------------------------------------------
    # Phase 2: LOCAL ALERTS & SHAP REASONS
    # -------------------------------------------------------------------------
    st.header("Phase 2: Intrusion Alerts & Evidence")

    classes = list(model.classes_)
    alert_limit = min(30, len(df))

    for i in range(alert_limit):
        current_row = X_scaled.iloc[i:i+1]
        pred = model.predict(current_row)[0]

        src_ip = f"Host-{i+1}"
        dst_ip = "Server"

        if str(pred).lower() not in ["normal", "0"]:
            try:
                class_idx = classes.index(pred)
                sv = explainer.shap_values(current_row)[class_idx][0]
                top_indices = np.argsort(np.abs(sv))[-2:]
                top_indices = top_indices[::-1]
                reason1 = X.columns[top_indices[0]]
                reason2 = X.columns[top_indices[1]]
            except:
                reason1 = "feature"
                reason2 = "anomaly"

            st.warning(f"""
🚨 **INTRUSION DETECTED**
* **Connection:** {src_ip} ➔ {dst_ip}
* **Attack Category:** {pred}
* **Evidence (Why):** Detection triggered primarily by **{reason1}** and **{reason2}**.
            """)
        else:
            st.info(f"✅ **Normal Traffic:** {src_ip} ➔ {dst_ip}")

    # -------------------------------------------------------------------------
    # Traffic Timeline Graph
    # -------------------------------------------------------------------------
    st.subheader("Network Traffic Timeline")
    if 'duration' in df.columns:
        df['time_bin'] = pd.cut(df['duration'], bins=10, labels=False)
        traffic = df.groupby(['time_bin', target_col]).size().unstack(fill_value=0)
        st.line_chart(traffic)
    else:
        traffic = df.groupby([target_col]).size()
        st.line_chart(traffic)