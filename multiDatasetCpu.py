import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import shap
import warnings
import time
import seaborn as sns
import psutil  # New: For CPU monitoring
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    confusion_matrix, recall_score  
)
from xgboost import XGBClassifier

# --------------------------
# NSL-KDD Standard Column Names (Header-less Support)
# --------------------------
NSL_KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'Attack_type'
]

# --------------------------
# Core Config & Dark Cyber Theme (UNCHANGED)
# --------------------------
warnings.filterwarnings("ignore")
st.set_page_config(page_title="ML-Based Intrusion Detection System", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; color: white; }
    .stAlert { border-radius: 10px; border: 1px solid #30363d; }
    div[data-testid="stMetricValue"] { color: #58a6ff; font-size: 24px; }
    h1, h2, h3, h4, p, small, li { color: #c9d1d9; }
    .attack-alert { border-left: 5px solid #ff4b4b; background-color: #1c1c1c; padding: 20px; margin-bottom: 10px; border-radius: 5px; }
    .stDataFrame { color: #c9d1d9; }
    </style>
    """, unsafe_allow_html=True)

# --------------------------
# Title & Research Theme (Updated with CPU Monitoring)
# --------------------------
st.title(" Intrusion Detection System Using Machine Learning")
st.write(" **Research Topic:** Comparative Analysis of ML Models for Network Intrusion Detection + XAI + CPU Efficiency")

# --------------------------
# Sidebar Controls (Dataset + Model Selectors)
# --------------------------
st.sidebar.header(" ML Model Control Panel")
st.sidebar.markdown("### Interactive IDS Model Selector")
st.sidebar.markdown(" Drives Real-Time Intrusion Alerts\n  Updates Model-Specific SHAP XAI Deep Dive\n Static 3-Model Benchmark (Including CPU Efficiency)")

# Dataset Selector
st.sidebar.markdown("### Dataset Selector")
dataset_choice = st.sidebar.selectbox(
    "Choose Dataset",
    ["ML-EdgeIIoT", "NSL-KDD"]
)

# Model Selector
model_choice = st.sidebar.selectbox(
    "Select Active IDS Detection Model",
    ["Random Forest (Baseline)", "XGBoost (SOTA)", "MLP Neural Network (Deep Learning)"]
)

# --------------------------
# Data Loading & Preprocessing (Fixed: Consecutive Classes + Header Support)
# --------------------------
# Set file path
if dataset_choice == "ML-EdgeIIoT":
    file_path = 'ML-EdgeIIoT-dataset.csv'
else:
    file_path = 'NSL_KDD_Train.csv'

@st.cache_data
def load_and_prep(path, dataset_type):
    try:
        # Load dataset (handle header-less NSL-KDD)
        if dataset_type == "NSL-KDD":
            df = pd.read_csv(path, low_memory=False, nrows=15000, names=NSL_KDD_COLUMNS)
        else:
            df = pd.read_csv(path, low_memory=False, nrows=15000)
        
        # Drop non-IDS columns
        drop_cols = ['frame.time', 'ip.src_host', 'ip.dst_host', 'unnamed: 0', 'id', 'index', 'Attack_label']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        
        target = 'Attack_type'
        # Encode categorical features
        le = LabelEncoder()
        for col in df.columns:
            if df[col].dtype == object and col != target:
                df[col] = le.fit_transform(df[col].astype(str))
        
        df = df.dropna()
        df[target] = le.fit_transform(df[target].astype(str))
        
        # Step 1: Filter classes with <2 samples (avoid stratify error)
        class_counts = df[target].value_counts()
        valid_classes = class_counts[class_counts >= 2].index
        df = df[df[target].isin(valid_classes)].reset_index(drop=True)
        
        # Step 2: Reindex classes to be consecutive (fix Invalid classes error)
        unique_valid_classes = sorted(df[target].unique())
        class_mapping = {old: new for new, old in enumerate(unique_valid_classes)}
        df[target] = df[target].map(class_mapping)
        
        # Update LabelEncoder to match new consecutive labels
        le.classes_ = np.array([le.classes_[old] for old in unique_valid_classes])
        
        return df, le, target
    except FileNotFoundError:
        return None, None, None

# Load data + error handling
df, le, target_col = load_and_prep(file_path, dataset_choice)
if df is None:
    st.error(f"Dataset file not found! Place '{file_path}' in the same folder as this script.")
    st.stop()

# Global variables (UNCHANGED)
class_names = list(le.classes_)
feature_names = df.drop(columns=[target_col]).columns.tolist()
class_names_shap = class_names if len(class_names) >= len(np.unique(df[target_col])) else [f"Class {i}" for i in range(len(np.unique(df[target_col])))]
y = df[target_col]
X = df.drop(columns=[target_col])

# Preprocessing (UNCHANGED)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)
stratify_param = y if all(df[target_col].value_counts() >= 2) else None
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=stratify_param
)

# --------------------------
# CPU Monitoring Functions (New)
# --------------------------
def get_cpu_usage():
    """Return current CPU usage (%) and number of cores used."""
    cpu_percent = psutil.cpu_percent(interval=0.1)  # 0.1s sample for accuracy
    cpu_cores_used = sum(1 for core in psutil.cpu_percent(percpu=True) if core > 0)
    return cpu_percent, cpu_cores_used

def monitor_cpu_during_inference(model, X):
    """Measure inference time + CPU usage during model inference."""
    # Warmup CPU monitoring
    psutil.cpu_percent(interval=0.1, percpu=False)
    start_cpu, _ = get_cpu_usage()
    start_time = time.perf_counter()
    
    # Run inference (1000 samples for stable measurement)
    model.predict(X.iloc[:1000])
    
    # Calculate metrics
    end_time = time.perf_counter()
    end_cpu, cores_used = get_cpu_usage()
    avg_cpu = (start_cpu + end_cpu) / 2  # Average CPU during inference
    inf_time_per_packet = (end_time - start_time) / 1000  # Seconds per packet
    
    return {
        "inference_time": inf_time_per_packet,
        "avg_cpu_usage": avg_cpu,
        "cores_used": cores_used
    }

# --------------------------
# KPI Dashboard (Updated with System CPU Info)
# --------------------------
st.header(f" Research Overview & {dataset_choice} IDS KPIs")
# System CPU info (static)
total_cores = psutil.cpu_count(logical=True)
current_system_cpu, _ = get_cpu_usage()

attack_count = len(df[df[target_col] != 0])
normal_count = len(df[df[target_col] == 0])
col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric(" Analyzed Network Packets", f"{len(df):,}")
with col2: st.metric(" Normal Traffic Packets", f"{normal_count:,}")
with col3: st.metric("Detected Intrusions", f"{attack_count:,}")
with col4: st.metric("System CPU Cores", total_cores)
with col5: st.metric("Current System CPU (%)", f"{current_system_cpu:.1f}")
st.divider()

# --------------------------
# Phase 1: Threat Landscape Visualization (UNCHANGED)
# --------------------------
st.header(f" Phase 1: {dataset_choice} Threat Landscape & Intrusion Distribution")
c1, c2 = st.columns([1, 2])

# Donut Chart
with c1:
    threat_data = df[target_col].value_counts().reset_index()
    threat_data.columns = ['Class_ID', 'Count']
    threat_data['Intrusion_Type'] = threat_data['Class_ID'].apply(lambda x: class_names_shap[x] if x < len(class_names_shap) else f"Unknown {x}")
    fig = px.pie(
        threat_data, values='Count', names='Intrusion_Type', hole=0.5,
        title="Intrusion Type Distribution", color_discrete_sequence=px.colors.sequential.RdBu
    )
    fig.update_layout(template="plotly_dark", margin=dict(t=50, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)

# Time Series
with c2:
    st.subheader("Network Traffic & Intrusion Spikes (Batch-wise)")
    df_vis = df.copy()
    df_vis['Traffic_Batch'] = np.arange(len(df_vis)) // 100
    df_vis['Intrusion_Type'] = df_vis[target_col].apply(lambda x: class_names_shap[x] if x < len(class_names_shap) else f"Unknown {x}")
    timeline = df_vis.groupby(['Traffic_Batch', 'Intrusion_Type']).size().unstack(fill_value=0)
    st.line_chart(timeline, use_container_width=True)

# --------------------------
# Phase 2: Real-Time Intrusion Alerts (UNCHANGED)
# --------------------------
st.header(" Phase 2: Real-Time Network Intrusion Detection Alerts")
st.markdown(f"*Active IDS Model: {model_choice} | Analysis of First 5 Test Network Packets*")

# Train active model
if model_choice == "Random Forest (Baseline)":
    active_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
elif model_choice == "XGBoost (SOTA)":
    active_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss').fit(X_train, y_train)
else:
    active_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42).fit(X_train, y_train)

# Generate alerts
for i in range(5):
    packet = X_test.iloc[i:i+1]
    pred_idx = active_model.predict(packet)[0]
    pred_name = class_names_shap[pred_idx] if pred_idx < len(class_names_shap) else "Unknown Intrusion"
    model_name = model_choice.split(' ')[0]
    if pred_idx != 0:
        st.markdown(f"""
            <div class="attack-alert">
                <span style="color: #ff4b4b; font-weight: bold">🚨 INTRUSION DETECTED (Packet {i+1})</span><br>
                <small>IDS Model: {model_name} | Intrusion Type: {pred_name.upper()}</small><br>
                <p style="margin-top: 10px; color: #c9d1d9">The model flagged this packet due to anomalous network features (e.g., packet length, flow duration, protocol type).</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"✅ **Packet {i+1}**: Normal Network Traffic (Verified by {model_name} IDS Model)")

# --------------------------
# Phase 2.5: SHAP XAI Deep Dive (UNCHANGED)
# --------------------------
st.header(" Phase 2.5: Active Model SHAP XAI Deep Dive (Intrusion Feature Importance)")
st.markdown(f"*Detailed Feature Importance for {model_choice} | Explains *Why* the Model Detects Intrusions*")

# SHAP Calculation
plt.figure(figsize=(12, 7), facecolor='#0e1117')
if "Random Forest" in model_choice or "XGBoost" in model_choice:
    shap_explainer = shap.TreeExplainer(active_model)
    shap_vals = shap_explainer.shap_values(X_test.iloc[:50])
else:
    shap_background = shap.sample(X_train, 20, random_state=42)
    shap_explainer = shap.KernelExplainer(active_model.predict_proba, shap_background)
    shap_vals = shap_explainer.shap_values(X_test.iloc[:20])

# SHAP Plot
shap.summary_plot(
    shap_vals, X_test.iloc[:50], feature_names=feature_names,
    plot_type="bar", class_names=class_names_shap, show=False
)
plt.gcf().set_facecolor('#0e1117')
plt.gca().tick_params(colors='#c9d1d9')
plt.gca().xaxis.label.set_color('#c9d1d9')
plt.gca().yaxis.label.set_color('#c9d1d9')
plt.title(f"SHAP Feature Importance - {model_choice}", color='#c9d1d9', pad=20)
st.pyplot(plt.gcf())
plt.close()
st.divider()

# --------------------------
# Phase 3: ML-IDS Benchmark (Updated with CPU Metrics)
# --------------------------
st.header("Phase 3: ML-IDS Model Benchmark (Performance + CPU Efficiency)")
st.markdown("*IDS-Specific Metrics + CPU Usage | Precision/F1/Inference Time/CPU/Model Size*")

# Train benchmark models
@st.cache_data
def train_benchmark_models(Xtr, ytr):
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr)
    xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss').fit(Xtr, ytr)
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42).fit(Xtr, ytr)
    return rf, xgb, mlp

rf_bench, xgb_bench, mlp_bench = train_benchmark_models(X_train, y_train)
rf_pred = rf_bench.predict(X_test)
xgb_pred = xgb_bench.predict(X_test)
mlp_pred = mlp_bench.predict(X_test)

# Model Size Calculation (UNCHANGED)
def get_model_size(model, model_type):
    if model_type == "rf":
        return sum([tree.tree_.node_count for tree in model.estimators_])
    elif model_type == "xgb":
        return model.n_estimators * 100
    elif model_type == "mlp":
        return sum([w.size for w in model.coefs_]) + sum([b.size for b in model.intercepts_])

# Calculate All Metrics (Including CPU)
rf_metrics = monitor_cpu_during_inference(rf_bench, X_test)
xgb_metrics = monitor_cpu_during_inference(xgb_bench, X_test)
mlp_metrics = monitor_cpu_during_inference(mlp_bench, X_test)

metrics = {
    "RF": {
        "acc": accuracy_score(y_test, rf_pred),
        "f1": f1_score(y_test, rf_pred, average='weighted'),
        "recall": recall_score(y_test, rf_pred, average='weighted', zero_division=0),
        "inf_time": rf_metrics["inference_time"],
        "avg_cpu": rf_metrics["avg_cpu_usage"],
        "cores_used": rf_metrics["cores_used"],
        "size": get_model_size(rf_bench, "rf"),
        "interp": "High (Tree-Based)"
    },
    "XGB": {
        "acc": accuracy_score(y_test, xgb_pred),
        "f1": f1_score(y_test, xgb_pred, average='weighted'),
        "recall": recall_score(y_test, xgb_pred, average='weighted', zero_division=0),
        "inf_time": xgb_metrics["inference_time"],
        "avg_cpu": xgb_metrics["avg_cpu_usage"],
        "cores_used": xgb_metrics["cores_used"],
        "size": get_model_size(xgb_bench, "xgb"),
        "interp": "Medium (SHAP Interpretable)"
    },
    "MLP": {
        "acc": accuracy_score(y_test, mlp_pred),
        "f1": f1_score(y_test, mlp_pred, average='weighted'),
        "recall": recall_score(y_test, mlp_pred, average='weighted', zero_division=0),
        "inf_time": mlp_metrics["inference_time"],
        "avg_cpu": mlp_metrics["avg_cpu_usage"],
        "cores_used": mlp_metrics["cores_used"],
        "size": get_model_size(mlp_bench, "mlp"),
        "interp": "Low (Black-Box DL)"
    }
}

# Benchmark Table (With CPU Metrics)
benchmark_df = pd.DataFrame({
    "ML IDS Model": ["Random Forest (Baseline)", "XGBoost (SOTA)", "MLP Neural Network (DL)"],
    "Overall Test Accuracy": [f"{metrics['RF']['acc']:.4f}", f"{metrics['XGB']['acc']:.4f}", f"{metrics['MLP']['acc']:.4f}"],
    "Weighted F1-Score (IDS Core)": [f"{metrics['RF']['f1']:.4f}", f"{metrics['XGB']['f1']:.4f}", f"{metrics['MLP']['f1']:.4f}"],
    "Weighted Recall (Threat Detection)": [f"{metrics['RF']['recall']:.4f}", f"{metrics['XGB']['recall']:.4f}", f"{metrics['MLP']['recall']:.4f}"],
    "Inference Time (s/packet)": [f"{metrics['RF']['inf_time']:.6f}", f"{metrics['XGB']['inf_time']:.6f}", f"{metrics['MLP']['inf_time']:.6f}"],
    "Avg CPU Usage (%)": [f"{metrics['RF']['avg_cpu']:.1f}", f"{metrics['XGB']['avg_cpu']:.1f}", f"{metrics['MLP']['avg_cpu']:.1f}"],
    "Cores Used": [metrics['RF']['cores_used'], metrics['XGB']['cores_used'], metrics['MLP']['cores_used']],
    "Model Parameters (Size Proxy)": [metrics['RF']['size'], metrics['XGB']['size'], metrics['MLP']['size']],
    "Interpretability (IDS Compliance)": [metrics['RF']['interp'], metrics['XGB']['interp'], metrics['MLP']['interp']]
})
st.dataframe(benchmark_df, use_container_width=True)
st.caption(" Model Size: Proxy values for fair cross-model comparison | CPU Usage: Measured during inference on 1000 test packets")

# --------------------------
# New: CPU Usage Visualization
# --------------------------
st.subheader("Model CPU Efficiency Comparison")
cpu_data = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost", "MLP Neural Network"],
    "Avg CPU Usage (%)": [metrics['RF']['avg_cpu'], metrics['XGB']['avg_cpu'], metrics['MLP']['avg_cpu']],
    "Cores Used": [metrics['RF']['cores_used'], metrics['XGB']['cores_used'], metrics['MLP']['cores_used']]
})

# Dual Y-Axis Plot (CPU Usage + Cores Used)
fig, ax1 = plt.subplots(figsize=(10, 5), facecolor='#0e1117')

# Bar: Avg CPU Usage
bars = ax1.bar(cpu_data["Model"], cpu_data["Avg CPU Usage (%)"], color='#58a6ff', alpha=0.7, label="Avg CPU Usage (%)")
ax1.set_xlabel("ML Model", color='#c9d1d9')
ax1.set_ylabel("Avg CPU Usage (%)", color='#58a6ff')
ax1.tick_params(axis='y', labelcolor='#58a6ff')
ax1.tick_params(axis='x', labelcolor='#c9d1d9', rotation=45)

# Line: Cores Used
ax2 = ax1.twinx()
ax2.plot(cpu_data["Model"], cpu_data["Cores Used"], color='#ff4b4b', marker='o', linewidth=3, markersize=8, label="Cores Used")
ax2.set_ylabel("Cores Used", color='#ff4b4b')
ax2.tick_params(axis='y', labelcolor='#ff4b4b')

# Add value labels to bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f"{height:.1f}%", ha='center', va='bottom', color='#c9d1d9')

# Add value labels to line
for i, val in enumerate(cpu_data["Cores Used"]):
    ax2.text(i, val + 0.1, str(val), ha='center', va='bottom', color='#c9d1d9')

# Title + Layout
plt.title(f"CPU Efficiency: {dataset_choice} Dataset", color='#c9d1d9', pad=20)
fig.tight_layout()
st.pyplot(fig)
plt.close()

st.divider()

# --------------------------
# Phase 3.5: Confusion Matrices (UNCHANGED)
# --------------------------
st.header(" Phase 3.5: Confusion Matrices Comparison")
st.markdown("*Side-by-side performance: Random Forest, XGBoost, and Transformer/MLP*")

# Compute confusion matrices
cm_rf = confusion_matrix(y_test, rf_pred)
cm_xgb = confusion_matrix(y_test, xgb_pred)
cm_mlp = confusion_matrix(y_test, mlp_pred)

# Display plots
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Random Forest")
    plt.figure(figsize=(5, 4), facecolor='#0e1117')
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names_shap, yticklabels=class_names_shap, cbar=False)
    plt.xlabel('Predicted', color='#c9d1d9')
    plt.ylabel('True', color='#c9d1d9')
    plt.xticks(color='#c9d1d9', rotation=45, ha='right')
    plt.yticks(color='#c9d1d9')
    st.pyplot(plt.gcf())
    plt.close()
with col2:
    st.subheader("XGBoost")
    plt.figure(figsize=(5, 4), facecolor='#0e1117')
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names_shap, yticklabels=class_names_shap, cbar=False)
    plt.xlabel('Predicted', color='#c9d1d9')
    plt.ylabel('True', color='#c9d1d9')
    plt.xticks(color='#c9d1d9', rotation=45, ha='right')
    plt.yticks(color='#c9d1d9')
    st.pyplot(plt.gcf())
    plt.close()
with col3:
    st.subheader("Transformer / MLP")
    plt.figure(figsize=(5, 4), facecolor='#0e1117')
    sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names_shap, yticklabels=class_names_shap, cbar=False)
    plt.xlabel('Predicted', color='#c9d1d9')
    plt.ylabel('True', color='#c9d1d9')
    plt.xticks(color='#c9d1d9', rotation=45, ha='right')
    plt.yticks(color='#c9d1d9')
    st.pyplot(plt.gcf())
    plt.close()
st.divider()

# --------------------------
# Phase 4: 3-Model SHAP XAI Comparison (UNCHANGED)
# --------------------------
st.header(" Phase 4: 3-Model SHAP XAI Comparison (Intrusion Feature Importance)")
st.markdown("*Core Research XAI Insight | How Each Model Detects Intrusions | IDS Regulatory Compliance*")
shap_sample = X_test.iloc[:30]
col1, col2, col3 = st.columns(3)

# Random Forest SHAP
with col1:
    st.subheader("Random Forest (Baseline)")
    plt.figure(figsize=(6, 5), facecolor='#0e1117')
    rf_explainer = shap.TreeExplainer(rf_bench)
    rf_shap = rf_explainer.shap_values(shap_sample)
    shap.summary_plot(rf_shap, shap_sample, feature_names=feature_names, plot_type="bar", class_names=class_names_shap, show=False)
    plt.gcf().set_facecolor('#0e1117')
    plt.gca().tick_params(colors='#c9d1d9')
    plt.gca().xaxis.label.set_color('#c9d1d9')
    plt.gca().yaxis.label.set_color('#c9d1d9')
    st.pyplot(plt.gcf())
    plt.close()

# XGBoost SHAP
with col2:
    st.subheader("XGBoost (SOTA)")
    plt.figure(figsize=(6, 5), facecolor='#0e1117')
    xgb_explainer = shap.TreeExplainer(xgb_bench)
    xgb_shap = xgb_explainer.shap_values(shap_sample)
    shap.summary_plot(xgb_shap, shap_sample, feature_names=feature_names, plot_type="bar", class_names=class_names_shap, show=False)
    plt.gcf().set_facecolor('#0e1117')
    plt.gca().tick_params(colors='#c9d1d9')
    plt.gca().xaxis.label.set_color('#c9d1d9')
    plt.gca().yaxis.label.set_color('#c9d1d9')
    st.pyplot(plt.gcf())
    plt.close()

# MLP SHAP
with col3:
    st.subheader("MLP Neural Network (DL)")
    plt.figure(figsize=(6, 5), facecolor='#0e1117')
    mlp_background = shap.sample(X_train, 20, random_state=42)
    mlp_explainer = shap.KernelExplainer(mlp_bench.predict_proba, mlp_background)
    mlp_shap = mlp_explainer.shap_values(shap_sample)
    shap.summary_plot(mlp_shap, shap_sample, feature_names=feature_names, plot_type="bar", class_names=class_names_shap, show=False)
    plt.gcf().set_facecolor('#0e1117')
    plt.gca().tick_params(colors='#c9d1d9')
    plt.gca().xaxis.label.set_color('#c9d1d9')
    plt.gca().yaxis.label.set_color('#c9d1d9')
    st.pyplot(plt.gcf())
    plt.close()

