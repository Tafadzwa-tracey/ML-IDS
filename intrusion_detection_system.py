'''import streamlit as st  # library used to build web app UI
import pandas as pd  # library used for reading and manipulating CSV file
from sklearn.preprocessing import LabelEncoder, StandardScaler  # LabelEncoder converts text labels into numbers, StandardScaler normalizes numeric features
from sklearn.ensemble import RandomForestClassifier  # imports Random Forest learning model
from sklearn.model_selection import train_test_split  # split data into training and testing set
from sklearn.metrics import classification_report  # Evaluates model performance by comparing actual and predicted labels
import matplotlib.pyplot as plt  # library for plotting graphs
import shap  # explain why model made a prediction
import numpy as np  # library for numeric sorting and indexing

st.title("ML Intrusion Detection System")  # display the title

# ------------------ Upload CSV ------------------
uploaded_file = st.file_uploader("Upload CSV file", type="csv")  # Creates button that lets user upload file

if uploaded_file:
    df = pd.read_csv(uploaded_file)  # Read CSV file into a pandas dataframe
    st.subheader("Dataset Preview")  # display title
    st.dataframe(df.head())  # shows first 5 rows of dataset on the app

    # ------------------ Preprocessing ------------------
    if 'Protocol' in df.columns:  # checks if column Protocol exists
        df['Protocol'] = LabelEncoder().fit_transform(df['Protocol'])  # Convert protocol names TCP ,UDP etc into numbers

    for col in ['Packets', 'Bytes']:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Converts Packets and Bytes columns to numbers, invalid numbers become NaN

    df = df.dropna(subset=['Packets', 'Bytes'])  # Remove rows where Packets or Bytes are missing
    df[['Packets', 'Bytes']] = StandardScaler().fit_transform(df[['Packets', 'Bytes']])  # Scales the values using mean 0 and standard deviation

    # ------------------ Features and Target ------------------
    X = df.drop(columns=['Label', 'Time', 'Source IP', 'Dest IP'])  # X input features removing columns that should not be used for training
    y = df['Label']  # y is the target Normal/Attack type

    # ------------------ Train Model ------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data into 80% training, 20% testing
    model = RandomForestClassifier(n_estimators=100, random_state=42)  # Create Random Forest with 100 decision trees
    model.fit(X_train, y_train)  # Train model using training data
    st.success("Model Trained Successfully!")  # Message to the app
    # Add this to your Streamlit code under the "Evaluation" section
    st.subheader("Global Feature Importance (SHAP)")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    st.pyplot(fig)

    # ------------------ Evaluation ------------------
    y_pred = model.predict(X_test)  # Predict labels for test data
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))  # Displays precision, recall, f1-score and accuracy


    # ------------------ SHAP Explanation ------------------
    st.subheader("Intrusion Alerts & Why They Were Detected")  # title for AI output

    explainer = shap.TreeExplainer(model)  # Creates a SHAP explainer for random forest
    shap_values = explainer.shap_values(X)  # Calculate feature importance for each prediction
    classes = model.classes_  # Get all possible class labels Normal, Dos etc

    # ------------------ Loop Through Attack Rows ------------------
    attack_rows = df.index[df['Label'].str.lower() != "normal"].tolist()  # Get all rows where label is not Normal

    for i in attack_rows:  # Process each network record one by one
        row = df.iloc[i]  # select row by its position
        src_ip = row['Source IP']  # Extract source IP
        dst_ip = row['Dest IP']  # Extract destination IP

        # Predict label and get probabilities for this row
        pred = model.predict(X.iloc[i:i+1])[0]  # Predict the label for one row at a time
        pred_prob = model.predict_proba(X.iloc[i:i+1])[0]  # Get probability for each class
        class_idx = list(classes).index(pred)  # Get index of predicted class

        # Get SHAP values for this row and class
        sv = shap_values[class_idx][i]  # SHAP values indicate feature contribution for predicted class

        # Get top 3 most important features
        top_n = 3
        top_indices = np.argsort(np.abs(sv))[-top_n:][::-1]  # Sort absolute SHAP values descending, largest first
        reasons = [X.columns[idx] for idx in top_indices]  # Map feature indices to column names

        # Display alert with prediction confidence and top features
        st.warning(f""" 🚨 INTRUSION DETECTED 
        From: {src_ip} → {dst_ip} 
        Attack Type: {pred} ({pred_prob[class_idx]*100:.1f}% confident) 
        Why: {', '.join(reasons)} """)  # Shows which features contributed most to the prediction

        # Optional: display mini bar chart of top features
        feature_values = sv[top_indices]  # Get SHAP values for top features
        chart_data = pd.DataFrame({
            'Feature': reasons,  # feature names
            'SHAP Value': feature_values  # corresponding SHAP values
        }).set_index('Feature')  # Set feature as index for plotting
        st.bar_chart(chart_data)  # Display bar chart of top features

    # ------------------ Normal Traffic Messages ------------------
    normal_rows = df.index[df['Label'].str.lower() == "normal"].tolist()  # Get rows where traffic is normal
    for i in normal_rows:
        row = df.iloc[i]  # select row
        st.info(f"✅ Normal: {row['Source IP']} → {row['Dest IP']}")  # Display normal traffic message

    # ------------------ Traffic Overview Chart ------------------
    st.subheader("Traffic Overview")  # Section title for chart
    if 'Time' in df.columns:  # Ensure dataset has a time column
        traffic = df.groupby(['Time', 'Label']).size().unstack(fill_value=0)  # Count traffic over time per label
        st.line_chart(traffic)  # Display line chart in Streamlit'''

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import shap
import numpy as np

# Page Config
st.set_page_config(page_title="ML IDS Research", layout="wide")
st.title("ML Intrusion Detection System")
st.write("🔬 **Research Topic:** Explainable AI (XAI) for Cybersecurity in Telecom")

# 1. Upload CSV
uploaded_file = st.file_uploader("Upload Network Logs (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # --- Preprocessing ---
    if 'Protocol' in df.columns:
        df['Protocol'] = LabelEncoder().fit_transform(df['Protocol'].astype(str))

    for col in ['Packets', 'Bytes']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Cleaning: Remove missing values
    df = df.dropna(subset=['Packets', 'Bytes', 'Label'])
    
    # Scaling
    scaler = StandardScaler()
    df[['Packets', 'Bytes']] = scaler.fit_transform(df[['Packets', 'Bytes']])

    # --- Feature Selection ---
    # We keep the names in 'X' so the model stays happy
    X = df.drop(columns=['Label', 'Time', 'Source IP', 'Dest IP'], errors='ignore')
    y = df['Label']

    # --- Train Model ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    st.success("✅ Model Trained Successfully with Feature Names!")

    # --- Evaluation ---
    y_pred = model.predict(X_test)
    st.subheader("Model Performance Metrics")
    st.text(classification_report(y_test, y_pred))

    # -------------------------------------------------------------------------
    # Phase 1: GLOBAL RESEARCH BASELINE (The "Figure 1" for your paper)
    # -------------------------------------------------------------------------
    st.header("Phase 1: Global Feature Importance")
    explainer = shap.TreeExplainer(model)
    
    # Sample 100 rows for SHAP to save hotspot data/speed
    sample_size = min(100, len(X_test))
    sample_X = X_test.iloc[:sample_size]
    shap_values_global = explainer.shap_values(sample_X)

    fig_global, ax_global = plt.subplots()
    shap.summary_plot(
        shap_values_global, 
        sample_X, 
        plot_type="bar", 
        class_names=model.classes_, 
        show=False
    )
    st.pyplot(fig_global)
    st.info("📊 **Research Note:** This chart proves which features define your telecom network security.")

    # -------------------------------------------------------------------------
    # Phase 2: LOCAL ALERTS (The "Why" reasons)
    # -------------------------------------------------------------------------
    st.header("Phase 2: Intrusion Alerts & Evidence")

    classes = list(model.classes_)
    
    # We calculate SHAP for the top rows of the original dataframe
    # Using X.iloc to ensure names are preserved
    alert_limit = min(30, len(df)) 
    shap_values_local = explainer.shap_values(X.iloc[:alert_limit])

    for i in range(alert_limit):
        row = df.iloc[i]
        src_ip = row.get('Source IP', 'Unknown')
        dst_ip = row.get('Dest IP', 'Unknown')

        # FIX: Pass a DataFrame slice (X.iloc[i:i+1]) instead of a numpy array
        # This keeps the feature names (Packets, Bytes) so the model doesn't error out
        current_row = X.iloc[i : i+1]
        pred = model.predict(current_row)[0]

        if str(pred).lower() != "normal":
            # Identify the index of the predicted attack class
            class_idx = classes.index(pred)
            
            # Get SHAP values for this specific row and class
            sv = shap_values_local[class_idx][i]

            # Find the top 2 features contributing to this specific alert
            top_indices = np.argsort(np.abs(sv))[-2:]
            top_indices = top_indices[::-1] # Sort descending

            reason1 = X.columns[top_indices[0]]
            reason2 = X.columns[top_indices[1]]

            st.warning(f"""
🚨 **INTRUSION DETECTED**
* **Connection:** {src_ip} ➔ {dst_ip}
* **Attack Category:** {pred}
* **Evidence (Why):** Detection triggered primarily by **{reason1}** and **{reason2}**.
            """)
        else:
            st.info(f"✅ **Normal Traffic:** {src_ip} ➔ {dst_ip}")

    # Traffic Overview
    if 'Time' in df.columns:
        st.subheader("Network Traffic Timeline")
        traffic = df.groupby(['Time', 'Label']).size().unstack(fill_value=0)
        st.line_chart(traffic)