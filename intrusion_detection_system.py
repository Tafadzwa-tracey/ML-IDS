import streamlit as st  # library used to build web app UI
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
        st.line_chart(traffic)  # Display line chart in Streamlit