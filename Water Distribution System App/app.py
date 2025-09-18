
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt

# Set Streamlit page configuration
st.set_page_config(page_title="Anomaly Detection Interface", layout="wide")

# Load models
models = {
    "Logistic Regression": joblib.load("models/lg_model.pkl"),
    "Random Forest": joblib.load("models/rf_model.pkl"),
    "Gradient Boosting": joblib.load("models/gb_model.pkl"),
    "SVM": joblib.load("models/svm_model.pkl"),
    "ANN": joblib.load("models/ann_model.pkl")
}

# Load scaler
scaler = joblib.load("models/scaler.pkl")

# Upload file
st.title("Anomaly Detection for Water Distribution Systems")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preprocessing Summary")
    st.write(f"Initial shape: {df.shape}")
    df = df.select_dtypes(include=[np.number])
    df.fillna(df.mean(), inplace=True)
    st.write(f"After removing non-numeric columns and imputing missing values: {df.shape}")

    if " ATT_FLAG" in df.columns:
        X = df.drop(" ATT_FLAG", axis=1)
        y = df[" ATT_FLAG"]
    else:
        X = df
        y = None

    X_scaled = scaler.transform(X)

    st.subheader("Model Selection")
    model_choice = st.selectbox("Select a model", list(models.keys()))
    model = models[model_choice]

    y_pred = model.predict(X_scaled)
    df["Prediction"] = y_pred

    st.subheader("Detection Results")
    st.write(df.head())

    if y is not None:
        st.text("Classification Report")
        st.text(classification_report(y, y_pred))

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_scaled)[:, 1]
        else:
            y_score = model.decision_function(X_scaled)
        fpr, tpr, _ = roc_curve(y, y_score)
        roc_auc = auc(fpr, tpr)
        st.subheader("ROC Curve")
        st.line_chart(pd.DataFrame({'FPR': fpr, 'TPR': tpr}))
        st.write(f"AUC Score: {roc_auc:.2f}")

    # Download prediction table
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
