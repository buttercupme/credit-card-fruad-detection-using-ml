import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import BorderlineSMOTE
import warnings
warnings.filterwarnings("ignore")

# ---------------------- Streamlit App Title ---------------------- #
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection")
st.markdown("Upload your credit card transaction data in CSV format.")

# ---------------------- File Upload ---------------------- #
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

@st.cache_resource
def train_model_cached(X_train, y_train):
    # Apply BorderlineSMOTE BEFORE training
    smote = BorderlineSMOTE(random_state=42, kind='borderline-1')
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    model = XGBClassifier(
        scale_pos_weight=4.4,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )

    model.fit(X_resampled, y_resampled)
    return model


@st.cache_resource
def get_shap_explainer(_model):
    return shap.Explainer(_model)

# ---------------------- Main Logic ---------------------- #
if uploaded_file is not None:
    try:
        data = load_data(uploaded_file)
    except Exception as e:
        st.error(f"❌ Failed to load CSV file: {e}")
        st.stop()

    if 'Class' not in data.columns:
        st.error("❌ Your CSV must contain a 'Class' column for labels (0 = Legit, 1 = Fraud).")
        st.stop()

    st.success("✅ File loaded successfully!")
    st.write("Preview of dataset:", data.head())

    # ---------------------- Split Data ---------------------- #
    X = data.drop("Class", axis=1)
    y = data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # ---------------------- Model Training ---------------------- #
    with st.spinner("⏳ Training the model..."):
        model = train_model_cached(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    st.metric(label="🎯 Model Accuracy on Test Set", value=f"{accuracy*100:.2f}%")

    # ---------------------- Tabbed Interface ---------------------- #
    tab1, tab2, tab3 = st.tabs(["📊 Data Visualization", "📌 Make Prediction", "🔍 Explainability"])

    with tab1:
        st.subheader("Fraud vs. Non-Fraud Transactions")
        fig1 = px.histogram(data, x='Class', title='Distribution of Transaction Classes (0=Legit, 1=Fraud)')
        st.plotly_chart(fig1)

    with tab2:
        st.subheader("Predict a Transaction")
        idx = st.number_input("Enter transaction index (0 to {}):".format(len(X_test)-1), min_value=0, max_value=len(X_test)-1, step=1)
        selected_transaction = X_test.iloc[idx:idx+1]
        prediction = model.predict(selected_transaction)[0]
        prediction_prob = model.predict_proba(selected_transaction)[0][1]

        result = "Fraud" if prediction == 1 else "Legit"
        st.write(f"*Prediction:* {result}")
        st.write(f"*Probability of Fraud:* {prediction_prob:.2%}")

    with tab3:
        st.subheader("SHAP Explainability")

        with st.spinner("🔍 Computing SHAP values..."):
            explainer = get_shap_explainer(model)
            shap_values = explainer(X_test.iloc[:200])  # Use a smaller sample

        st.write("Feature Importance Plot (SHAP Summary):")
        shap.summary_plot(shap_values, X_test.iloc[:200], plot_type="bar", show=False)
        st.pyplot(bbox_inches='tight')
        st.markdown("✔ SHAP values calculated for 200 transactions.")

else:
    st.warning("📂 Please upload a CSV file to get started.")
