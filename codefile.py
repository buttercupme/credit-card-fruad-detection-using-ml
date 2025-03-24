import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit as st
import shap
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go

# Set the page configuration
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
            font-family: 'Arial', sans-serif;
        }
        .stApp {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stTextInput>div>input {
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        .stPlotlyChart {
            max-width: 100%;
            margin: auto;
        }
        .section-header {
            font-size: 24px;
            margin-bottom: 16px;
            font-weight: bold;
        }
        .section-subheader {
            font-size: 20px;
            margin-bottom: 12px;
            font-weight: bold;
        }
        .info-text {
            font-size: 16px;
            margin-bottom: 8px;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def train_model(data):
    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]
    legit_sample = legit.sample(n=len(fraud), random_state=2)
    data = pd.concat([legit_sample, fraud], axis=0)
    X = data.drop(columns="Class", axis=1)
    y = data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    return model, train_acc, test_acc, X_train, X_test, y_train, y_test

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='RdBu', labels=dict(x="Predicted", y="Actual"))
    fig.update_xaxes(side="top", tickmode="array", tickvals=[0, 1], ticktext=["Legitimate", "Fraudulent"])
    fig.update_yaxes(tickmode="array", tickvals=[0, 1], ticktext=["Legitimate", "Fraudulent"])
    return fig

def parse_transaction_string(transaction_string, feature_names):
    values = transaction_string.split(",")
    if len(values) != len(feature_names):
        raise ValueError(f"Expected {len(feature_names)} features, but got {len(values)}.")
    transaction = {feature_names[i]: float(values[i]) for i in range(len(values))}
    return transaction

def shap_plot_to_html(shap_plot):
    shap_html = f"<head>{shap.getjs()}</head><body>{shap_plot.html()}</body>"
    return shap_html

def plot_shap_importance(shap_values, features):
    feature_importance = np.mean(np.abs(shap_values), axis=0)
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=feature_importance_df['Importance'],
        y=feature_importance_df['Feature'],
        orientation='h',
        marker=dict(color=feature_importance_df['Importance'], colorscale=['pink', 'blue'])
    ))

    fig.update_layout(
        title="SHAP Feature Importance",
        xaxis_title="Mean SHAP Value (Feature Importance)",
        yaxis_title="Features",
        template='plotly_white'
    )

    return fig

st.title("Credit Card Fraud Detection")

with st.spinner("Loading data..."):
    file = st.file_uploader("Upload a CSV file containing credit card transaction data:")
    if file is not None:
        data = load_data(file)
        st.write(f"Data loaded successfully! Data shape: {data.shape}")

        with st.spinner("Training model..."):
            model, train_acc, test_acc, X_train, X_test, y_train, y_test = train_model(data)

        st.subheader("Model Performance")
        st.write(f"Training accuracy: {train_acc:.2f}")
        st.write(f"Test accuracy: {test_acc:.2f}")

        y_pred = model.predict(X_test)
        st.write("Confusion Matrix:")

        # Plot confusion matrix with explanation
        fig_cm = plot_confusion_matrix(y_test, y_pred)
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown("""
            ### Confusion Matrix Explanation
            The confusion matrix shows the performance of the classification model. It consists of four blocks:

            - **True Positive (Top-Left)**: The number of legitimate transactions correctly classified.
            - **False Positive (Top-Right)**: The number of legitimate transactions incorrectly classified as fraudulent.
            - **False Negative (Bottom-Left)**: The number of fraudulent transactions incorrectly classified as legitimate.
            - **True Negative (Bottom-Right)**: The number of fraudulent transactions correctly classified.

            Each value in the matrix indicates the count of transactions for each classification outcome.
        """)

        st.subheader("Feature Importance using SHAP")
        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer.shap_values(X_test)

        # SHAP feature importance plot
        fig_shap_importance = plot_shap_importance(shap_values, X_train.columns)
        st.plotly_chart(fig_shap_importance, use_container_width=True)

        st.subheader("Check a Transaction")
        feature_names = data.drop(columns="Class", axis=1).columns
        num_features = len(feature_names)

        transaction_string = st.text_input(
            f"Enter {num_features} transaction features (comma-separated)",
            ",".join(["0.0"] * num_features),
            help="Enter the transaction features as comma-separated values"
        )

        if transaction_string:
            try:
                transaction = parse_transaction_string(transaction_string, feature_names)
                st.write("Parsed transaction:", transaction)
            except ValueError as e:
                st.error(e)

        if st.button("Submit Transaction"):
            try:
                transaction = parse_transaction_string(transaction_string, feature_names)
                transaction_df = pd.DataFrame([transaction])
                prediction = model.predict(transaction_df)
                prediction_proba = model.predict_proba(transaction_df)
                st.write(f"Prediction: {'Fraudulent' if prediction[0] == 1 else 'Legitimate'}")
                st.write(f"Prediction Probability: {prediction_proba[0]}")

                # SHAP explanation for the specific transaction
                shap_values_transaction = explainer.shap_values(transaction_df)
                shap_plot = shap.force_plot(explainer.expected_value, shap_values_transaction, transaction_df)

                # Convert SHAP plot to HTML
                shap_html = shap_plot_to_html(shap_plot)
                components.html(shap_html, height=400)

            except ValueError as e:
                st.error(e)
