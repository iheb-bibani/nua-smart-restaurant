import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === Load model from GitHub ===
model_url = "https://raw.githubusercontent.com/iheb-bibani/nua-smart-restaurant/main/rf_model.pkl"

@st.cache_resource
def load_model():
    response = requests.get(model_url)
    model = joblib.load(BytesIO(response.content))
    return model

model = load_model()

# === Load data from GitHub ===
train_url = "https://raw.githubusercontent.com/iheb-bibani/nua-smart-restaurant/main/train_set.csv"
test_url = "https://raw.githubusercontent.com/iheb-bibani/nua-smart-restaurant/main/test_set.csv"

@st.cache_data
def load_data():
    train_df = pd.read_csv(train_url, parse_dates=['Date'])
    test_df = pd.read_csv(test_url, parse_dates=['Date'])
    return train_df, test_df

train_set, test_set = load_data()

# === Prepare data ===
X_train = train_set.drop(columns=["Revenue", "Date"])
y_train = train_set["Revenue"]
X_test = test_set.drop(columns=["Revenue", "Date"])
y_test = test_set["Revenue"]

# === Streamlit UI ===
st.title("Random Forest Evaluation & SHAP Interpretation")

# === Predictions & Errors ===
y_pred = model.predict(X_test)
error = y_test.reset_index(drop=True) - y_pred

# === Metrics ===
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

col1, col2, col3 = st.columns(3)
col1.metric("R² Score", f"{r2:.4f}")
col2.metric("MAE", f"{mae:.2f}")
col3.metric("RMSE", f"{rmse:.2f}")

# === Tabs ===
tab1, tab2, tab3 = st.tabs(["Predictions", "Errors", "SHAP Analysis"])

# === Tab 1: Predictions vs Actual ===
with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_set['Date'], y=y_test, mode='lines+markers', name='Actual'))
    fig.add_trace(go.Scatter(x=test_set['Date'], y=y_pred, mode='lines+markers', name='Predicted', line=dict(dash='dash')))
    fig.update_layout(title="Predicted vs Actual Revenue", xaxis_title="Date", yaxis_title="Revenue")
    st.plotly_chart(fig)

# === Tab 2: Errors ===
with tab2:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=np.arange(len(error)), y=np.abs(error), mode='lines', name='Absolute Error'))
    fig1.update_layout(title="Absolute Error Over Time", xaxis_title="Index", yaxis_title="Error")

    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=error, nbinsx=30, name="Error Distribution"))
    fig2.update_layout(title="Error Distribution", xaxis_title="Error", yaxis_title="Frequency")

    st.plotly_chart(fig1)
    st.plotly_chart(fig2)

# === Tab 3: SHAP Analysis ===
with tab3:
    st.subheader("SHAP Analysis")

    # Avoid deprecated warning for global use of pyplot
    st.set_option('deprecation.showPyplotGlobalUse', False)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    st.write("### Feature Importance (Bar Plot)")
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    st.pyplot(bbox_inches='tight')

    st.write("### SHAP Summary Plot")
    shap.summary_plot(shap_values, X_test)
    st.pyplot(bbox_inches='tight')
