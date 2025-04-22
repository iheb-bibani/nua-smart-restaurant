import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import shap
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
@st.cache_resource
def load_model():
    with open('rf_model.pkl', 'rb') as f:
        return joblib.load(f)

loaded_model = load_model()

# URLs GitHub bruts
BASE_RAW_URL = "https://raw.githubusercontent.com/MTSDEVS/MTS_AIWA_AI/main"
DATA_URL = f"{BASE_RAW_URL}/Data/Data/Dotky"

train_url = f"{DATA_URL}/train_set.csv"
test_url = f"{DATA_URL}/test_set.csv"

# Chargement des données depuis GitHub
@st.cache_data
def load_data():
    train_df = pd.read_csv(train_url, parse_dates=['Date'])
    test_df = pd.read_csv(test_url, parse_dates=['Date'])
    return train_df, test_df

# === Chargement et préparation des données ===
train_set, test_set = load_data()

X_train_uncorr = train_set.drop(columns=["Revenue", "Date"])
y_train = train_set["Revenue"]
X_test_uncorr = test_set.drop(columns=["Revenue", "Date"])
y_test = test_set["Revenue"]

# === Interface principale Streamlit ===
st.title("Random Forest Model Evaluation and SHAP Analysis")

# Prédictions
y_pred = loaded_model.predict(X_test_uncorr)
error = y_test.reset_index(drop=True) - y_pred

# === Affichage des métriques ===
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

col1, col2, col3 = st.columns(3)
col1.metric("R² Score", f"{r2:.4f}")
col2.metric("MAE", f"{mae:.2f}")
col3.metric("RMSE", f"{rmse:.2f}")

# === Onglets de visualisation ===
tab1, tab2, tab3 = st.tabs(["Predictions", "Errors", "SHAP"])

# Onglet 1 : Prédictions vs Réel
with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_set['Date'], y=y_test, mode='lines+markers', name='Actual'))
    fig.add_trace(go.Scatter(x=test_set['Date'], y=y_pred, mode='lines+markers', name='Predicted', line=dict(dash='dash')))
    fig.update_layout(title="Predictions vs Actual", xaxis_title="Date", yaxis_title="Revenue")
    st.plotly_chart(fig)

# Onglet 2 : Erreurs
with tab2:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=np.arange(len(error)), y=np.abs(error), mode='lines', name='Absolute Error'))
    fig1.update_layout(title="Absolute Error", xaxis_title="Index", yaxis_title="Error")

    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=error, nbinsx=30, name="Error Distribution"))
    fig2.update_layout(title="Error Distribution", xaxis_title="Error", yaxis_title="Count")

    st.plotly_chart(fig1)
    st.plotly_chart(fig2)

# Onglet 3 : SHAP Analysis
with tab3:
    st.subheader("SHAP Analysis")
    explainer = shap.TreeExplainer(loaded_model)
    shap_values = explainer.shap_values(X_test_uncorr)

    st.write("### Feature Importance")
    fig_summary, ax_summary = plt.subplots()
    shap.summary_plot(shap_values, X_test_uncorr, plot_type="bar", show=False)
    st.pyplot(fig_summary)
    plt.close(fig_summary)

    st.write("### SHAP Values")
    fig_shap, ax_shap = plt.subplots()
    shap.summary_plot(shap_values, X_test_uncorr, show=False)
    st.pyplot(fig_shap)
    plt.close(fig_shap)
