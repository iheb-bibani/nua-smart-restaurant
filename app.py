import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import shap
import pickle # Using pickle for scaler/lists
# import joblib # Uncomment if model was saved with joblib
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.preprocessing import StandardScaler # Only needed for type hints, not execution

# --- Page Configuration ---
st.set_page_config(layout="wide") # Use wider layout

# === Configuration: File Paths ===
# Ensure these files are in the same directory as the script or provide full paths
MODEL_PATH = "catboost_model.pkl"
SCALER_PATH = "scaler.pkl"
SCALER_COLUMNS_PATH = "scaler_columns.pkl"
FINAL_MODEL_COLUMNS_PATH = "final_model_columns.pkl"
# Using GitHub URL for test data for demonstration, could be a local path too
TEST_DATA_URL = "https://raw.githubusercontent.com/iheb-bibani/nua-smart-restaurant/main/test_set.csv"

# === Load Artifacts (Model, Scaler, Column Lists) ===

@st.cache_resource # Cache resource, doesn't change unless file changes
def load_model(path):
    """Loads the model from a pickle file."""
    try:
        # If model saved with pickle:
        with open(path, "rb") as f:
            model = pickle.load(f)
        # If model saved with joblib:
        # from joblib import load
        # model = load(path)
        print("Model loaded successfully.")
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {path}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource # Cache resource
def load_scaler(path):
    """Loads the scaler object from a pickle file."""
    try:
        with open(path, "rb") as f:
            scaler = pickle.load(f)
        print("Scaler loaded successfully.")
        return scaler
    except FileNotFoundError:
        st.error(f"Error: Scaler file not found at {path}")
        return None
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None

@st.cache_data # Cache data, content won't change
def load_column_list(path):
    """Loads a list of columns from a pickle file."""
    try:
        with open(path, "rb") as f:
            columns = pickle.load(f)
        print(f"Column list loaded successfully from {path}.")
        return columns
    except FileNotFoundError:
        st.error(f"Error: Column list file not found at {path}")
        return None
    except Exception as e:
        st.error(f"Error loading column list from {path}: {e}")
        return None

@st.cache_data # Cache test data
def load_test_data(url):
    """Loads the test dataset from a URL or local path."""
    try:
        test_df = pd.read_csv(url, parse_dates=['Date'])
        print("Test data loaded successfully.")
        return test_df
    except Exception as e:
        st.error(f"Error loading test data from {url}: {e}")
        return None

# --- Load everything ---
model = load_model(MODEL_PATH)
scaler = load_scaler(SCALER_PATH)
scaler_columns = load_column_list(SCALER_COLUMNS_PATH)
final_model_columns = load_column_list(FINAL_MODEL_COLUMNS_PATH)
test_set = load_test_data(TEST_DATA_URL)

# --- Check if loading was successful before proceeding ---
if not all([model, scaler, scaler_columns, final_model_columns, test_set is not None]):
    st.error("One or more essential files (model, scaler, column lists, test data) failed to load. Cannot proceed.")
    st.stop() # Halt execution if essentials are missing

# === Prepare Data from Loaded Test Set ===
# Features need preprocessing; Target is used as is for evaluation
X_test_original = test_set.drop(columns=["Revenue", "Date"]).copy() # Keep original for reference
y_test = test_set["Revenue"].copy()
test_dates = test_set["Date"].copy()

# --- Preprocessing Pipeline (Mirroring API/Training) ---
# This block defines X_test_processed_final
st.sidebar.write("--- Preprocessing Test Data ---") # Put messages in sidebar
try:
    # 1. Ensure DataFrame has the columns the scaler expects
    st.sidebar.write(f"Columns expected by scaler: {len(scaler_columns)}")
    st.sidebar.write(f"Columns in loaded X_test: {X_test_original.shape[1]}")
    X_test_for_scaler = X_test_original[scaler_columns]
    st.sidebar.write(f"Shape before scaling: {X_test_for_scaler.shape}")

    # 2. Apply the SCALER TRANSFORM
    scaled_data = scaler.transform(X_test_for_scaler)
    X_test_scaled = pd.DataFrame(scaled_data, index=X_test_for_scaler.index, columns=scaler_columns)
    st.sidebar.write(f"Shape after scaling: {X_test_scaled.shape}")

    # 3. Select only the final features the model was trained on
    st.sidebar.write(f"Columns expected by final model: {len(final_model_columns)}")
    # Define the crucial variable here:
    X_test_processed_final = X_test_scaled[final_model_columns]
    st.sidebar.write(f"Shape after final selection: {X_test_processed_final.shape}")
    st.sidebar.success("Preprocessing complete.")

except KeyError as e:
    st.error(f"Preprocessing Error: Missing column - {e}. Check if 'scaler_columns.pkl' or 'final_model_columns.pkl' matches the test data columns needed.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred during preprocessing: {e}")
    # import traceback # Uncomment for debugging
    # st.error(traceback.format_exc()) # Uncomment for debugging
    st.stop()
# --- End Preprocessing ---


# === Streamlit UI ===
st.title("CatBoost Model Evaluation & SHAP Interpretation")

# === Predictions & Errors ===
st.write("--- Making Predictions ---")
try:
    y_pred = model.predict(X_test_processed_final)
    error = y_test.reset_index(drop=True) - y_pred # Ensure indices align if y_test index wasn't reset
    st.success("Predictions generated.")
except Exception as e:
    st.error(f"Error during prediction: {e}")
    st.stop()


st.write("--- Performance Metrics ---")
# === Metrics ===
try:
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2:.4f}")
    col2.metric("MAE", f"{mae:.2f}")
    col3.metric("RMSE", f"{rmse:.2f}")
except Exception as e:
    st.error(f"Error calculating metrics: {e}")


# === Tabs for Visualization ===
tab1, tab2, tab3 = st.tabs(["Predictions vs Actual", "Error Analysis", "SHAP Explanations"])

# === Tab 1: Predictions vs Actual ===
with tab1:
    st.header("Predictions vs Actual Revenue")
    fig_pred_actual = go.Figure()
    fig_pred_actual.add_trace(go.Scatter(x=test_dates, y=y_test, mode='lines', name='Actual Revenue'))
    fig_pred_actual.add_trace(go.Scatter(x=test_dates, y=y_pred, mode='lines', name='Predicted Revenue', line=dict(dash='dash')))
    fig_pred_actual.update_layout(xaxis_title="Date", yaxis_title="Revenue", legend_title="Legend")
    st.plotly_chart(fig_pred_actual, use_container_width=True)

# === Tab 2: Errors ===
with tab2:
    st.header("Error Analysis")
    col_err1, col_err2 = st.columns(2)

    with col_err1:
        st.subheader("Absolute Error Over Time")
        fig_err_time = go.Figure()
        fig_err_time.add_trace(go.Scatter(x=test_dates, y=np.abs(error), mode='lines', name='Absolute Error'))
        fig_err_time.update_layout(xaxis_title="Date", yaxis_title="Absolute Error")
        st.plotly_chart(fig_err_time, use_container_width=True)

    with col_err2:
        st.subheader("Error Distribution")
        fig_err_dist = go.Figure()
        fig_err_dist.add_trace(go.Histogram(x=error, nbinsx=30, name="Error Distribution"))
        fig_err_dist.update_layout(xaxis_title="Prediction Error (Actual - Predicted)", yaxis_title="Frequency")
        st.plotly_chart(fig_err_dist, use_container_width=True)

# === Tab 3: SHAP Analysis ===
with tab3:
    st.header("SHAP Analysis")
    st.write("Calculating SHAP values... (This may take a moment for larger datasets)")

    try:
        # Initialize SHAP explainer with the loaded model
        explainer = shap.TreeExplainer(model)
        # Calculate SHAP values on the *final processed* test data
        shap_values = explainer.shap_values(X_test_processed_final)

        st.success("SHAP values calculated.")

        # --- Display SHAP Plots ---
        st.subheader("Feature Importance (Mean Absolute SHAP)")
        # Use matplotlib context manager if needed for complex setups
        fig_bar, ax_bar = plt.subplots()
        # Pass the processed data for plotting feature values correctly
        shap.summary_plot(shap_values, X_test_processed_final, plot_type="bar", show=False)
        st.pyplot(fig_bar)
        plt.close(fig_bar) # Close the figure to release memory

        st.subheader("SHAP Summary Plot (Beeswarm - Feature Impact)")
        st.write("Shows how high/low values of a feature impact the prediction.")
        fig_summary, ax_summary = plt.subplots()
         # Pass the processed data for plotting feature values correctly
        shap.summary_plot(shap_values, X_test_processed_final, show=False)
        st.pyplot(fig_summary)
        plt.close(fig_summary) # Close the figure

    except Exception as e:
        st.error(f"An error occurred during SHAP analysis: {e}")
        # import traceback # Uncomment for debugging
        # st.text(traceback.format_exc()) # Uncomment for debugging
