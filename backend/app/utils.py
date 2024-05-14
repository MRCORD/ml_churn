import requests
import os
import pandas as pd
import joblib
import shap
from matplotlib import pyplot as plt
from catboost import CatBoostClassifier, Pool
import logging

import io
import base64

from app.models import *

# Directory of utils.py
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables
MODEL_PATH = os.path.join(UTILS_DIR, os.pardir, os.getenv("MODEL_PATH"))
DATA_PATH = os.path.join(UTILS_DIR, os.pardir, os.getenv("DATA_PATH"))


# Create a logger for the utils module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the log level to INFO

# Create a file handler and set its log level
file_handler = logging.FileHandler('utils.log')  # Create or append to the 'utils.log' file
file_handler.setLevel(logging.INFO)

# Define a formatter for file output
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)


def load_data():
    try:
        data = pd.read_parquet(DATA_PATH)
        logger.info("Data loaded")
        return data
    except Exception as e:
        logger.error(f"An error occurred in load_data(): {e}")
        raise

def load_x_y(file_path):
    try:
        data = joblib.load(file_path)
        data.reset_index(drop=True, inplace=True)
        logger.info(f"Data loaded from {file_path}")
        return data
    except Exception as e:
        logger.error(f"An error occurred in load_x_y(): {e}")
        raise

def load_model():
    try:
        model = CatBoostClassifier()
        model.load_model(MODEL_PATH)
        logger.info("Model loaded")
        return model
    except Exception as e:
        logger.error(f"An error occurred in load_model(): {e}")
        raise

def compute_max_values(data):
    try:
        max_tenure = int(data['tenure'].max())
        max_monthly_charges = float(data['MonthlyCharges'].max())
        max_total_charges = float(data['TotalCharges'].max())            
        max_values = {
            'max_tenure': max_tenure,
            'max_monthly_charges': max_monthly_charges,
            'max_total_charges': max_total_charges
        }
        logger.info("Max values computed")
        return max_values
    
    except Exception as e:
        logger.error(f"An error occurred in compute_max_values(): {e}")
        raise
    
def get_churn_probability(data, model):
    try:
        # Convert incoming data into a DataFrame
        dataframe = pd.DataFrame.from_dict(data, orient='index').T
        # Make the prediction
        churn_probability = model.predict_proba(dataframe)[0][1]
        logger.info(f"Churn probability calculated: {churn_probability}")
        return churn_probability
    except Exception as e:
        logger.error(f"An error occurred in get_churn_probability(): {e}")
        raise
    
    
def calculate_shap(model, X_train, X_test):
    try:
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values_cat_train = explainer.shap_values(X_train)
        shap_values_cat_test = explainer.shap_values(X_test)
        logger.info("SHAP values calculated")
        return explainer, shap_values_cat_train, shap_values_cat_test    
    except Exception as e:
        logger.error(f"An error occurred in calculate_shap(): {e}")
        raise

def display_shap_summary(shap_values_cat_train, X_train):
    try:
        # Create the plot summarizing the SHAP values
        shap.summary_plot(shap_values_cat_train, X_train, plot_type="bar", plot_size=(12,12))
        summary_fig, _ = plt.gcf(), plt.gca()
        logger.info("SHAP summary displayed")
        return summary_fig
    except Exception as e:
        logger.error(f"An error occurred in display_shap_summary(): {e}")
        raise
    
def summary(model, data, X_train, X_test):
    try:
        # Calculate SHAP values
        explainer, shap_values_cat_train, shap_values_cat_test = calculate_shap(model, X_train, X_test)

        # Summarize and visualize SHAP values
        summary_fig = display_shap_summary(shap_values_cat_train, X_train)
        logger.info("SHAP summary DONE!")
        return summary_fig
    except Exception as e:
        logger.error(f"An error occurred in plot_shap(): {e}")
        raise



def plot_shap_values(model, explainer, shap_values_cat_train, shap_values_cat_test, customer_id, X_test, X_train):
    try:
        # Check if customerID exists in X_test
        if customer_id in X_test['customerID'].values:
            # Visualize SHAP values for a specific customer
            customer_index = X_test[X_test['customerID'] == customer_id].index[0]
            fig, ax_2 = plt.subplots(figsize=(6,6), dpi=200)
            shap.decision_plot(explainer.expected_value, shap_values_cat_test[customer_index], X_test[X_test['customerID'] == customer_id], link="logit")
            return fig
        else:
            logger.error(f"Customer ID {customer_id} does not exist in X_test")
            return None
    except Exception as e:
        logger.error(f"An error occurred in plot_shap_values(): {e}")
        raise

def display_shap_waterfall_plot(explainer, expected_value, shap_values, feature_names, max_display=20):
    try:
        # Create SHAP waterfall drawing
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        shap.plots._waterfall.waterfall_legacy(expected_value, shap_values, feature_names=feature_names, max_display=max_display, show=False)
        # st.pyplot(fig)
        # plt.close()
        return fig
    except Exception as e:
        logger.error(f"An error occurred in display_shap_waterfall_plot(): {e}")
        raise

def plot_shap(model, data, customer_id, X_train, X_test):
    try:
        # Calculate SHAP values
        explainer, shap_values_cat_train, shap_values_cat_test = calculate_shap(model, X_train, X_test)
        shap_plot = plot_shap_values(model, explainer, shap_values_cat_train, shap_values_cat_test, customer_id, X_test, X_train)

        # Waterfall
        customer_index = X_test[X_test['customerID'] == customer_id].index[0]
        waterfall_plot = display_shap_waterfall_plot(explainer, explainer.expected_value, shap_values_cat_test[customer_index], feature_names=X_test.columns, max_display=20)
        
        return shap_plot, waterfall_plot
    except Exception as e:
        logger.error(f"An error occurred in plot_shap(): {e}")
        raise
    
def encode_fig_to_base64(fig):
    """
    Encode a matplotlib figure to a base64 string.

    Parameters:
    fig (matplotlib.figure.Figure): The figure to encode.

    Returns:
    str: The base64 encoded string.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    encoded_fig = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)  # Close the figure to free up memory
    return encoded_fig