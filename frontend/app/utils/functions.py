import shap
import pandas as pd
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from pyarrow import parquet as pq
from catboost import CatBoostClassifier, Pool
import joblib
import base64
from PIL import Image
import io


# Path of the trained model and data
MODEL_PATH = "model/catboost_model.cbm" 
DATA_PATH = "data/churn_data_regulated.parquet"


@st.cache_resource
def load_data():
    data = pd.read_parquet(DATA_PATH)
    return data

def load_x_y(file_path):
    data = joblib.load(file_path)
    data.reset_index(drop=True, inplace=True)
    return data

def load_model():
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model
    

def calculate_shap(model, X_train, X_test):
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values_cat_train = explainer.shap_values(X_train)
    shap_values_cat_test = explainer.shap_values(X_test)
    return explainer, shap_values_cat_train, shap_values_cat_test

def plot_shap_values(model, explainer, shap_values_cat_train, shap_values_cat_test, customer_id, X_test, X_train):
    # Visualize SHAP values for a specific customer
    customer_index = X_test[X_test['customerID'] == customer_id].index[0]
    fig, ax_2 = plt.subplots(figsize=(6,6), dpi=200)
    shap.decision_plot(explainer.expected_value, shap_values_cat_test[customer_index], X_test[X_test['customerID'] == customer_id], link="logit")
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=X_test.columns, y=shap_values_cat_test[customer_index], mode='markers', marker=dict(color=shap_values_cat_test[customer_index], colorscale='Viridis')))
    # fig.update_layout(title='SHAP Values for Customer {}'.format(customer_id), xaxis_title='Features', yaxis_title='SHAP Values')
    # st.plotly_chart(fig)
    st.pyplot(fig)
    plt.close()

def display_shap_summary(shap_values_cat_train, X_train):
    # Create the plot summarizing the SHAP values
    shap.summary_plot(shap_values_cat_train, X_train, plot_type="bar", plot_size=(12,12))
    summary_fig, _ = plt.gcf(), plt.gca()
    st.pyplot(summary_fig)
    plt.close()

def display_shap_waterfall_plot(explainer, expected_value, shap_values, feature_names, max_display=20):
    # Create SHAP waterfall drawing
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    shap.plots._waterfall.waterfall_legacy(expected_value, shap_values, feature_names=feature_names, max_display=max_display, show=False)
    st.pyplot(fig)
    plt.close()

def summary(model, data, X_train, X_test):
    # Calculate SHAP values
    explainer, shap_values_cat_train, shap_values_cat_test = calculate_shap(model, X_train, X_test)

    # Summarize and visualize SHAP values
    display_shap_summary(shap_values_cat_train, X_train)

def plot_shap(model, data, customer_id, X_train, X_test):
    # Calculate SHAP values
    explainer, shap_values_cat_train, shap_values_cat_test = calculate_shap(model, X_train, X_test)
    
    # Visualize SHAP values
    plot_shap_values(model, explainer, shap_values_cat_train, shap_values_cat_test, customer_id, X_test, X_train)

    # Waterfall
    customer_index = X_test[X_test['customerID'] == customer_id].index[0]
    display_shap_waterfall_plot(explainer, explainer.expected_value, shap_values_cat_test[customer_index], feature_names=X_test.columns, max_display=20)

def decode_fig_from_base64(base64_string):
    base64_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(base64_bytes))
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    return fig