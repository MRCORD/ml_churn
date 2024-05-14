import streamlit as st
import requests

from utils.functions import *

# st.set_page_config(
#     layout="wide")

st.title("Telco Customer Churn Project")

# model = load_model()
# data = load_data()

# X_train = load_x_y("data/X_train.pkl")
# X_test = load_x_y("data/X_test.pkl")
# y_train = load_x_y("data/y_train.pkl")
# y_test = load_x_y("data/y_test.pkl")

# max_tenure = data['tenure'].max()
# max_monthly_charges = data['MonthlyCharges'].max()
# max_total_charges = data['TotalCharges'].max()


# summary(model, data, X_train, X_test)

backend_url = "http://backend:8000"

response = requests.get(f"{backend_url}/v1/features")

# st.write(response.status_code)

if response.status_code == 200:
    data = response.json()

    if "summary_fig" in data:
        st.pyplot(decode_fig_from_base64(data["summary_fig"]))
    else:
        st.write(data["error"])