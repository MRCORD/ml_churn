import streamlit as st
import requests

from utils.functions import *

# st.set_page_config(
#     layout="wide")


st.write("# Feature Importance")

st.write("This page shows the summary of feature importance based on SHAP values.")


backend_url = "http://backend:8000"

response = requests.get(f"{backend_url}/v1/features")

# st.write(response.status_code)

if response.status_code == 200:
    data = response.json()

    if "summary_fig" in data:
        st.pyplot(decode_fig_from_base64(data["summary_fig"]))
    else:
        st.write(data["error"])