import streamlit as st
import requests

from utils.functions import *



st.write("# User-based SHAP")
st.write("This page enables to select a specific customer and visualize the SHAP values for that customer, helping understand the model’s prediction rationale.")

backend_url = "http://backend:8000" 

response = requests.get(f"{backend_url}/v1/pickles_load")

if response.status_code == 200:
    data = response.json()
    X_train = pd.read_json(data["X_train"], orient="split")
    X_test = pd.read_json(data["X_test"], orient="split")
    y_train = pd.read_json(data["y_train"], orient="split", typ='series')
    y_test = pd.read_json(data["y_test"], orient="split", typ='series')
    
    available_customer_ids = X_test['customerID'].tolist()

    # Customer ID text input
    customer_id = st.selectbox("Choose the Customer", available_customer_ids)
    customer_index = X_test[X_test['customerID'] == customer_id].index[0]
    st.write(f'Customer {customer_id}: Actual value for the Customer Churn : {y_test.iloc[customer_index]}')
    
    response = requests.get(f"{backend_url}/v1/model_consult")
    
    #st.write(response.status_code)

    if response.status_code == 200:
        data = response.json()
        y_pred = data["y_pred"]
        
        st.write(f'Customer {customer_id}: CatBoost Model\'s prediction for the Customer Churn : {y_pred[customer_index]}')


    response = requests.post(f"{backend_url}/v1/shap", json={"customer_id": customer_id})
    
    # st.write(response.status_code)

    if response.status_code == 200:
        data = response.json()

        if "shap_plot" in data:
            st.pyplot(decode_fig_from_base64(data["shap_plot"]))
            st.pyplot(decode_fig_from_base64(data["waterfall_plot"]))
            
        else:
            st.write(data["error"])


        # plot_shap(model, data, customer_id, X_train=X_train, X_test=X_test)
