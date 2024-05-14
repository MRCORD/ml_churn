import streamlit as st
import requests

from utils.functions import *


st.title("Telco Customer Churn Project")

# model = load_model()
# data = load_data()

# X_train = load_x_y("data/X_train.pkl")
# X_test = load_x_y("data/X_test.pkl")
# y_train = load_x_y("data/y_train.pkl")
# y_test = load_x_y("data/y_test.pkl")

backend_url = "http://backend:8000" 

response = requests.get(f"{backend_url}/v1/data")

# st.write(response.status_code)


if response.status_code == 200:
    data = response.json()
    max_values = data["max_values"]
    
    max_tenure = max_values["max_tenure"]
    max_monthly_charges = max_values["max_monthly_charges"]
    max_total_charges = max_values["max_total_charges"]

    # Retrieving data from the user
    customerID = "6464-UIAEA"
    gender = st.selectbox("Gender:", ("Female", "Male"))
    senior_citizen = st.number_input("SeniorCitizen (0: No, 1: Yes)", min_value=0, max_value=1, step=1)
    partner = st.selectbox("Partner:", ("No", "Yes"))
    dependents = st.selectbox("Dependents:", ("No", "Yes"))
    tenure = st.number_input("Tenure:", min_value=0, max_value=max_tenure, step=1)
    phone_service = st.selectbox("PhoneService:", ("No", "Yes"))
    multiple_lines = st.selectbox("MultipleLines:", ("No", "Yes"))
    internet_service = st.selectbox("InternetService:", ("No", "DSL", "Fiber optic"))
    online_security = st.selectbox("OnlineSecurity:", ("No", "Yes"))
    online_backup = st.selectbox("OnlineBackup:", ("No", "Yes"))
    device_protection = st.selectbox("DeviceProtection:", ("No", "Yes"))
    tech_support = st.selectbox("TechSupport:", ("No", "Yes"))
    streaming_tv = st.selectbox("StreamingTV:", ("No", "Yes"))
    streaming_movies = st.selectbox("StreamingMovies:", ("No", "Yes"))
    contract = st.selectbox("Contract:", ("Month-to-month", "One year", "Two year"))
    paperless_billing = st.selectbox("PaperlessBilling", ("No", "Yes"))
    payment_method = st.selectbox("PaymentMethod:", ("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"))
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=max_monthly_charges, step=0.01)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=max_total_charges, step=0.01)

    # Confirmation button
    confirmation_button = st.button("Confirm")



    # When the confirmation button is clicked
    if confirmation_button:
        # Convert user-entered data into a data frame
        new_customer_data = {
            "customerID": customerID,
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }

        # # Predict churn probability using the model
        # churn_probability = model.predict_proba(new_customer_data)[:, 1]
        
        # st.json(new_customer_data)
        
        response = requests.post(f"{backend_url}/v1/predict", json=new_customer_data)
        
        # st.write(response.status_code)

        
        if response.status_code == 200:
            data = response.json()
                        
            if "churn_probability" in data:
                churn_probability = data["churn_probability"]
                
                # st.write(churn_probability)

                # Format churn probability
                formatted_churn_probability = "{:.2%}".format(churn_probability)

                big_text = f"<h1>Churn Probability: {formatted_churn_probability}</h1>"
                st.markdown(big_text, unsafe_allow_html=True)
                st.write(new_customer_data)