import os
import streamlit as st
from st_pages import Page, show_pages, add_page_title

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# st.set_page_config(layout="wide")

show_pages(
    [
        Page(os.path.join(current_dir, "main.py"), "Home", "🏠"),
        Page(os.path.join(current_dir, "pages/features.py"), "Feature Importance", "🔍"),
        Page(os.path.join(current_dir, "pages/shap.py"), "User-based SHAP", "📊"),
        Page(os.path.join(current_dir, "pages/predict.py"), "Probability of CHURN", "🔮"),
    ]
)

st.write("# Welcome to Churn Project! 🚀")

st.markdown(
    """
This application provides an interactive and user-friendly interface for exploring a customer churn prediction model. 
The app consists of multiple pages, each serving a specific purpose:


- Home: The main page of the app.
- Feature Importance: Displays the importance of features in predicting customer churn.
- User-based SHAP: Shows the SHAP values for a specific customer.
- Probability of CHURN: Predicts the probability of churn for a customer.
    
## Technology Stack

### Frontend: Streamlit

[Streamlit](https://streamlit.io/) is an open-source Python framework specifically designed for creating data science and machine learning applications. Its intuitive syntax and ease of use make it ideal for rapidly prototyping and deploying interactive web apps. In this project, Streamlit is employed to construct the user interface, enabling users to intuitively interact with the food product data through a web browser.

### Backend: FastAPI

[FastAPI](https://fastapi.tiangolo.com/) is a high-performance web framework built on top of Python's ASGI (Asynchronous Server Gateway Interface). It offers a streamlined approach to building APIs, emphasizing both speed and ease of development. The backend in this application leverages FastAPI to handle critical aspects like:


## Data Source

Telco Customer Churn

The Telco customer churn data contains information about a fictional telco company that provided home phone and Internet services to 7043 customers in California in Q3. It indicates which customers have left, stayed, or signed up for their service. Multiple important demographics are included for each customer, as well as a Satisfaction Score, Churn Score, and Customer Lifetime Value (CLTV) index.


    """
)
