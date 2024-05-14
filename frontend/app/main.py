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