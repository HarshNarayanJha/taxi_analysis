import streamlit as st

home_page = st.Page("pages/home_page.py", title="NYC Taxi Model | Home")
regression_page = st.Page(
    "pages/predict_regression_page.py",
    title="NYC Taxi Fare Regression",
    url_path="regression",
)
classification_page = st.Page(
    "pages/predict_classification_page.py",
    title="NYC Taxi Fare Classification",
    url_path="classification",
)

current_page = st.navigation([home_page, regression_page, classification_page])
current_page.run()
