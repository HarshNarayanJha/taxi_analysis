import streamlit as st

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

if __name__ == "__page__":
    st.set_page_config(page_title="NYC Taxi Model")

    st.title("NYC Taxi Fare ML Project")
    st.text("Description about the project")
    st.text("Choose Regression or Classification from the sidebar.")

    st.page_link(regression_page, label="Try out Regression on this problem")
    st.page_link(classification_page, label="Try out Classification on this problem")
