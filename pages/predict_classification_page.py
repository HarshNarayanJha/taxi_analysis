import datetime
import pickle
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="NYC Taxi Model | Fare Classification Model")

st.header("NYC Taxi Fare ML Project")
st.subheader("Classification")

st.text("Choose your inputs")

# Input Choices
borough_choices: list[str] = [
    "Manhattan",
    "Brooklyn",
    "Queens",
    "Bronx",
    "EWR",
    "Staten Island",
    "Unknown",
]

location_ids_choices: list[int] = list(range(1, 266))
location_ids_choices.remove(103)
location_ids_choices.remove(104)
location_ids_choices.remove(110)
location_ids_choices.remove(99)

months_choices: list[int] = list(range(1, 13))
days_choices: list[int] = list(range(1, 32))
hours_choices: list[int] = list(range(24))

# Get Inputs
loc_id: int = st.selectbox("Location ID", location_ids_choices)
month: int = st.selectbox("Month", months_choices)
day: int = st.selectbox("Day of Month", days_choices)
hour: int = st.selectbox("Hour of Day", hours_choices)
holiday: bool = st.checkbox("Is Holiday")
borough: str = st.selectbox("Borough", borough_choices)

weekday: int = datetime.datetime(day=day, month=month, year=2024).weekday()
weekend: bool = datetime.datetime(day=day, month=month, year=2024).weekday() in (5, 6)


@st.cache_resource
def load_model() -> RandomForestClassifier:
    with open("./taxi_classification_model.pkl", "rb") as data:
        model: RandomForestClassifier = pickle.load(data)
    return model


def predict() -> float:
    """
    Predict if the given inputs lead to Low or High Earning Class.

    Returns:
        int: Predicted earnings class (0 for Low earnings, 1 for High earnings)
    """
    X = pd.DataFrame(
        [[loc_id, month, day, hour, weekday, weekend, holiday, borough]],
        columns=np.array(
            [
                "PULocationID",
                "transaction_month",
                "transaction_day",
                "transaction_hour",
                "transaction_week_day",
                "weekend",
                "is_holiday",
                "Borough",
            ]
        ),
    )

    # Transform Input Data
    X = pd.get_dummies(X)

    dummies_locations = location_ids_choices.copy()
    dummies_locations_columns = [f"PULocationID_{x}" for x in dummies_locations]

    dummies_borough = borough_choices.copy()
    dummies_borough.remove(borough)
    dummies_borough_columns = [f"Borough_{x}" for x in dummies_borough]

    X = pd.concat(
        [
            X,
            pd.DataFrame(
                [[False] * len(dummies_locations)],
                columns=np.array(dummies_locations_columns),
            ),
        ],
        axis=1,
    )

    X.loc[:, f"PULocationID_{loc_id}"] = True

    X = pd.concat(
        [
            X,
            pd.DataFrame(
                [[False] * len(dummies_borough)],
                columns=np.array(dummies_borough_columns),
            ),
        ],
        axis=1,
    )

    X = X.drop("PULocationID", axis=1)

    model: RandomForestClassifier = load_model()

    X = X[model.feature_names_in_]
    y: np.ndarray = model.predict(X)

    prediction: float = y[0]
    return prediction


st.divider()

if st.button("Predict My Earnings!"):
    prediction: float = predict()
    st.success(f"#### Predicted Earning Class: {'Low' if prediction == 0 else 'High'}")
