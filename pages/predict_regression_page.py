import datetime
import pickle
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

st.set_page_config(page_title="NYC Taxi Model | Fare Regression Model")

st.header("NYC Taxi Fare ML Project")
st.subheader("Regression")

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
def load_model() -> HistGradientBoostingRegressor:
    with open("./taxi_regression_model.pkl", "rb") as data:
        model: HistGradientBoostingRegressor = pickle.load(data)
    return model


def predict() -> float:
    """
    Predict total fare amount for a taxi ride.

    This is a machine learning model that predicts the total fare
    amount for a taxi ride based on input location, date and borough
    information.

    The model uses historical NYC taxi data and borough mapping to
    make predictions based on various features like:
    - Location ID: Pickup location ID
    - Time info: Month, day, hour
    - Calendar data: Weekday, weekend, holiday status
    - Borough name
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

    model: HistGradientBoostingRegressor = load_model()

    X = X[model.feature_names_in_]
    y: np.ndarray = model.predict(X)

    prediction: float = y[0]
    return prediction


st.divider()

if st.button("Predict My Earnings!"):
    prediction: float = predict()
    st.success(f"#### Predicted Earnings: ${prediction:.2f} 💸")
