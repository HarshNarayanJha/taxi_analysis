# %% [markdown]
# ## New York Taxi Analysis Data Analysis / Machine Learning project
# ### 1. Import Libraries

# %% 1. Import Libraries
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# %% [markdown]
# ### 2. Import Data

# %% 2. Import Data
taxi_oct_2024 = pd.read_parquet("./data/yellow_tripdata_2024-10.parquet")
taxi_data = pd.concat([taxi_oct_2024])
taxi_data.head()
taxi_data.shape

# %% [markdown]
# ### 3. Data Exploration

# %% 3. Data Exploration
taxi_data.columns
taxi_data = pd.DataFrame(
    taxi_data[
        [
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "passenger_count",
            "trip_distance",
            "RatecodeID",
            "PULocationID",
            "DOLocationID",
            "payment_type",
            "total_amount",
        ]
    ]
)

# %%
taxi_data.head()

# %%
taxi_data.hist(figsize=(20, 10), bins=60)

# %%
taxi_data["RatecodeID"].value_counts()

# %%
taxi_data.reset_index().plot(kind="scatter", y="total_amount", x="index", figsize=(10, 5))

# %% [markdown]
"""
Two things to deal with: -ve values and outliers (very high values)
"""

# %%
print(taxi_data[taxi_data["total_amount"] < 0].shape)
taxi_data[taxi_data["total_amount"] < 0].reset_index().plot(kind="scatter", y="total_amount", x="index", figsize=(10, 5))

# %%
print(taxi_data[taxi_data["total_amount"] < 0].head())
print(taxi_data[taxi_data["total_amount"] < 0]["payment_type"].value_counts())  # type: ignore
taxi_data[taxi_data["total_amount"] < 0]["trip_distance"].hist(figsize=(10, 5), bins=60)  # type: ignore

# %% [markdown]
"""
Since most of the -ve fared trips are of type 3 and 4 (No Charge and Dispute) and of trip_distance 0, we can safely ignore them.
"""

# %%
print(taxi_data[taxi_data["total_amount"] == 0].shape)
print(taxi_data[taxi_data["total_amount"] == 0].head())
print(taxi_data[taxi_data["total_amount"] == 0]["payment_type"].value_counts())  # type: ignore
taxi_data[taxi_data["total_amount"] == 0]["trip_distance"].hist(  # type: ignore
    figsize=(10, 5), bins=60
)
taxi_data[taxi_data["total_amount"] == 0].reset_index().plot(kind="scatter", y="total_amount", x="index", figsize=(10, 5))

# %% [markdown]
"""
Hence we can safely ignore -ves and 0's. But what about very high values?
"""

# %%
print(taxi_data[taxi_data["total_amount"] > 250].shape)
print(taxi_data["total_amount"].mean())

# %% [markdown]
"""
Keep Total Amount less than $250
"""

# %% 4. Data Cleaning
taxi_data_filtered = taxi_data[taxi_data["total_amount"].between(0, 250)]
print(taxi_data.shape)
print(taxi_data_filtered.shape)

# %%
### check for missing values
print(taxi_data_filtered.shape)
print(taxi_data_filtered.isna().sum())
taxi_data_filtered = taxi_data_filtered.dropna()
print(taxi_data_filtered.shape)

# %% [markdown]
# ### 5. Data Preperation

# %% 5. Data Preperation

# make a copy of it
taxi_data_prepared = taxi_data_filtered.copy()

# %%

# check types
print(taxi_data_prepared.dtypes)

# fix types
taxi_data_prepared.loc[:, "tpep_pickup_datetime"] = pd.to_datetime(taxi_data_prepared.loc[:, "tpep_pickup_datetime"])
taxi_data_prepared.loc[:, "tpep_dropoff_datetime"] = pd.to_datetime(taxi_data_prepared.loc[:, "tpep_dropoff_datetime"])

taxi_data_prepared.loc[:, "passenger_count"] = taxi_data_prepared["passenger_count"].astype(int)
taxi_data_prepared.loc[:, "RatecodeID"] = taxi_data_prepared["RatecodeID"].astype(str)
taxi_data_prepared.loc[:, "PULocationID"] = taxi_data_prepared["PULocationID"].astype(str)
taxi_data_prepared.loc[:, "DOLocationID"] = taxi_data_prepared["DOLocationID"].astype(str)
taxi_data_prepared.loc[:, "payment_type"] = taxi_data_prepared["payment_type"].astype(str)
print()
print(taxi_data_prepared.dtypes)

# %%

# we transform values into formats we need

taxi_data_prepared["transaction_date"] = pd.to_datetime(
    taxi_data_prepared["tpep_pickup_datetime"].dt.date  # type: ignore
)
taxi_data_prepared["transaction_year"] = taxi_data_prepared["tpep_pickup_datetime"].dt.year  # type: ignore
taxi_data_prepared["transaction_month"] = taxi_data_prepared["tpep_pickup_datetime"].dt.month  # type: ignore
taxi_data_prepared["transaction_day"] = taxi_data_prepared["tpep_pickup_datetime"].dt.day  # type: ignore
taxi_data_prepared["transaction_hour"] = taxi_data_prepared["tpep_pickup_datetime"].dt.hour  # type: ignore

taxi_data_prepared.head()
taxi_data_prepared.hist(figsize=(20, 10), bins=60)

# %%

# fix year and month to 2024 10

taxi_data_prepared = taxi_data_prepared[taxi_data_prepared["transaction_year"] == 2024]
taxi_data_prepared = pd.DataFrame(taxi_data_prepared[taxi_data_prepared["transaction_month"] == 10])

taxi_data_prepared.hist(figsize=(20, 10), bins=60)

# %% [markdown]
"""
trip_distance too contains humongous values, guess we can remove > 100 (only 22 records)
"""

# %%

taxi_data_prepared["trip_distance"].hist(figsize=(10, 5), bins=60)
print(taxi_data_prepared[taxi_data_prepared["trip_distance"] > 100].shape)
# print(taxi_data_prepared["trip_distance"].nsmallest(n=100))

# %%

# filter out on trip_distance

print(taxi_data_prepared.shape)
taxi_data_prepared = taxi_data_prepared[taxi_data_prepared["trip_distance"].between(0, 100)]
print(taxi_data_prepared.shape)
# %%

# Note down categorical and numerical features
categorical_columns = [
    "PULocationID",
    "transaction_date",
    "transaction_month",
    "transaction_day",
    "transaction_hour",
]
numerical_columns = ["trip_distance", "total_amount"]
all_needed_columns = categorical_columns + numerical_columns

# %%

main_taxi_df: pd.DataFrame = pd.DataFrame(taxi_data_prepared[all_needed_columns])
print(main_taxi_df.shape)
main_taxi_df.head()

# %% [markdown]
"""
Now, Aggregate data points
Now it's a good time to think about what we want to predict. We need to transform our data to a certain format
"""

# %%

taxi_grouped_by_region = main_taxi_df.groupby(categorical_columns).mean().reset_index()
taxi_grouped_by_region["count_of_transactions"] = main_taxi_df.groupby(categorical_columns).count().reset_index()["total_amount"]
print(taxi_grouped_by_region.shape)
taxi_grouped_by_region.head()

# %%

taxi_grouped_by_region["trip_distance"].hist(bins=100, figsize=(10, 5))

# %%

taxi_grouped_by_region["total_amount"].hist(bins=100, figsize=(10, 5))

# %% [markdown]
# ### 6. Benchmark Model

# %% 6. Benchmark Model
data_from_benchmark_model = taxi_grouped_by_region.copy()

# %%
categorical_features_benchmark = [
    "PULocationID",
    "transaction_month",
    "transaction_day",
    "transaction_hour",
]
# do NOT include trip_distance here, since it directly coorelates with total_amount, and that's cheating actually!
input_features_benchmark = categorical_features_benchmark
target_feature_benchmark = "total_amount"

# %% [markdown]
# #### 6.1. Train Test Split

# %% 6.1. Train Test Split

X_bench = data_from_benchmark_model[input_features_benchmark]
y_bench = data_from_benchmark_model[target_feature_benchmark]

# one-hot encode
X_bench = pd.get_dummies(X_bench)

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_bench, y_bench, test_size=0.2, random_state=50)

# %% [markdown]
# #### 6.2. Fit a model into the data

# %% 6.2. Fit a model into the data

tree = DecisionTreeRegressor(max_depth=10)
tree.fit(X_train_b, y_train_b)

# %% [markdown]
# #### 6.3. Model evalution

# %% 6.3. Model evalution

model_at_hand = tree

y_pred_b = model_at_hand.predict(X_test_b)

print("mean absolute error", mean_absolute_error(y_test_b, y_pred_b))
print("mean squared error", mean_squared_error(y_test_b, y_pred_b))
print("root mean squared error", np.sqrt(mean_squared_error(y_test_b, y_pred_b)))
print("r2", r2_score(y_test_b, y_pred_b))

# %%
data = {"true": y_test_b, "pred": y_pred_b}
results = pd.DataFrame(data)

results.plot(figsize=(15, 8), kind="scatter", x="true", y="pred")

# %% [markdown]
# ### 7. Feature Engineering

# %% 7. Feature Engineering

data_with_new_features: pd.DataFrame = taxi_grouped_by_region.copy()

# %% [markdown]
# #### 7.1 Date-related features

# %% 7.1 Date-related features
data_with_new_features["transaction_week_day"] = data_with_new_features["transaction_date"].dt.weekday
data_with_new_features["weekend"] = data_with_new_features["transaction_week_day"].apply(lambda x: True if x in (5, 6) else False)

# %%

from pandas.tseries.holiday import USFederalHolidayCalendar  # noqa: E402

cal = USFederalHolidayCalendar()
holidays = pd.to_datetime(cal.holidays(start="2024", end="2025").date)  # type: ignore
data_with_new_features["is_holiday"] = data_with_new_features["transaction_date"].isin(holidays)

print(data_with_new_features["weekend"].value_counts())
print(data_with_new_features["is_holiday"].value_counts())
data_with_new_features.head()

# %% [markdown]
# #### 7.2 Borough Information

# %% 7.2 Borough Information

zone_lookup = pd.read_csv("data/taxi_zone_lookup.csv")
zone_lookup = zone_lookup[["LocationID", "Borough"]]
zone_lookup["LocationID"] = zone_lookup["LocationID"].astype(str)
zone_lookup.head()

# %%
data_with_new_features = data_with_new_features.merge(zone_lookup, left_on="PULocationID", right_on="LocationID", how="left")
data_with_new_features.drop("LocationID", axis=1, inplace=True)
data_with_new_features.head()

# %%
data_with_new_features["Borough"].value_counts()
# %% [markdown]
# #### 7.3 Weather related information

# %% 7.3 Weather related information

# Don't have the weather dataset

# %% [markdown]
# ### 8. Model Training

# %% 8. Model Training

data_for_model = data_with_new_features.copy()

# %%

categorical_features = [
    "PULocationID",
    "transaction_month",
    "transaction_day",
    "transaction_hour",
    "transaction_week_day",
    "weekend",
    "is_holiday",
    "Borough",
]
input_features = categorical_features
target_feature = "total_amount"

# %% [markdown]
# #### 8.1 Train Test Split

# %% 8.1 Train Test Split

X = data_for_model[input_features]
y = data_for_model[target_feature]

# one-hot encode
X = pd.get_dummies(X)
print(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=80)

# %% [markdown]
# ### 8.2 Decision Tree

# %% 8.2 Decision Tree

tree = DecisionTreeRegressor(max_depth=10)
tree.fit(X_train, y_train)

# %%
model_at_hand = tree

y_pred = model_at_hand.predict(X_test)

print("mean absolute error", mean_absolute_error(y_test, y_pred))
print("mean squared error", mean_squared_error(y_test, y_pred))
print("root mean squared error", np.sqrt(mean_squared_error(y_test, y_pred)))
print("r2", r2_score(y_test, y_pred))

# %%
data = {"true": y_test_b, "pred": y_pred_b}
results = pd.DataFrame(data)

results.plot(figsize=(20, 10), kind="scatter", x="true", y="pred")

# %% [markdown]
# #### 8.3 Random Forest

# %% 8.3 Random Forest

model = RandomForestRegressor()
model.fit(X_train, y_train)

# %%

model_at_hand = model

y_pred = model_at_hand.predict(X_test)

print("mean absolute error", mean_absolute_error(y_test, y_pred))
print("mean squared error", mean_squared_error(y_test, y_pred))
print("root mean squared error", np.sqrt(mean_squared_error(y_test, y_pred)))
print("r2", r2_score(y_test, y_pred))

# %%
data = {"true": y_test_b, "pred": y_pred_b}
results = pd.DataFrame(data)

results.plot(figsize=(20, 10), kind="scatter", x="true", y="pred")

# %% [markdown]
# #### 8.4 Gradient Boosting

# %% 8.4 Gradient Boosting

model = HistGradientBoostingRegressor()
model.fit(X_train, y_train)

# %%

model_at_hand = model

y_pred = model_at_hand.predict(X_test)

print("mean absolute error", mean_absolute_error(y_test, y_pred))
print("mean squared error", mean_squared_error(y_test, y_pred))
print("root mean squared error", np.sqrt(mean_squared_error(y_test, y_pred)))
print("r2", r2_score(y_test, y_pred))

# %%
data = {"true": y_test_b, "pred": y_pred_b}
results = pd.DataFrame(data)

results.plot(figsize=(20, 10), kind="scatter", x="true", y="pred")

# %% [markdown]
# #### 8.5 Comparing Algorithm Performances
#
# | Algorithm             |       MAE |      RMSE |        R2 |
# |:----------------------|----------:|----------:|----------:|
# | Benchmark Model       |     12.61 |     18.90 |      0.23 |
# | Decision Tree         |     10.78 |     17.64 |      0.36 |
# | Random Forest         |      9.82 |     17.04 |      0.40 |
# | Hist Gradient Boosting|      9.62 |     15.95 |      0.47 |
#

# %% [markdown]
# #### 9. Save the Model

# %% Save the Best Model (For Now)

with open("taxi_regression_model.pkl", "wb") as file:
    pickle.dump(model_at_hand, file)

# %% [markdown]
# ### 10. Tuning

# %% 10. Tuning

# %% [markdown]
# #### 10.1 Find Best Parameters

# %% 10.1 Find Best Parameters
# Will do these. They take time!

# %% [markdown]
# ### 11. Classification

# %% 11. Classification
# Yes, we will turn this into a classification problem
data_with_new_features["total_amount"].hist(bins=100, figsize=(10, 5))

# %%

nyc_class = data_with_new_features.copy()
# a 20 split point is good enough!
nyc_class["earning_class"] = data_with_new_features["total_amount"].apply(lambda x: "low" if x <= 25 else "high")
nyc_class["earning_class_binary"] = nyc_class["earning_class"].apply(lambda x: 0 if x == "low" else 1)
nyc_class.head()


# %%
nyc_class["earning_class"].value_counts()

# %%

categorical_features = [
    "PULocationID",
    "transaction_month",
    "transaction_day",
    "transaction_hour",
    "transaction_week_day",
    "weekend",
    "is_holiday",
    "Borough",
]
input_features = categorical_features
target_feature = "earning_class_binary"

# %%

X_c = nyc_class[input_features]
y_c = nyc_class[target_feature]

# one-hot encode
X_c = pd.get_dummies(X_c)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.33, random_state=100)

# %%

clf = RandomForestClassifier()
clf.fit(X_train_c, y_train_c)

# %%

y_pred_c = clf.predict(X_test_c)

print(confusion_matrix(y_test_c, y_pred_c))
print("accuracy", accuracy_score(y_test_c, y_pred_c))
print("precision", precision_score(y_test_c, y_pred_c))
print("recall", recall_score(y_test_c, y_pred_c))

# %% [markdown]
# ### 12. Save Classification Model

# %% 12. Save Classification Model

with open("taxi_classification_model.pkl", "wb") as file:
    pickle.dump(clf, file)
