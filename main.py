# %% 1. Import Libraries
import pandas as pd

# %% 2. Import Data
taxi_oct_2024 = pd.read_parquet("./data/yellow_tripdata_2024-10.parquet")
taxi_data = pd.concat([taxi_oct_2024])
taxi_data.head()
taxi_data.shape

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

# %% Cell 4
taxi_data.head()

# %% Cell 5
taxi_data.hist(figsize=(20, 10), bins=60)

# %% Cell 6
taxi_data["RatecodeID"].value_counts()

# %% Cell 7
taxi_data.reset_index().plot(
    kind="scatter", y="total_amount", x="index", figsize=(10, 5)
)

### Two things to deal with: -ve values and outliers (very high values)

# %% Cell 8
print(taxi_data[taxi_data["total_amount"] < 0].shape)
taxi_data[taxi_data["total_amount"] < 0].reset_index().plot(
    kind="scatter", y="total_amount", x="index", figsize=(10, 5)
)

# %% Cell 9
print(taxi_data[taxi_data["total_amount"] < 0].head())
print(taxi_data[taxi_data["total_amount"] < 0]["payment_type"].value_counts())
taxi_data[taxi_data["total_amount"] < 0]["trip_distance"].hist(figsize=(10, 5), bins=60)

### Since most of the -ve fared trips are of type 3 and 4 (No Charge and Dispute) and of trip_distance 0, we can safely ignore them.

# %% Cell 10
print(taxi_data[taxi_data["total_amount"] == 0].shape)
print(taxi_data[taxi_data["total_amount"] == 0].head())
print(taxi_data[taxi_data["total_amount"] == 0]["payment_type"].value_counts())
taxi_data[taxi_data["total_amount"] == 0]["trip_distance"].hist(
    figsize=(10, 5), bins=60
)
taxi_data[taxi_data["total_amount"] == 0].reset_index().plot(
    kind="scatter", y="total_amount", x="index", figsize=(10, 5)
)
### Hence we can safely ignore -ves and 0's. But what about very high values?

# %% Cell 11
print(taxi_data[taxi_data["total_amount"] > 250].shape)
print(taxi_data["total_amount"].mean())

### Keep Total Amount less than $250

# %% Cell 4. Data Cleaning
taxi_data_filtered = taxi_data[taxi_data["total_amount"].between(0, 250)]
print(taxi_data.shape)
print(taxi_data_filtered.shape)

# %% Cell 13
### check for missing values
print(taxi_data_filtered.shape)
print(taxi_data_filtered.isna().sum())
taxi_data_filtered = taxi_data_filtered.dropna()
print(taxi_data_filtered.shape)

# %% Cell 5. Data Preperation

# make a copy of it
taxi_data_prepared = taxi_data_filtered.copy()

# %% Cell 15

# check types
print(taxi_data_prepared.dtypes)

# fix types
taxi_data_prepared.loc[:, "tpep_pickup_datetime"] = pd.to_datetime(
    taxi_data_prepared.loc[:, "tpep_pickup_datetime"]
)
taxi_data_prepared.loc[:, "tpep_dropoff_datetime"] = pd.to_datetime(
    taxi_data_prepared.loc[:, "tpep_dropoff_datetime"]
)

taxi_data_prepared.loc[:, "passenger_count"] = taxi_data_prepared[
    "passenger_count"
].astype(int)
taxi_data_prepared.loc[:, "RatecodeID"] = taxi_data_prepared["RatecodeID"].astype(str)
taxi_data_prepared.loc[:, "PULocationID"] = taxi_data_prepared["PULocationID"].astype(
    str
)
taxi_data_prepared.loc[:, "DOLocationID"] = taxi_data_prepared["DOLocationID"].astype(
    str
)
taxi_data_prepared.loc[:, "payment_type"] = taxi_data_prepared["payment_type"].astype(
    str
)
print()
print(taxi_data_prepared.dtypes)

# %% Cell 16

# we transform values into formats we need

taxi_data_prepared["transaction_date"] = pd.to_datetime(
    taxi_data_prepared["tpep_pickup_datetime"].dt.date
)
taxi_data_prepared["transaction_year"] = taxi_data_prepared[
    "tpep_pickup_datetime"
].dt.year
taxi_data_prepared["transaction_month"] = taxi_data_prepared[
    "tpep_pickup_datetime"
].dt.month
taxi_data_prepared["transaction_day"] = taxi_data_prepared[
    "tpep_pickup_datetime"
].dt.day
taxi_data_prepared["transaction_hour"] = taxi_data_prepared[
    "tpep_pickup_datetime"
].dt.hour

taxi_data_prepared.head()
taxi_data_prepared.hist(figsize=(20, 10), bins=60)

# %% Cell 17

# fix year and month to 2024 10

taxi_data_prepared = taxi_data_prepared[taxi_data_prepared["transaction_year"] == 2024]
taxi_data_prepared = pd.DataFrame(
    taxi_data_prepared[taxi_data_prepared["transaction_month"] == 10]
)

taxi_data_prepared.hist(figsize=(20, 10), bins=60)

# %% Cell 18

# trip_distance too contains humongous values, guess we can remove > 100 (only 22 records)
taxi_data_prepared["trip_distance"].hist(figsize=(10, 5), bins=60)
print(taxi_data_prepared[taxi_data_prepared["trip_distance"] > 100].shape)
# print(taxi_data_prepared["trip_distance"].nsmallest(n=100))

# %% Cell 19

# filter out on trip_distance

print(taxi_data_prepared.shape)
taxi_data_prepared = taxi_data_prepared[
    taxi_data_prepared["trip_distance"].between(0, 100)
]
print(taxi_data_prepared.shape)
# %% Cell 20

# Note down categorical and numerical features
categorical_columns = ["PULocationID", "transaction_date", "transaction_month", "transaction_day", "transaction_hour"]
numerical_columns = ["trip_distance", "total_distance"]
all_needed_columns = categorical_columns + numerical_columns

# %% Cell 21
