# %% Cell 1
import pandas

# %% Cell 2
taxi_df = pandas.read_parquet("./data/yellow_tripdata_2024-10.parquet")
taxi_df.head()
taxi_df.shape
