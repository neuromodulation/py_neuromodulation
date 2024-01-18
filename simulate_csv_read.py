import pandas as pd
import numpy as np
import time
import pyarrow
from pyarrow import csv

# Time Write pandas: 30.8 seconds, Time read: 6.6 seconds
# Time Write pyarrow: 7.9 seconds, Time read: 0.65 seconds

# Time Write pandas with pyarrow engine currently not possible, Time read: 0.7 seconds


# Define the size of the DataFrame
n_rows = 10**6
n_cols = 100

# Create a DataFrame with random values
df = pd.DataFrame(
    np.random.randint(0, 100, size=(n_rows, n_cols)),
    columns=["col" + str(i) for i in range(n_cols)],
)

for SAVE_PYARROW in [True, False]:
    print(f"SAVE_PYARROW: {SAVE_PYARROW}")
    start_time = time.time()

    if SAVE_PYARROW:
        csv.write_csv(pyarrow.Table.from_pandas(df), "data.csv")
    else:
        # Save the DataFrame to a CSV file
        df.to_csv("data.csv", index=False)

    # Calculate the elapsed time
    elapsed_time = time.time() - start_time

    print(f"Time elapsed: {elapsed_time} seconds")

for READ_PYARROW in [True, False]:
    print(f"READ_PYARROW: {READ_PYARROW}")
    start_time = time.time()

    if READ_PYARROW:
        table = csv.read_csv("data.csv")
        df_pyarrow = table.to_pandas()
    else:
        # Read the CSV file into a DataFrame
        df_pd = pd.read_csv("data.csv")

    # Calculate the elapsed time
    elapsed_time = time.time() - start_time

    print(f"Time elapsed: {elapsed_time} seconds")

print("Use pandas pyarrow engine to read the csv file time")
start_time = time.time()
df = pd.read_csv("data.csv", engine="pyarrow")
elapsed_time = time.time() - start_time
print(f"Time elapsed: {elapsed_time} seconds")
