import pandas as pd
import glob
import os

# ---------------------------------------
# Find all stock CSV files
# ---------------------------------------

files = glob.glob("data/raw/stocks/*.csv")

print(f"Found {len(files)} stock files.")

all_data = []

# ---------------------------------------
# Read every stock
# ---------------------------------------

for file in files:

    print(f"Reading: {file}")

    ticker = os.path.basename(file).replace(".csv", "")

    df = pd.read_csv(file, skiprows=[1, 2])

    # Add ticker/company identifier
    df["Ticker"] = ticker

    all_data.append(df)

# ---------------------------------------
# Combine everything
# ---------------------------------------

combined = pd.concat(
    all_data,
    ignore_index=True
)

# ---------------------------------------
# Convert date
# ---------------------------------------

combined["Date"] = pd.to_datetime(
    combined["Price"]
    if "Price" in combined.columns
    else combined["Date"]
)

# ---------------------------------------
# Sort
# ---------------------------------------

combined = combined.sort_values(
    ["Date", "Ticker"]
)

# ---------------------------------------
# Create processed folder
# ---------------------------------------

os.makedirs(
    "data/processed",
    exist_ok=True
)

# ---------------------------------------
# Save
# ---------------------------------------

combined.to_csv(
    "data/processed/all_stocks.csv",
    index=False
)

# ---------------------------------------
# Information
# ---------------------------------------

print("\nCombined dataset created!")

print("Rows:", len(combined))
print("Columns:", len(combined.columns))

print("\nStocks:")
print(combined["Ticker"].unique())

print("\nDataset preview:")
print(combined.head())

print("\nSaved to:")
print("data/processed/all_stocks.csv")