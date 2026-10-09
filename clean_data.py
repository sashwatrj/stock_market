import pandas as pd
import glob
import os

# Find all downloaded stock files
files = glob.glob("data/raw/stocks/*.csv")

cleaned_stocks = []

print(f"Found {len(files)} stock files.\n")

for file in files:

    ticker = os.path.basename(file).replace(".csv", "")

    print(f"Cleaning {ticker}...")

    # Read yfinance CSV
    df = pd.read_csv(file, skiprows=[1, 2])

    # Rename first column to Date
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    # Add ticker
    df["Ticker"] = ticker

    # Convert Date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Keep only useful columns
    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Ticker"
    ]

    # Only keep columns that actually exist
    available_columns = [
        col for col in required_columns
        if col in df.columns
    ]

    df = df[available_columns]

    # Convert numerical columns
    numerical_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for column in numerical_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Remove invalid rows
    df = df.dropna(
        subset=["Date", "Close"]
    )

    # Remove duplicate dates
    df = df.drop_duplicates(
        subset=["Date", "Ticker"]
    )

    cleaned_stocks.append(df)


# Combine all stocks
data = pd.concat(
    cleaned_stocks,
    ignore_index=True
)

# Sort by ticker and date
data = data.sort_values(
    ["Ticker", "Date"]
)

# Reset index
data = data.reset_index(drop=True)

# Create processed directory
os.makedirs(
    "data/processed",
    exist_ok=True
)

# Save as CSV
data.to_csv(
    "data/processed/clean_stocks.csv",
    index=False
)

# Also save as Parquet
data.to_parquet(
    "data/processed/clean_stocks.parquet",
    index=False
)

print("\n-----------------------------")
print("CLEANING COMPLETE")
print("-----------------------------")

print("\nNumber of stocks:")
print(data["Ticker"].nunique())

print("\nNumber of rows:")
print(len(data))

print("\nColumns:")
print(data.columns.tolist())

print("\nMissing values:")
print(data.isna().sum())

print("\nStocks:")
print(data["Ticker"].unique())

print("\nFirst 10 rows:")
print(data.head(10))

print("\nSaved:")
print("data/processed/clean_stocks.csv")
print("data/processed/clean_stocks.parquet")