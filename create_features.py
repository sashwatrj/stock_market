import pandas as pd

print("Loading cleaned stock data...")

# Load the cleaned dataset
data = pd.read_parquet(
    "data/processed/clean_stocks.parquet"
)

print("Data loaded!")
print("Rows:", len(data))
print("Number of stocks:", data["Ticker"].nunique())


# ---------------------------------------
# Sort data correctly
# ---------------------------------------

data = data.sort_values(
    ["Ticker", "Date"]
).reset_index(drop=True)


# ---------------------------------------
# 1. Daily return
# ---------------------------------------

data["return_1d"] = (
    data.groupby("Ticker")["Close"]
    .pct_change()
)


# ---------------------------------------
# 2. 5-day historical return
# ---------------------------------------

data["return_5d"] = (
    data.groupby("Ticker")["Close"]
    .pct_change(5)
)


# ---------------------------------------
# 3. Future 5-day return
# ---------------------------------------

data["future_return_5d"] = (
    data.groupby("Ticker")["Close"]
    .shift(-5)
    / data["Close"]
) - 1


# ---------------------------------------
# 4. Prediction target
# ---------------------------------------

# 1 = price goes UP over next 5 days
# 0 = price goes DOWN over next 5 days

data["target"] = (
    data["future_return_5d"] > 0
).astype(int)


# ---------------------------------------
# Save
# ---------------------------------------

data.to_parquet(
    "data/processed/features_basic.parquet",
    index=False
)

data.to_csv(
    "data/processed/features_basic.csv",
    index=False
)


# ---------------------------------------
# Show results
# ---------------------------------------

print("\n==============================")
print("FEATURE ENGINEERING COMPLETE")
print("==============================")

print("\nNumber of stocks:")
print(data["Ticker"].nunique())

print("\nNumber of rows:")
print(len(data))

print("\nColumns:")
print(data.columns.tolist())

print("\nTarget distribution:")
print(data["target"].value_counts())

print("\nExample rows:")

print(
    data[
        [
            "Date",
            "Ticker",
            "Close",
            "return_1d",
            "return_5d",
            "future_return_5d",
            "target"
        ]
    ].head(20)
)

print("\nSaved files:")
print("data/processed/features_basic.parquet")
print("data/processed/features_basic.csv")
