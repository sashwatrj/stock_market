import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands


print("Loading feature dataset...")

data = pd.read_parquet(
    "data/processed/features_basic.parquet"
)

print("Data loaded!")
print("Stocks:", data["Ticker"].nunique())


# ---------------------------------------
# Sort data
# ---------------------------------------

data = data.sort_values(
    ["Ticker", "Date"]
).reset_index(drop=True)


# ---------------------------------------
# Store processed stocks
# ---------------------------------------

processed_stocks = []


# ---------------------------------------
# Process each stock separately
# ---------------------------------------

for ticker, group in data.groupby("Ticker"):

    print(f"Processing {ticker}...")

    group = group.copy()

    # -----------------------------
    # RSI
    # -----------------------------

    group["RSI"] = RSIIndicator(
        close=group["Close"],
        window=14
    ).rsi()


    # -----------------------------
    # MACD
    # -----------------------------

    macd = MACD(
        close=group["Close"],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )

    group["MACD"] = macd.macd()

    group["MACD_signal"] = macd.macd_signal()

    group["MACD_histogram"] = macd.macd_diff()


    # -----------------------------
    # Moving averages
    # -----------------------------

    group["SMA_20"] = (
        group["Close"]
        .rolling(20)
        .mean()
    )

    group["SMA_50"] = (
        group["Close"]
        .rolling(50)
        .mean()
    )

    group["EMA_20"] = (
        group["Close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )


    # -----------------------------
    # Bollinger Bands
    # -----------------------------

    bollinger = BollingerBands(
        close=group["Close"],
        window=20,
        window_dev=2
    )

    group["BB_high"] = (
        bollinger.bollinger_hband()
    )

    group["BB_low"] = (
        bollinger.bollinger_lband()
    )

    group["BB_width"] = (
        group["BB_high"] -
        group["BB_low"]
    )


    # -----------------------------
    # Volatility
    # -----------------------------

    group["volatility_20d"] = (
        group["return_1d"]
        .rolling(20)
        .std()
    )


    # -----------------------------
    # Volume change
    # -----------------------------

    group["volume_change"] = (
        group["Volume"]
        .pct_change()
    )


    # -----------------------------
    # Momentum
    # -----------------------------

    group["momentum_10d"] = (
        group["Close"]
        .pct_change(10)
    )


    # Add processed stock
    processed_stocks.append(group)


# ---------------------------------------
# Combine all stocks
# ---------------------------------------

data = pd.concat(
    processed_stocks,
    ignore_index=True
)


# ---------------------------------------
# Sort again
# ---------------------------------------

data = data.sort_values(
    ["Ticker", "Date"]
).reset_index(drop=True)


# ---------------------------------------
# Save
# ---------------------------------------

data.to_parquet(
    "data/processed/features_technical.parquet",
    index=False
)

data.to_csv(
    "data/processed/features_technical.csv",
    index=False
)


# ---------------------------------------
# Results
# ---------------------------------------

print("\n==============================")
print("TECHNICAL FEATURES COMPLETE")
print("==============================")


print("\nNumber of stocks:")
print(data["Ticker"].nunique())


print("\nNumber of rows:")
print(len(data))


print("\nColumns:")
print(data.columns.tolist())


print("\nTechnical feature preview:")

print(
    data[
        [
            "Date",
            "Ticker",
            "Close",
            "RSI",
            "MACD",
            "MACD_signal",
            "MACD_histogram",
            "SMA_20",
            "SMA_50",
            "EMA_20",
            "BB_width",
            "volatility_20d",
            "volume_change",
            "momentum_10d"
        ]
    ].head(20)
)


print("\nSaved:")
print("data/processed/features_technical.parquet")
print("data/processed/features_technical.csv")