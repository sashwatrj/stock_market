import yfinance as yf
import pandas as pd
import os
import time

# -----------------------------------
# Stocks we want
# -----------------------------------

stocks = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "ITC.NS",
    "LT.NS",
    "BHARTIARTL.NS",
    "AXISBANK.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
    "TATAMOTORS.NS",
    "HINDUNILVR.NS",
    "KOTAKBANK.NS"
]

# -----------------------------------
# Create folder
# -----------------------------------

os.makedirs("data/raw/stocks", exist_ok=True)

# -----------------------------------
# Download each stock
# -----------------------------------

for ticker in stocks:

    print(f"Downloading {ticker}...")

    try:

        data = yf.download(
            ticker,
            start="2020-01-01",
            end="2026-01-01",
            auto_adjust=False,
            progress=False
        )

        if data.empty:
            print(f"No data found for {ticker}")
            continue

        # Save individual stock
        filename = f"data/raw/stocks/{ticker.replace('.NS', '')}.csv"

        data.to_csv(filename)

        print(f"Saved: {filename}")

    except Exception as e:

        print(f"Error downloading {ticker}: {e}")

    time.sleep(1)

print("\nAll downloads completed!")