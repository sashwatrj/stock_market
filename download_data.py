import yfinance as yf

print("Downloading stock data...")

data = yf.download(
    "RELIANCE.NS",
    start="2020-01-01",
    end="2026-01-01"
)

print("Download complete!")

# Daily return
data["return_1d"] = data["Close"].pct_change()

# Future 5-day return
data["future_return_5d"] = (
    data["Close"].shift(-5) / data["Close"]
) - 1

# Target: 1 = UP, 0 = DOWN
data["target"] = (
    data["future_return_5d"] > 0
).astype(int)

# Show results
print("\nFirst 10 rows:")
print(data.head(10))

print("\nDataset shape:")
print(data.shape)

print("\nTarget distribution:")
print(data["target"].value_counts())

# Save
data.to_csv("data/raw/reliance_with_target.csv")

print("\nDataset saved successfully!")