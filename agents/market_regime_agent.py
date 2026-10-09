import pandas as pd
import yfinance as yf


class MarketRegimeAgent:
    """
    Identifies the current broad market regime
    using the NIFTY 50 index.
    """

    def get_market_data(self):

        print("Downloading NIFTY 50 data...")

        data = yf.download(
            "^NSEI",
            period="5y",
            auto_adjust=True,
            progress=False
        )

        # Handle yfinance multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()

        return data

    def analyze(self, data):

        data["SMA_50"] = (
            data["Close"]
            .rolling(50)
            .mean()
        )

        data["SMA_200"] = (
            data["Close"]
            .rolling(200)
            .mean()
        )

        latest = data.dropna().iloc[-1]

        close = float(latest["Close"])
        sma50 = float(latest["SMA_50"])
        sma200 = float(latest["SMA_200"])

        if close > sma50 and sma50 > sma200:

            regime = "BULL"

        elif close < sma50 and sma50 < sma200:

            regime = "BEAR"

        else:

            regime = "SIDEWAYS"

        return {
            "Market": "NIFTY 50",
            "Close": close,
            "SMA_50": sma50,
            "SMA_200": sma200,
            "Regime": regime
        }


if __name__ == "__main__":

    agent = MarketRegimeAgent()

    data = agent.get_market_data()

    result = agent.analyze(data)

    print("\nMARKET REGIME")
    print("=" * 60)

    for key, value in result.items():

        print(f"{key}: {value}")