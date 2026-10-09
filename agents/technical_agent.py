import pandas as pd


class TechnicalAgent:
    """
    Analyzes technical indicators and produces
    a simple directional assessment.
    """

    def analyze(self, data):

        results = []

        for ticker, stock in data.groupby("Ticker"):

            stock = stock.sort_values("Date").copy()

            latest = stock.iloc[-1]

            score = 0
            reasons = []

            # RSI
            if latest["RSI"] < 30:
                score += 1
                reasons.append("RSI indicates oversold conditions")
            elif latest["RSI"] > 70:
                score -= 1
                reasons.append("RSI indicates overbought conditions")

            # MACD
            if latest["MACD"] > latest["MACD_signal"]:
                score += 1
                reasons.append("MACD is above signal line")
            else:
                score -= 1
                reasons.append("MACD is below signal line")

            # Moving average
            if latest["Close"] > latest["SMA_50"]:
                score += 1
                reasons.append("Price is above 50-day SMA")
            else:
                score -= 1
                reasons.append("Price is below 50-day SMA")

            # Momentum
            if latest["momentum_10d"] > 0:
                score += 1
                reasons.append("10-day momentum is positive")
            else:
                score -= 1
                reasons.append("10-day momentum is negative")

            if score > 0:
                direction = "UP"
            elif score < 0:
                direction = "DOWN"
            else:
                direction = "NEUTRAL"

            results.append({
                "Ticker": ticker,
                "Technical_Score": score,
                "Technical_Direction": direction,
                "Technical_Reasons": "; ".join(reasons)
            })

        return pd.DataFrame(results)


if __name__ == "__main__":

    data = pd.read_parquet(
        "data/processed/features_technical.parquet"
    )

    agent = TechnicalAgent()

    results = agent.analyze(data)

    print("\nTECHNICAL AGENT RESULTS")
    print("=" * 60)
    print(results.to_string(index=False))

    results.to_csv(
        "data/processed/technical_agent_results.csv",
        index=False
    )

    print("\nSaved:")
    print("data/processed/technical_agent_results.csv")