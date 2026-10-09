import pandas as pd


class RiskAgent:
    """
    Estimates recent volatility and drawdown risk
    for each stock.
    """

    def analyze(self, data):

        results = []

        for ticker, stock in data.groupby("Ticker"):

            stock = stock.sort_values("Date").copy()

            latest = stock.iloc[-1]

            # Recent volatility
            volatility = latest["volatility_20d"]

            # Maximum drawdown over available history
            running_max = stock["Close"].cummax()

            drawdown = (
                stock["Close"] / running_max
            ) - 1

            max_drawdown = drawdown.min()

            # Simple risk classification
            if volatility >= 0.03:
                risk_level = "HIGH"
            elif volatility >= 0.015:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            results.append({
                "Ticker": ticker,
                "Volatility_20d": volatility,
                "Max_Drawdown": max_drawdown,
                "Risk_Level": risk_level
            })

        return pd.DataFrame(results)


if __name__ == "__main__":

    data = pd.read_parquet(
        "data/processed/features_technical.parquet"
    )

    agent = RiskAgent()

    results = agent.analyze(data)

    print("\nRISK AGENT RESULTS")
    print("=" * 70)

    print(results.to_string(index=False))

    results.to_csv(
        "data/processed/risk_agent_results.csv",
        index=False
    )

    print("\nSaved:")
    print("data/processed/risk_agent_results.csv")