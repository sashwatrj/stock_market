import pandas as pd
import numpy as np


class PortfolioBacktest:

    def __init__(self, transaction_cost=0.001):
        self.transaction_cost = transaction_cost

    def run(self, data):

        data = data.copy()

        data = data.replace(
            [float("inf"), float("-inf")],
            float("nan")
        )

        data = data.dropna(
            subset=["future_return_5d"]
        )

        data["Position"] = data["Prediction"].map({
            "UP": 1,
            "DOWN": -1
        }).fillna(0)

        # Equal-weight position across stocks
        daily = (
            data.groupby("Date")
            .agg(
                Strategy_Return=(
                    "future_return_5d",
                    lambda x: x.mean()
                ),
                Average_Position=(
                    "Position",
                    "mean"
                ),
                Stocks=("Ticker", "count")
            )
            .reset_index()
        )

        # Apply direction to return
        signed_returns = (
            data["Position"] *
            data["future_return_5d"]
        )

        data["Signed_Return"] = signed_returns

        daily = (
            data.groupby("Date")
            .agg(
                Strategy_Return=(
                    "Signed_Return",
                    "mean"
                ),
                Stocks=("Ticker", "count")
            )
            .reset_index()
        )

        # Transaction cost approximation
        daily["Transaction_Cost"] = (
            self.transaction_cost / 5
        )

        daily["Net_Return"] = (
            daily["Strategy_Return"]
            - daily["Transaction_Cost"]
        )

        # Equity curve
        daily["Equity"] = (
            1 + daily["Net_Return"]
        ).cumprod()

        # Metrics
        cumulative_return = (
            daily["Equity"].iloc[-1] - 1
        )

        mean_return = daily["Net_Return"].mean()
        std_return = daily["Net_Return"].std()

        sharpe = 0

        if std_return > 0:
            sharpe = (
                mean_return /
                std_return
            ) * np.sqrt(252)

        downside = daily.loc[
            daily["Net_Return"] < 0,
            "Net_Return"
        ]

        downside_std = downside.std()

        sortino = 0

        if downside_std > 0:
            sortino = (
                mean_return /
                downside_std
            ) * np.sqrt(252)

        running_max = daily["Equity"].cummax()

        drawdown = (
            daily["Equity"] /
            running_max
        ) - 1

        max_drawdown = drawdown.min()

        win_rate = (
            daily["Net_Return"] > 0
        ).mean()

        metrics = {
            "Cumulative Return": cumulative_return,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Maximum Drawdown": max_drawdown,
            "Win Rate": win_rate
        }

        return daily, metrics


if __name__ == "__main__":

    data = pd.read_csv(
        "data/processed/walk_forward_backtest.csv"
    )

    agent = PortfolioBacktest()

    portfolio, metrics = agent.run(data)

    print("\n")
    print("=" * 70)
    print("PORTFOLIO BACKTEST")
    print("=" * 70)

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    portfolio.to_csv(
        "data/processed/portfolio_backtest.csv",
        index=False
    )

    print("\nSaved:")
    print(
        "data/processed/portfolio_backtest.csv"
    )