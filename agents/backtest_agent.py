import pandas as pd
import numpy as np

from xgboost import XGBClassifier


class BacktestAgent:

    def __init__(self):

        self.features = [
            "return_1d",
            "return_5d",
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

    def run(self, data):

        data = data.copy()

        data = data.replace(
            [float("inf"), float("-inf")],
            float("nan")
        )

        data = data.dropna(
            subset=self.features + ["target"]
        )

        data = data.sort_values(
            ["Date", "Ticker"]
        ).reset_index(drop=True)

        dates = sorted(data["Date"].unique())

        predictions = []

        # Train periodically using only past data
        for i in range(0, len(dates), 20):

            current_date = dates[i]

            train_dates = dates[:i]

            if len(train_dates) < 250:
                continue

            train = data[
                data["Date"].isin(train_dates)
            ]

            test = data[
                data["Date"] == current_date
            ]

            if len(test) == 0:
                continue

            model = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss"
            )

            model.fit(
                train[self.features],
                train["target"]
            )

            probability = model.predict_proba(
                test[self.features]
            )[:, 1]

            test = test.copy()

            test["UP_Probability"] = probability

            test["Prediction"] = np.where(
                probability >= 0.5,
                "UP",
                "DOWN"
            )

            predictions.append(test)

            print(
                f"Backtested: {current_date}"
            )

        if not predictions:
            raise ValueError(
                "Not enough historical data "
                "for walk-forward backtesting."
            )

        results = pd.concat(
            predictions,
            ignore_index=True
        )

        return results

    def calculate_metrics(self, results):

        results = results.sort_values(
            ["Date", "Ticker"]
        ).copy()

        results["Position"] = (
            results["Prediction"]
            .map({
                "UP": 1,
                "DOWN": -1
            })
        )

        results["Strategy_Return"] = (
            results["Position"]
            * results["future_return_5d"]
        )

        cumulative_return = (
            1 + results["Strategy_Return"]
        ).prod() - 1

        mean_return = (
            results["Strategy_Return"].mean()
        )

        std_return = (
            results["Strategy_Return"].std()
        )

        if std_return > 0:

            sharpe = (
                mean_return /
                std_return
            ) * np.sqrt(252)

        else:

            sharpe = 0

        equity = (
            1 + results["Strategy_Return"]
        ).cumprod()

        running_max = equity.cummax()

        drawdown = (
            equity / running_max
        ) - 1

        max_drawdown = drawdown.min()

        win_rate = (
            results["Strategy_Return"] > 0
        ).mean()

        return {
            "Cumulative Return": cumulative_return,
            "Sharpe Ratio": sharpe,
            "Maximum Drawdown": max_drawdown,
            "Win Rate": win_rate
        }


if __name__ == "__main__":

    data = pd.read_parquet(
        "data/processed/features_technical.parquet"
    )

    agent = BacktestAgent()

    results = agent.run(data)

    metrics = agent.calculate_metrics(
        results
    )

    print("\n")
    print("=" * 70)
    print("WALK-FORWARD BACKTEST RESULTS")
    print("=" * 70)

    for name, value in metrics.items():

        print(
            f"{name}: {value:.4f}"
        )

    results.to_csv(
        "data/processed/walk_forward_backtest.csv",
        index=False
    )

    print("\nSaved:")
    print(
        "data/processed/walk_forward_backtest.csv"
    )