import pandas as pd

from xgboost import XGBClassifier


class PredictionAgent:
    """
    Uses XGBoost to predict the probability
    that a stock's 5-day direction will be UP.
    """

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

        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )

    def train(self, data):

        data = data.replace(
            [float("inf"), float("-inf")],
            float("nan")
        )

        data = data.dropna(
            subset=self.features + ["target"]
        )

        data = data.sort_values(
            ["Date", "Ticker"]
        )

        unique_dates = sorted(data["Date"].unique())

        split_index = int(len(unique_dates) * 0.8)

        train_dates = unique_dates[:split_index]

        train = data[
            data["Date"].isin(train_dates)
        ]

        X_train = train[self.features]
        y_train = train["target"]

        self.model.fit(X_train, y_train)

        print("Prediction Agent: model trained.")

    def predict(self, data):

        latest = (
            data.sort_values("Date")
            .groupby("Ticker")
            .tail(1)
            .copy()
        )

        latest = latest.replace(
            [float("inf"), float("-inf")],
            float("nan")
        )

        latest = latest.dropna(
            subset=self.features
        )

        probabilities = self.model.predict_proba(
            latest[self.features]
        )[:, 1]

        latest["UP_Probability"] = probabilities

        latest["DOWN_Probability"] = (
            1 - probabilities
        )

        latest["Prediction"] = latest[
            "UP_Probability"
        ].apply(
            lambda x: "UP" if x >= 0.5 else "DOWN"
        )

        return latest[
            [
                "Ticker",
                "Date",
                "UP_Probability",
                "DOWN_Probability",
                "Prediction"
            ]
        ]


if __name__ == "__main__":

    data = pd.read_parquet(
        "data/processed/features_technical.parquet"
    )

    agent = PredictionAgent()

    agent.train(data)

    predictions = agent.predict(data)

    print("\nPREDICTION AGENT RESULTS")
    print("=" * 70)

    print(predictions.to_string(index=False))

    predictions.to_csv(
        "data/processed/prediction_agent_results.csv",
        index=False
    )

    print("\nSaved:")
    print(
        "data/processed/prediction_agent_results.csv"
    )