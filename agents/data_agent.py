import pandas as pd


class DataAgent:
    """
    Responsible for loading and validating market data.
    """

    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        data = pd.read_parquet(self.data_path)

        print(f"Data loaded: {len(data)} rows")
        print(f"Stocks: {data['Ticker'].nunique()}")

        return data

    def validate_data(self, data):
        required_columns = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Ticker"
        ]

        missing = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

        data = data.sort_values(
            ["Ticker", "Date"]
        ).reset_index(drop=True)

        print("Data validation successful.")

        return data


if __name__ == "__main__":

    agent = DataAgent(
        "data/processed/features_technical.parquet"
    )

    data = agent.load_data()
    data = agent.validate_data(data)

    print("\nData Agent completed successfully.")