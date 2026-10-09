import pandas as pd
from transformers import pipeline


class NewsSentimentAgent:

    def __init__(self):

        print("Loading FinBERT...")

        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert"
        )

        print("FinBERT loaded.")

    def analyze(self, news_data):

        results = []

        for _, row in news_data.iterrows():

            result = self.sentiment_pipeline(
                row["Headline"]
            )[0]

            results.append({
                "Date": row["Date"],
                "Ticker": row["Ticker"],
                "Headline": row["Headline"],
                "Sentiment": result["label"],
                "Sentiment_Score": result["score"]
            })

        return pd.DataFrame(results)


if __name__ == "__main__":

    news = pd.read_csv(
        "data/raw/news/news_sample.csv"
    )

    agent = NewsSentimentAgent()

    results = agent.analyze(news)

    print("\nNEWS SENTIMENT RESULTS")
    print("=" * 80)

    print(results.to_string(index=False))

    results.to_csv(
        "data/processed/news_sentiment.csv",
        index=False
    )

    print("\nSaved:")
    print("data/processed/news_sentiment.csv")