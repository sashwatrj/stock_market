import pandas as pd


class NewsSummaryAgent:
    """
    Converts individual news sentiment scores into
    one daily sentiment score for each stock.
    """

    def summarize(self, news):

        # Convert sentiment to numerical values
        sentiment_map = {
            "positive": 1,
            "negative": -1,
            "neutral": 0
        }

        news["Sentiment_Value"] = (
            news["Sentiment"]
            .str.lower()
            .map(sentiment_map)
        )

        # Average sentiment for each stock
        summary = (
            news
            .groupby("Ticker")
            .agg(
                News_Sentiment=(
                    "Sentiment_Value",
                    "mean"
                ),
                News_Count=(
                    "Headline",
                    "count"
                )
            )
            .reset_index()
        )

        # Convert numerical score to direction
        summary["News_Direction"] = (
            summary["News_Sentiment"]
            .apply(
                lambda x:
                "POSITIVE" if x > 0.1
                else "NEGATIVE" if x < -0.1
                else "NEUTRAL"
            )
        )

        return summary


if __name__ == "__main__":

    news = pd.read_csv(
        "data/processed/news_sentiment.csv"
    )

    agent = NewsSummaryAgent()

    summary = agent.summarize(news)

    print("\nNEWS SUMMARY")
    print("=" * 70)

    print(summary.to_string(index=False))

    summary.to_csv(
        "data/processed/news_summary.csv",
        index=False
    )

    print("\nSaved:")
    print("data/processed/news_summary.csv")
    