import pandas as pd
from sentiment import FinBERTSentiment


def main():
    """
    Test FinBERT sentiment pipeline independently.
    """

    # -----------------------------
    # 1. Create sample news dataset
    # -----------------------------
    data = {
        "date": [
            "2020-03-10",
            "2020-03-10",
            "2020-03-11",
            "2020-03-11",
            "2020-03-12"
        ],
        "headline": [
            "Markets plunge amid pandemic fears",
            "Federal Reserve announces emergency stimulus",
            "Investors optimistic about recovery",
            "Oil prices collapse due to demand shock",
            "Global markets stabilize after volatility spike"
        ]
    }

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])

    # -----------------------------
    # 2. Load FinBERT Engine
    # -----------------------------
    analyzer = FinBERTSentiment(
        model_path="C:/hf_models/finbert"
    )

    # -----------------------------
    # 3. Process Headlines
    # -----------------------------
    sentiment_daily = analyzer.process_headlines(df)

    # -----------------------------
    # 4. Print Results (testing only)
    # -----------------------------
    print("\nDaily Sentiment Output:")
    print(sentiment_daily)


# -----------------------------
# Python entry point
# -----------------------------
if __name__ == "__main__":
    main()
