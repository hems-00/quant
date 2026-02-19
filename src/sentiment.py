from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import requests
import os

class FinBERTSentiment:
    def __init__(self, model_path="C:/hf_models/finbert", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Robust loading: Try local path first, fallback to HuggingFace Hub
        try:
            if os.path.exists(model_path):
                print(f"Loading local FinBERT from {model_path}...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            else:
                raise FileNotFoundError("Local model not found")
        except Exception as e:
            print(f"Local fetch failed ({e}). Downloading 'yiyanghkust/finbert-tone' from Hub...")
            model_id = "yiyanghkust/finbert-tone"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_id)

        self.model.to(self.device)
        self.model.eval()
        self.labels = self.model.config.id2label
        print(f"FinBERT loaded on {self.device}")

    def predict_batch(self, texts, batch_size=32):
        scores = []
        # Filter non-strings
        valid_texts = [str(t) for t in texts if str(t).strip() != ""]
        
        if not valid_texts:
            return []

        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            probs = torch.softmax(outputs.logits, dim=1)
            
            # Dynamic label finding (handles different model versions)
            pos_id = next((k for k, v in self.labels.items() if "positive" in v.lower()), 2)
            neg_id = next((k for k, v in self.labels.items() if "negative" in v.lower()), 0)
            
            # Sentiment Score: Positive Probability - Negative Probability
            # Range: -1 (Very Negative) to +1 (Very Positive)
            sentiment = probs[:, pos_id] - probs[:, neg_id]
            scores.extend(sentiment.cpu().numpy())

        return scores

    def process_headlines(self, df):
        """
        Takes a dataframe with 'headline' and 'date' columns.
        Returns daily aggregated sentiment features.
        """
        if df.empty or "headline" not in df.columns:
            print("No headlines to process.")
            return pd.DataFrame()

        print(f"Scoring {len(df)} headlines...")
        df = df.copy()
        df["sentiment_score"] = self.predict_batch(df["headline"].tolist())

        # Aggregate by Date
        daily = df.groupby("date").agg(
            sent_mean=("sentiment_score", "mean"),
            sent_volatility=("sentiment_score", "std"), # Divergence in news
            headline_count=("sentiment_score", "count")
        )
        
        # Fill NaN std (single article days) with 0
        daily = daily.fillna(0)
        return daily

def fetch_market_news(api_key, queries=None, days_back=5):
    """
    Fetches news for specific sectors using NewsAPI.
    """
    if not api_key:
        print("Error: No API Key provided.")
        return pd.DataFrame()

    if queries is None:
        # Optimized presets for Indian Market Risk
        queries = [
            "Nifty 50", 
            "Sensex", 
            "Bank Nifty", 
            "RBI Repo Rate", 
            "Indian Economy", 
            "Oil Prices India",
            "FII DII activity"
        ]

    all_articles = []
    base_url = "https://newsapi.org/v2/everything"
    
    # Calculate date range
    from datetime import datetime, timedelta
    from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

    print(f"Fetching news from {from_date}...")

    for q in queries:
        params = {
            "q": q,
            "language": "en",
            "from": from_date,
            "sortBy": "publishedAt", # Latest news first
            "pageSize": 50,          # Get enough depth
            "apiKey": api_key
        }
        
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if data.get("status") == "ok":
                articles = data.get("articles", [])
                print(f"Found {len(articles)} articles for '{q}'")
                
                for a in articles:
                    all_articles.append({
                        "date": a["publishedAt"][:10], # ISO format YYYY-MM-DD
                        "headline": a["title"],
                        "source": a["source"]["name"],
                        "sector_tag": q
                    })
            else:
                print(f"API Error for '{q}': {data.get('message')}")
                
        except Exception as e:
            print(f"Request failed for '{q}': {e}")

    df = pd.DataFrame(all_articles)
    
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        # Dedup: Same headline might appear in 'Nifty' and 'Sensex' searches
        initial_len = len(df)
        df = df.drop_duplicates(subset=["headline"])
        print(f"Deduplicated {initial_len} -> {len(df)} unique articles.")
        
        # Sort
        df = df.sort_values(by="date", ascending=False)
        
    return df



# ---------------------------------------------------------
# Test execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Configuration
    # You can set via env var OR hardcode it here
    ENV_KEY = os.getenv("NEWS_API_KEY")
    HARDCODED_KEY = "986575cc59f64f37837b65113fa457a7"
    
    API_KEY = ENV_KEY if ENV_KEY else HARDCODED_KEY
    
    if API_KEY and API_KEY != "your_key_here":
        print(f"--- 1. Fetching News (Key ends in ...{API_KEY[-4:]}) ---")
        
        # Increased fetch limit to ensure good coverage
        news_df = fetch_market_news(API_KEY, days_back=7)
        
        if not news_df.empty:
            print(f"\n--- 2. Scoring Sentiment (FinBERT) on {len(news_df)} articles ---")
            analyzer = FinBERTSentiment() # Auto-downloads if local missing
            daily_sentiment = analyzer.process_headlines(news_df)
            
            print("\n--- 3. Daily Sentiment Signal (Latest) ---")
            print(daily_sentiment)
            
            # Save for the Model
            output_path = "../data/processed/latest_news_sentiment.csv"
            daily_sentiment.to_csv(output_path)
            print(f"Saved sentiment signal to {output_path}")
        else:
            print("News fetch returned empty. Check API limits or query parameters.")
            
    else:
        print("No valid API Key detected.")
        print("Set 'NEWS_API_KEY' env var or update HARDCODED_KEY in script.")
        
        # Fallback test with dummy data
        print("\nRunning Mock Test (for debugging logic only):")
        mock_df = pd.DataFrame({
            "date": ["2023-01-01", "2023-01-01"], 
            "headline": ["Inflation hits new high", "Nifty rallies on strong earnings"]
        })
        analyzer = FinBERTSentiment()
        print(analyzer.process_headlines(mock_df))
