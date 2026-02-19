from src.data.loader import download_data
from src.features.engineering import engineer_features
import pandas as pd

# verify pandas version just in case
print(f"Pandas version: {pd.__version__}")

def main():
    ticker = "ORCL"
    start_date = "2023-01-01"
    end_date = "2024-01-01"

    print(f"--- Phase 1: Market Signal Engineering for {ticker} ---")

    # 1. Download Data
    df = download_data(ticker, start_date, end_date)
    if df is None:
        return

    print(f"Raw data shape: {df.shape}")
    print(df.head())

    # 2. Engineer Features
    df_features = engineer_features(df)
    
    if df_features is list: # Should not be list, but safety
         print("Error: Features function returned a list")
         return

    print("\n--- Feature Engineering Complete ---")
    print(f"Processed data shape: {df_features.shape}")
    
    # Save to CSV
    output_path = "data/processed/market_signals.csv"
    df_features.to_csv(output_path)
    print(f"Data saved to: {output_path}")

    print(df_features[['Close', 'returns', 'volatility_20d', 'momentum_10d', 'volume_anomaly', 'bb_width', 'trend_ema_50']].head())
    print("\nData Info:")
    print(df_features.info())

if __name__ == "__main__":
    main()
