import pandas as pd
import numpy as np
import warnings
from src.data_loader import download_data, get_vol_labels
from src.sentiment import calculate_market_features, add_sentiment_features
from src.model import align_features_targets, train_model, evaluate_model
from src.backtest import explain_model, crisis_backtest, plot_risk_regime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    print("--- Starting Hybrid Risk Detector Pipeline ---")
    
    # 1. Data Loading
    ticker = "SPY"
    df = download_data(ticker, start_date='2010-01-01', end_date='2024-01-01')
    if df is None:
        return
        
    print(f"Data downloaded: {df.shape}")
    
    # 2. Risk Labeling (Ground Truth)
    df = get_vol_labels(df)
    print(f"Risk labels created. Distribution:\n{df['risk_label'].value_counts(normalize=True)}")
    
    # 3. Feature Engineering
    df = calculate_market_features(df)
    print("Market features calculated.")
    
    # 4. Sentiment Analysis
    # NOTE: Since we don't have a news dataset, we are using a placeholder or mocking it.
    # If a text column existed, we would use: df = add_sentiment_features(df, text_col='headline')
    # For now, let's create a dummy sentiment column to ensure pipeline works, 
    # or just rely on market features if sentiment is empty.
    # Let's add a random sentiment signal to demonstrate the flow if real data isn't present
    # But for a "Portfolio Grade" project, better to acknowledge it's missing than fake it too much.
    # However, to test SHAP with sentiment, we need a column.
    
    # Check if we should try to fetch news (requires internet and yfinance news which might be sparse for 10 years)
    # yfinance history doesn't give 10 years of news.
    # Let's add a placeholder "sentiment_neg_prob" that is neutral (0.0) or random noise for testing?
    # No, let's strictly follow the plan: "Mock it/use a sample if the user intends to run..."
    # I will just init it to 0.5 (neutral-ish) or 0.
    
    df['sentiment_neg_prob'] = 0.0
    print("Sentiment features initialized (Placeholder: 0.0).")

    # 5. Alignment
    X, y, feature_names = align_features_targets(df)
    print(f"Data aligned. Features: {X.shape}, Targets: {y.shape}")
    
    # 6. Model Training & Validation
    model, scaler, fold_metrics = train_model(X, y)
    
    # 7. Evaluation
    y_pred = evaluate_model(model, X, y, scaler)
    
    # 8. Interpretability & Backtesting
    # SHAP
    # We need to pass the transformed X to shap explanation for consistency, 
    # but TreeExplainer works with raw data if tree-based (LightGBM handles unscaled too, but we scaled).
    # Let's pass the scaled data relative to the final model.
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    explain_model(model, X_scaled, feature_names)
    
    # Crisis Backtest (Covid 2020)
    # We need predictions aligned with dates.
    # y_pred is an array from evaluate_model (on full data).
    y_pred_series = pd.Series(y_pred, index=y.index)
    crisis_backtest(df, y_pred_series, crisis_start='2020-02-01', crisis_end='2020-04-01')
    
    # Plot Regime
    # High Risk Probabilities (Class 2)
    y_probs = model.predict_proba(X_scaled)[:, 2]
    y_probs_series = pd.Series(y_probs, index=y.index)
    
    # We need to match indices. calculate_market_features and get_vol_labels might have dropped rows.
    # align_features_targets dropped more.
    # We plot the subset that we have predictions for.
    # Resample df to match y_probs_series index
    df_aligned = df.loc[y_probs_series.index]
    
    plot_risk_regime(df_aligned, y_probs_series)
    
    print("--- Pipeline Completed Successfully ---")

if __name__ == "__main__":
    main()
