import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

def explain_model(model, X, feature_names):
    """
    Uses SHAP to explain feature importance.
    Plots the top features contributing to 'High Risk' class (Class 2).
    """
    # Create explainer
    # TreeExplainer is optimized for trees (LightGBM)
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X)
    
    # shap_values is a list of arrays for multiclass [class0, class1, class2]
    # We focus on Class 2 (High Risk)
    shap_vals_high_risk = shap_values[2]
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_high_risk, X, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance (High Risk Class)")
    plt.tight_layout()
    plt.savefig("results/shap_summary_high_risk.png")
    print("SHAP plot saved to results/shap_summary_high_risk.png")
    plt.close()

def crisis_backtest(df, y_pred_series, crisis_start='2020-02-01', crisis_end='2020-03-23'):
    """
    Analyzes model performance during a specific crisis window.
    Calculates lead time: days before volatility spike that model predicted High Risk.
    """
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    mask = (df.index >= crisis_start) & (df.index <= crisis_end)
    crisis_df = df.loc[mask]
    
    if crisis_df.empty:
        print(f"No data found for crisis period {crisis_start} to {crisis_end}")
        return

    # Assuming y_pred_series is aligned with df
    # Let's filter predictions too
    preds = y_pred_series.loc[mask]
    
    # Identify the "Peak Volatility" day in this window? Or the first day of "High Risk" label?
    # The user asks: "Lead Time — how many days before the volatility spike did the model transition to 'High Risk'?"
    
    # Let's define "Volatility Spike" as the day with MAX realized volatility in the window.
    # Or simply when the ground truth becomes 'High Risk'.
    
    # Find first day model predicted High Risk (2)
    high_risk_days = preds[preds == 2].index
    first_signal_date = high_risk_days.min() if not high_risk_days.empty else None
    
    # Find actual High Risk regime start in this window (from labels)
    actual_high_risk = crisis_df[crisis_df['risk_label'] == 2].index
    actual_start_date = actual_high_risk.min() if not actual_high_risk.empty else None
    
    print(f"\n--- Crisis Backtest ({crisis_start} to {crisis_end}) ---")
    
    if first_signal_date:
        print(f"First High Risk Signal: {first_signal_date.date()}")
        if actual_start_date:
            lead_time = (actual_start_date - first_signal_date).days
            print(f"Actual High Risk Start: {actual_start_date.date()}")
            print(f"Lead Time: {lead_time} days")
        else:
            print("Market did not reach High Risk regime in this window according to labels.")
    else:
        print("Model did not predict High Risk in this window.")

def plot_risk_regime(df, y_probs_series):
    """
    Dual-axis plot: Price on left, High Risk Probability on right (shaded).
    y_probs_series: Probability of Class 2 (High Risk).
    """
    plt.figure(figsize=(14, 7))
    
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Plot Price
    ax1.plot(df.index, df['Close'], color='black', alpha=0.7, label='Price')
    ax1.set_ylabel('Price', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Plot Probability as shaded area
    # y_probs_series should be aligned with df dates
    ax2.fill_between(df.index, y_probs_series, color='red', alpha=0.3, label='High Risk Prob')
    ax2.set_ylabel('High Risk Probability', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 1)
    
    plt.title("Price vs Constructed High Risk Probability")
    plt.savefig("results/risk_regime_plot.png")
    print("Risk regime plot saved to results/risk_regime_plot.png")
    plt.close()
