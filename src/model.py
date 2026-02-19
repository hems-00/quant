import pandas as pd
import numpy as np
import os
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
# Dynamic path resolution to fix "path not correct" error
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Assuming folder structure: src/model.py -> data/processed/nifty_features.csv
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "nifty_features.csv")

MODEL_PATH = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_PATH, exist_ok=True)

def load_and_prep_data(filepath=DATA_PATH):
    """
    Loads features, creates risk labels, and aligns data.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # 1. Create Ground Truth Risk Labels (if not already present)
    # Using realized volatility (rv_21) quartiles as per your original logic
    if "risk_label" not in df.columns:
        # Dynamic thresholds based on data distribution
        q33 = df["rv_21"].quantile(0.33)
        q66 = df["rv_21"].quantile(0.66)
        
        def label_risk(x):
            if x < q33: return 0   # Low Risk
            elif x < q66: return 1 # Medium Risk
            else: return 2         # High Risk
            
        df["risk_label"] = df["rv_21"].apply(label_risk)
        print(f"Risk Labels Created (Thresholds: Low<{q33:.4f}, High>{q66:.4f})")

    # 2. Select Features
    # We use the engineered features. 
    # NOTE: When you add sentiment later, just append 'sentiment_score' to this list.
    feature_cols = [
        "ema_signal", 
        "volume_z", 
        "atr_norm"
    ]
    
    # Check if features exist
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # 3. Shift Features (Avoid Lookahead Bias)
    # We want to predict Today's Risk using Yesterday's market data
    df[feature_cols] = df[feature_cols].shift(1)
    
    # Drop NaNs created by shifting
    df_clean = df.dropna()
    
    X = df_clean[feature_cols]
    y = df_clean["risk_label"]
    
    return X, y, df_clean

def train_model(X, y):
    """
    Trains LightGBM Classifier with TimeSeries Cross-Validation.
    """
    print("\n--- Starting Training (LightGBM) ---")
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    # LightGBM is generally the "Best Model" for tabular time-series 
    # because it handles non-linearities, is fast, and supports interpretation (SHAP).
    model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=500,        # More trees
        learning_rate=0.05,      # Slower learning for better generalization
        max_depth=5,
        random_state=42,
        class_weight="balanced", # Handle potential class imbalance
        verbosity=-1
    )
    
    fold = 1
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=[] # Removed early_stopping_rounds for compatibility with newer versions if needed
        )
        
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        scores.append(acc)
        print(f"Fold {fold} Accuracy: {acc:.4f}")
        fold += 1
        
    print(f"\nAverage CV Accuracy: {np.mean(scores):.4f}")
    
    # Final Fit on all data
    model.fit(X, y)
    
    # Save model
    save_path = os.path.join(MODEL_PATH, "lgbm_risk_model.pkl")
    joblib.dump(model, save_path)
    print(f"Model saved to: {save_path}")
    
    return model

def evaluate_model(model, X, y):
    """
    Evaluates the model.
    """
    preds = model.predict(X)
    print("\n--- Final Evaluation (In-Sample) ---")
    print(classification_report(y, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y, preds))
    return preds

# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    try:
        # 1. Load
        X, y, df = load_and_prep_data()
        
        # 2. Train
        model = train_model(X, y)
        
        # 3. Evaluate
        evaluate_model(model, X, y)
        
        print("\nReady for Sentiment Integration.")
        print("To combine later: Add 'sentiment' column to CSV, add it to 'feature_cols' list in this file.")
        
    except Exception as e:
        print(f"Error: {e}")
