import os
import pandas as pd
import numpy as np
import yfinance as yf

RAW_PATH = "../data/raw/nifty_raw.csv"
PROC_PATH = "../data/processed/nifty_features.csv"


def load_nifty_data():

    if os.path.exists(RAW_PATH):
        df = pd.read_csv(
            RAW_PATH,
            index_col=0,
            parse_dates=True
        )
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]

        df[numeric_cols] = df[numeric_cols].apply(
            pd.to_numeric,
            errors="coerce"
        )

        df = df.ffill().dropna()

    else:
        print("Downloading data from yfinance (one-time)...")

        df = yf.download(
            "^NSEI",
            period="10y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True   # faster API fetch
        )
        print(df.dtypes)
        df.to_csv(RAW_PATH)

    return df

def calculate_features(df):
    df["returns"] = df["Close"].pct_change()

    df["rv_21"] = (    #volatility 
        df["returns"]**2
    ).rolling(21).sum().pow(0.5)

    df["ema12"] = df["Close"].ewm(span=12).mean() # movinf average for 12 and 26 day
    df["ema26"] = df["Close"].ewm(span=26).mean()
    df["ema_signal"] = df["ema12"] - df["ema26"]

    vol_mean = df["Volume"].rolling(20).mean()  #volume zscore formula
    vol_std = df["Volume"].rolling(20).std()
    df["volume_z"] = (df["Volume"] - vol_mean) / vol_std

    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    close = df["Close"].squeeze()

    # True Range components
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df["atr"] = tr.rolling(14).mean()
    df["atr_norm"] = df["atr"] / df["Close"]

    df = df.ffill().dropna()

    return df

def save_processed(df):
    df.to_csv(PROC_PATH)
    print("Processed dataset saved.")


df = load_nifty_data()
df_features = calculate_features(df)
save_processed(df_features)