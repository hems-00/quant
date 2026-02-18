import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = "../data/processed/nifty_features.csv"


def load_data():
    df = pd.read_csv(
        DATA_PATH,
        index_col=0,
        parse_dates=True
    )
    return df


def plot_price_and_volatility(df):
    """
    Price vs Realized Volatility
    """

    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.plot(df.index, df["Close"], label="NIFTY50 Close")
    ax1.set_ylabel("Price")

    ax2 = ax1.twinx()
    ax2.plot(df.index, df["rv_21"], linestyle="--", label="Realized Volatility")
    ax2.set_ylabel("Volatility")

    plt.title("Price vs Realized Volatility")
    fig.tight_layout()
    plt.show()


def plot_ema_signal(df):
    plt.figure(figsize=(14, 5))

    plt.plot(df.index, df["ema_signal"])
    plt.axhline(0, linestyle="--")

    plt.title("EMA 12/26 Crossover Signal")
    plt.ylabel("EMA Difference")
    plt.show()


def plot_volume_zscore(df):
    plt.figure(figsize=(14, 5))

    plt.plot(df.index, df["volume_z"])
    plt.axhline(2, linestyle="--")
    plt.axhline(-2, linestyle="--")

    plt.title("Volume Z-Score (Participation Shock)")
    plt.ylabel("Z-score")
    plt.show()


def plot_atr(df):
    plt.figure(figsize=(14, 5))

    plt.plot(df.index, df["atr_norm"])

    plt.title("ATR Normalized by Price")
    plt.ylabel("ATR / Price")
    plt.show()


def main():

    df = load_data()

    print("Dataset Loaded:")
    print(df.head())

    plot_price_and_volatility(df)
    plot_ema_signal(df)
    plot_volume_zscore(df)
    plot_atr(df)


if __name__ == "__main__":
    main()
