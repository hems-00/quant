import pandas as pd

DATA_PATH = "../data/processed/nifty_features.csv"

df = pd.read_csv(
    DATA_PATH,
    index_col=0,
    parse_dates=True
)

q33 = df["rv_21"].quantile(0.33)
q66 = df["rv_21"].quantile(0.66)

def label_risk(x):
    if x < q33:
        return 0   # Low
    elif x < q66:
        return 1   # Medium
    else:
        return 2   # High

df["risk_label"] = df["rv_21"].apply(label_risk)

features = [
    "ema_signal",
    "volume_z",
    "atr_norm"
]

df[features] = df[features].shift(1)
df = df.dropna()

X = df[features]
y = df["risk_label"]

from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

from lightgbm import LGBMClassifier

model = LGBMClassifier(
    objective="multiclass",
    num_class=3,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5
)

from sklearn.metrics import classification_report

for train_idx, test_idx in tscv.split(X):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print(classification_report(y_test, preds))
