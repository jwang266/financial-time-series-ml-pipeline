from data_loader import load_price_data
from features import build_features


def main():
    df = load_price_data("AAPL", "2024-01-01", "2024-12-31")
    df_feat = build_features(df)

    df = df_feat.copy()
    X = df.drop(columns=["Date", "label_up_next_day"])
    y = df["label_up_next_day"]
    n = len(df)

    # time-based split: first 80% as train, last 20% as test
    split_idx = int(n * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]

    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    print(len(X_train))
    print(len(X_test))
    print(y_test.value_counts(normalize=True))

if __name__ == "__main__":
    main()
