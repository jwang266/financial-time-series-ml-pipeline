from data_loader import load_price_data
from features import build_features


def main():
    df = load_price_data("AAPL", "2024-01-01", "2024-12-31")
    print(df.head())
    print(df.shape)

    df_feat = build_features(df)
    print(df_feat.head())
    print(df_feat.shape)

    # small check
    print("Label distribution:")
    print(df_feat["label_up_next_day"].value_counts(normalize=True))
    print(df_feat.columns)

if __name__ == "__main__":
    main()
