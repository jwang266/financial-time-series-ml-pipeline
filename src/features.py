import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["daily_return"] = df["Close"].pct_change()
    df["rolling_mean_return_5d"] = df["daily_return"].rolling(window=5, min_periods=5).mean()
    df["rolling_volatility_5d"] = df["daily_return"].rolling(window=5, min_periods=5).std()
    df["rolling_mean_return_20d"] = df["daily_return"].rolling(window=20, min_periods=20).mean()
    df["rolling_volatility_20d"] = df["daily_return"].rolling(window=20, min_periods=20).std()
    df["avg_volume_5d"] = df["Volume"].rolling(window=5, min_periods=5).mean()

    df["relative_volume_5d"] = df["Volume"] / df["avg_volume_5d"]
    df["intraday_price_range"] = (df["High"] - df["Low"]) / df["Close"]

    df["future_return_1d"] = df["Close"].shift(-1) / df["Close"] - 1
    df["label_up_next_day"] = (df["future_return_1d"] > 0).astype(int)

    # drop rows with missing features or labels
    feature_cols = [
        "daily_return",
        "rolling_mean_return_5d",
        "rolling_volatility_5d",
        "rolling_mean_return_20d",
        "rolling_volatility_20d",
        "avg_volume_5d",
        "relative_volume_5d",
        "intraday_price_range",
    ]
    label_col = "label_up_next_day"

    # future_return_1d is only used to create the label
    required_cols = feature_cols + ["future_return_1d", label_col]
    df = df.dropna(subset=required_cols).reset_index(drop=True)  # last row of future_return_1d is NaN
    df = df.drop(columns=["future_return_1d"])

    return df
