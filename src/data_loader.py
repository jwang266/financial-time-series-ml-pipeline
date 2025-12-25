import yfinance as yf
import pandas as pd


def load_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)

    # flatten columns if they are MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    # ensure date ascending order
    df = df.sort_values("Date").reset_index(drop=True)
    # clear columns index name
    df.columns.name = None

    return df
