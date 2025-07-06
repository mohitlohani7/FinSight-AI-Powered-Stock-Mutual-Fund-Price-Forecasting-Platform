import yfinance as yf
import pandas as pd
import os

def fetch_full_stock_data(ticker, period="6mo", interval="1d", save=True):
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False)
        if df.empty:
            return None
        df.reset_index(inplace=True)
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df.dropna(inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        if save:
            os.makedirs("data", exist_ok=True)
            df.to_csv(f"data/{ticker.replace('.', '_')}_stock.csv", index=False)
        return df
    except Exception as e:
        print(f"❌ Error fetching stock data: {e}")
        return None
