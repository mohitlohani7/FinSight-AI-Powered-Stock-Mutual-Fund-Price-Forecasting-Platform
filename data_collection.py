import yfinance as yf
import pandas as pd
import os

def fetch_full_stock_data(ticker, period="6mo", interval="1d"):
    print(f"📥 Downloading stock data for {ticker}...")

    data = yf.download(ticker, period=period, interval=interval, auto_adjust=False)

    # Reset index to get Date column
    data.reset_index(inplace=True)

    # Flatten multi-index columns if needed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if col[0] != 'Date' else 'Date' for col in data.columns]

    # ✅ Debug print
    print("📋 Final Columns:", data.columns.tolist())

    # Drop if missing
    if "Date" in data.columns and "Close" in data.columns:
        data.dropna(subset=["Date", "Close"], inplace=True)
    else:
        raise KeyError(f"❌ Columns missing: {set(['Date', 'Close']) - set(data.columns)}")

    # Select required columns
    data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]

    # Format date
    data["Date"] = pd.to_datetime(data["Date"]).dt.date

    # Save to CSV
    os.makedirs("data", exist_ok=True)
    filename = f"data/{ticker}_stock.csv"
    data.to_csv(filename, index=False)
    print(f"✅ Cleaned data saved to: {filename}")

# 🔄 Try with TCS
fetch_full_stock_data("TCS.NS", period="6mo")
