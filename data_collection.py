import yfinance as yf
import pandas as pd
import os
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_ticker(ticker):
    """Validate if ticker exists and has data"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get('symbol') is not None
    except Exception as e:
        logger.error(f"Ticker validation failed for {ticker}: {e}")
        return False

def fetch_full_stock_data(ticker, period="6mo", interval="1d", save=True):
    """
    Enhanced stock data fetching with comprehensive error handling
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        save (bool): Whether to save data to CSV
    
    Returns:
        pandas.DataFrame or None: Stock data or None if failed
    """
    try:
        logger.info(f"Fetching data for {ticker} with period {period}")
        
        # Validate inputs
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Invalid ticker symbol")
        
        # Clean ticker symbol
        ticker = ticker.upper().strip()
        
        # Fetch data using yfinance
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval, auto_adjust=False)
        
        # Check if data is empty
        if df.empty:
            logger.warning(f"No data found for ticker {ticker}")
            return None
        
        # Reset index to make Date a column
        df.reset_index(inplace=True)
        
        # Standardize column names
        expected_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in expected_columns):
            logger.error(f"Missing required columns in data for {ticker}")
            return None
        
        # Select and reorder columns
        df = df[expected_columns]
        
        # Data cleaning
        df.dropna(inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        
        # Validate data quality
        if len(df) < 2:
            logger.warning(f"Insufficient data points for {ticker}: {len(df)} rows")
            return None
        
        # Check for valid price data
        if df["Close"].isnull().all() or (df["Close"] <= 0).all():
            logger.error(f"Invalid price data for {ticker}")
            return None
        
        # Remove any rows with invalid prices
        df = df[df["Close"] > 0]
        
        # Sort by date
        df = df.sort_values("Date").reset_index(drop=True)
        
        # Save data if requested
        if save:
            try:
                os.makedirs("data", exist_ok=True)
                filename = f"data/{ticker.replace('.', '_')}_stock_{period}.csv"
                df.to_csv(filename, index=False)
                logger.info(f"Data saved to {filename}")
            except Exception as e:
                logger.warning(f"Failed to save data: {e}")
        
        logger.info(f"Successfully fetched {len(df)} data points for {ticker}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching stock data for {ticker}: {e}")
        return None

def get_stock_info(ticker):
    """
    Get additional stock information
    
    Args:
        ticker (str): Stock ticker symbol
    
    Returns:
        dict: Stock information or empty dict if failed
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract relevant information
        stock_info = {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'marketCap': info.get('marketCap', 0),
            'currency': info.get('currency', 'USD'),
            'exchange': info.get('exchange', 'Unknown')
        }
        
        return stock_info
        
    except Exception as e:
        logger.error(f"Error fetching stock info for {ticker}: {e}")
        return {}

def fetch_multiple_stocks(tickers, period="6mo", interval="1d"):
    """
    Fetch data for multiple stocks
    
    Args:
        tickers (list): List of ticker symbols
        period (str): Time period
        interval (str): Data interval
    
    Returns:
        dict: Dictionary with ticker as key and DataFrame as value
    """
    results = {}
    
    for ticker in tickers:
        logger.info(f"Fetching data for {ticker}")
        data = fetch_full_stock_data(ticker, period, interval, save=False)
        if data is not None:
            results[ticker] = data
        else:
            logger.warning(f"Failed to fetch data for {ticker}")
    
    return results

def calculate_basic_metrics(df):
    """
    Calculate basic stock metrics
    
    Args:
        df (pandas.DataFrame): Stock data
    
    Returns:
        dict: Basic metrics
    """
    try:
        if df is None or df.empty:
            return {}
        
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
        
        metrics = {
            'current_price': current_price,
            'previous_price': prev_price,
            'price_change': current_price - prev_price,
            'price_change_pct': ((current_price - prev_price) / prev_price) * 100 if prev_price != 0 else 0,
            'high_52w': df['High'].max(),
            'low_52w': df['Low'].min(),
            'avg_volume': df['Volume'].mean(),
            'total_return': ((current_price - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100 if df['Close'].iloc[0] != 0 else 0
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {}
