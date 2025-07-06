import streamlit as st
import pandas as pd
import os
import logging
from datetime import datetime, timedelta
from data_collection import fetch_full_stock_data, get_stock_info, calculate_basic_metrics
from forecast_models import run_prophet_forecast, calculate_technical_indicators
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------- Streamlit Config ---------
st.set_page_config(
    page_title="📈 FinSight Pro - Stock Forecasting", 
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .forecast-summary {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #e9ecef;
        margin: 2rem 0;
    }
    .stSelectbox > div > div > div {
        background-color: #f8f9fa;
        border-radius: 8px;
    }
    .success-alert {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .error-alert {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0; font-size: 2.5rem; font-weight: 700;">📈 FinSight Pro</h1>
    <p style="color: white; margin: 0; font-size: 1.2rem; opacity: 0.9;">Professional AI-Powered Stock Forecasting Platform</p>
</div>
""", unsafe_allow_html=True)

# --------- Enhanced Sidebar ---------
st.sidebar.header("🛠️ Configuration Panel")

# Load Stock List with error handling
@st.cache_data
def load_stock_list():
    try:
        if os.path.exists("stock_list.csv"):
            stock_df = pd.read_csv("stock_list.csv")
            if stock_df.empty or 'Company Name' not in stock_df.columns or 'Ticker' not in stock_df.columns:
                raise ValueError("Invalid stock_list.csv format")
            return stock_df
        else:
            # Create default stock list if file doesn't exist
            default_stocks = {
                'Company Name': [
                    'TCS (NSE)', 'Reliance Industries (NSE)', 'Infosys (NSE)', 
                    'ICICI Bank (NSE)', 'HDFC Bank (NSE)', 'Apple Inc.', 
                    'Amazon.com Inc.', 'Google (Alphabet)', 'Microsoft Corp.', 'Tesla Inc.'
                ],
                'Ticker': [
                    'TCS.NS', 'RELIANCE.NS', 'INFY.NS', 'ICICIBANK.NS', 
                    'HDFCBANK.NS', 'AAPL', 'AMZN', 'GOOGL', 'MSFT', 'TSLA'
                ]
            }
            return pd.DataFrame(default_stocks)
    except Exception as e:
        logger.error(f"Error loading stock list: {e}")
        return pd.DataFrame({'Company Name': ['Error'], 'Ticker': ['AAPL']})

# Stock Selection
st.sidebar.subheader("📊 Stock Selection")
stock_df = load_stock_list()

# Category-based stock selection
indian_stocks = stock_df[stock_df['Ticker'].str.contains('.NS', na=False)]
us_stocks = stock_df[~stock_df['Ticker'].str.contains('.NS', na=False)]

stock_category = st.sidebar.selectbox(
    "Select Market:",
    ["🇮🇳 Indian Stocks (NSE)", "🇺🇸 US Stocks", "📋 All Stocks"]
)

if stock_category == "🇮🇳 Indian Stocks (NSE)":
    display_stocks = indian_stocks
elif stock_category == "🇺🇸 US Stocks":
    display_stocks = us_stocks
else:
    display_stocks = stock_df

try:
    selected = st.sidebar.selectbox("Choose a company:", display_stocks["Company Name"])
    ticker = display_stocks.loc[display_stocks["Company Name"] == selected, "Ticker"].iloc[0]
except Exception as e:
    st.sidebar.error(f"❌ Error selecting stock: {e}")
    ticker = "AAPL"  # Default fallback

# Custom ticker input
st.sidebar.markdown("---")
custom_ticker = st.sidebar.text_input("Or enter custom ticker:", placeholder="e.g., AAPL, TSLA")
if custom_ticker:
    ticker = custom_ticker.upper()
    selected = f"Custom: {ticker}"

# --------- Enhanced Forecast Settings ---------
st.sidebar.subheader("⚙️ Forecast Parameters")

period_options = {
    "1 Month": "1mo",
    "3 Months": "3mo", 
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y"
}
