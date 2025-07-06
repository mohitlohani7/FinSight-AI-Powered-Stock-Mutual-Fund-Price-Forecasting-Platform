import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
from datetime import datetime, timedelta
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.WARNING)

# Page Configuration
st.set_page_config(
    page_title="FinSight Pro - Stock Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .forecast-summary {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    .stSelectbox > div > div > div {
        background-color: #f8f9fa;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0; font-size: 2.5rem;">📈 FinSight Pro</h1>
    <p style="color: white; margin: 0; font-size: 1.2rem;">Professional AI-Powered Stock Forecasting Platform</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("🛠️ Configuration Panel")

# Stock Selection
st.sidebar.subheader("📊 Stock Selection")
stock_categories = {
    "🇮🇳 Indian Stocks (NSE)": {
        "Reliance Industries": "RELIANCE.NS",
        "Tata Consultancy Services": "TCS.NS",
        "Infosys": "INFY.NS",
        "ICICI Bank": "ICICIBANK.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "Adani Enterprises": "ADANIENT.NS",
        "Bharti Airtel": "BHARTIARTL.NS",
        "State Bank of India": "SBIN.NS"
    },
    "🇺🇸 US Stocks": {
        "Apple Inc.": "AAPL",
        "Microsoft": "MSFT",
        "Amazon": "AMZN",
        "Google (Alphabet)": "GOOGL",
        "Tesla": "TSLA",
        "Meta (Facebook)": "META",
        "Netflix": "NFLX",
        "NVIDIA": "NVDA"
    }
}

# Category selection
selected_category = st.sidebar.selectbox("Select Market:", list(stock_categories.keys()))
selected_stock = st.sidebar.selectbox("Select Stock:", list(stock_categories[selected_category].keys()))
ticker = stock_categories[selected_category][selected_stock]

# Custom ticker option
st.sidebar.markdown("---")
custom_ticker = st.sidebar.text_input("Or enter custom ticker:", placeholder="e.g., AAPL, TSLA")
if custom_ticker:
    ticker = custom_ticker.upper()
    selected_stock = f"Custom: {ticker}"

# Forecast Parameters
st.sidebar.subheader("⚙️ Forecast Parameters")
period_options = {
    "1 Month": "1mo",
    "3 Months": "3mo", 
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y"
}
selected_period = st.sidebar.selectbox("Historical Period:", list(period_options.keys()), index=2)
period = period_options[selected_period]

forecast_days = st.sidebar.slider("Forecast Days:", min_value=5, max_value=90, value=30, step=5)

# Advanced Options
st.sidebar.subheader("🔧 Advanced Options")
show_confidence_interval = st.sidebar.checkbox("Show Confidence Intervals", value=True)
show_volume = st.sidebar.checkbox("Show Volume Chart", value=True)
show_technical_indicators = st.sidebar.checkbox("Show Technical Indicators", value=True)

# Data fetching and processing functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(ticker, period):
    """Fetch stock data with comprehensive error handling"""
    try:
        # Clean the ticker symbol
        ticker = ticker.strip().upper()
        
        # Create yfinance ticker object
        stock = yf.Ticker(ticker)
        
        # Download data with error handling
        df = stock.history(period=period, auto_adjust=True, prepost=True)
        
        if df.empty:
            return None, f"No data found for ticker: {ticker}. Please check if the ticker symbol is correct."
        
        # Reset index to make Date a column
        df.reset_index(inplace=True)
        
        # Ensure proper column names
        expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if len(df.columns) >= 6:
            df.columns = expected_columns
        else:
            return None, f"Unexpected data format for ticker: {ticker}"
        
        # Data validation and cleaning
        if df['Close'].isnull().all():
            return None, "Invalid data: All closing prices are null"
        
        # Convert data types
        df['Date'] = pd.to_datetime(df['Date'])
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with null closing prices
        df = df.dropna(subset=['Close'])
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Final validation
        if len(df) < 5:
            return None, f"Insufficient data points ({len(df)}) for ticker: {ticker}"
        
        return df, None
    
    except Exception as e:
        error_msg = str(e)
        if "No data found" in error_msg:
            return None, f"Ticker '{ticker}' not found. Please verify the symbol is correct."
        elif "HTTPError" in error_msg or "ConnectionError" in error_msg:
            return None, "Network error: Please check your internet connection and try again."
        else:
            return None, f"Error fetching data for {ticker}: {error_msg}"

def calculate_technical_indicators(df):
    """Calculate technical indicators with error handling"""
    try:
        df = df.copy()
        
        # Ensure we have enough data for calculations
        if len(df) < 50:
            return df  # Return original df if not enough data for indicators
        
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        
        # Exponential Moving Average
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        
        # RSI Calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        
        # Avoid division by zero
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Fill any remaining NaN values
        df['RSI'] = df['RSI'].fillna(50)  # Neutral RSI value
        
        return df
        
    except Exception as e:
        st.warning(f"Could not calculate technical indicators: {str(e)}")
        return df

def run_prophet_forecast(df, forecast_days):
    """Run Prophet forecast with comprehensive error handling"""
    try:
        # Prepare data for Prophet with extensive validation
        prophet_df = df[['Date', 'Close']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Ensure proper datetime format
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], errors='coerce')
        
        # Convert Close prices to numeric, handling any string values
        prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')
        
        # Remove any rows with NaN values
        prophet_df = prophet_df.dropna()
        
        # Sort by date to ensure proper chronological order
        prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
        
        # Check for sufficient data
        if len(prophet_df) < 10:
            return None, None, "Insufficient data for forecasting (minimum 10 data points required)"
        
        # Check for duplicate dates and handle them
        if prophet_df['ds'].duplicated().any():
            prophet_df = prophet_df.groupby('ds').agg({'y': 'mean'}).reset_index()
        
        # Validate data types before Prophet
        if not pd.api.types.is_datetime64_any_dtype(prophet_df['ds']):
            return None, None, "Date column is not in datetime format"
        
        if not pd.api.types.is_numeric_dtype(prophet_df['y']):
            return None, None, "Price column is not numeric"
        
        # Create Prophet model with simpler configuration to avoid errors
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=len(prophet_df) > 14,  # Only if enough data
            yearly_seasonality=len(prophet_df) > 730,  # Only if enough data
            uncertainty_samples=50,  # Reduced for stability
            interval_width=0.8
        )
        
        # Fit model with error handling
        model.fit(prophet_df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=forecast_days, freq='D')
        
        # Make predictions
        forecast = model.predict(future)
        
        # Calculate accuracy metrics with better error handling
        try:
            # Get historical predictions (excluding forecast period)
            historical_forecast = forecast.iloc[:-forecast_days] if forecast_days > 0 else forecast
            
            # Ensure we have matching lengths
            min_len = min(len(prophet_df), len(historical_forecast))
            
            if min_len > 0:
                actual_values = prophet_df['y'].iloc[:min_len].values
                predicted_values = historical_forecast['yhat'].iloc[:min_len].values
                
                # Calculate metrics
                mae = mean_absolute_error(actual_values, predicted_values)
                rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
                
                # Calculate MAPE with protection against division by zero
                mape_values = np.abs((actual_values - predicted_values) / np.where(actual_values == 0, 1, actual_values))
                mape = np.mean(mape_values) * 100
                
                return forecast, (mae, rmse, mape), None
            else:
                return forecast, (0, 0, 0), None
                
        except Exception as metric_error:
            # Return forecast without metrics if calculation fails
            return forecast, (0, 0, 0), f"Metrics calculation warning: {str(metric_error)}"
        
    except Exception as e:
        error_msg = str(e)
        # Provide more specific error messages
        if "arg must be a list, tuple, 1-d array, or Series" in error_msg:
            return None, None, "Data format error: Please ensure your data contains valid numeric values"
        elif "empty" in error_msg.lower():
            return None, None, "No valid data available for forecasting"
        elif "datetime" in error_msg.lower():
            return None, None, "Date format error: Please check your date column"
        else:
            return None, None, f"Forecasting error: {error_msg}"

def create_main_chart(df, forecast=None, show_confidence=True, show_volume=True, show_technical=True):
    """Create comprehensive stock chart"""
    # Create subplot structure
    rows = 2 if show_volume else 1
    subplot_titles = ['Price & Forecast', 'Volume'] if show_volume else ['Price & Forecast']
    
    fig = make_subplots(
        rows=rows, cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3] if show_volume else [1.0]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC',
            increasing_line_color='#00d4aa',
            decreasing_line_color='#ff6b6b'
        ),
        row=1, col=1
    )
    
    # Technical indicators
    if show_technical and len(df) > 50:
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='red', width=1)
            ),
            row=1, col=1
        )
    
    # Forecast
    if forecast is not None:
        forecast_data = forecast.tail(forecast_days)
        
        fig.add_trace(
            go.Scatter(
                x=forecast_data['ds'],
                y=forecast_data['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Confidence intervals
        if show_confidence:
            fig.add_trace(
                go.Scatter(
                    x=forecast_data['ds'],
                    y=forecast_data['yhat_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_data['ds'],
                    y=forecast_data['yhat_lower'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(0, 100, 255, 0.2)',
                    name='Confidence Interval',
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
    
    # Volume chart
    if show_volume:
        fig.add_trace(
            go.Bar(
                x=df['Date'],
                y=df['Volume'],
                name='Volume',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f"{selected_stock} - Professional Analysis",
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        height=800 if show_volume else 600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Main application logic
if st.sidebar.button("🚀 Run Analysis", type="primary"):
    with st.spinner("Fetching real-time data..."):
        # Fetch data
        df, error = fetch_stock_data(ticker, period)
        
        if error:
            st.error(f"❌ {error}")
        else:
            # Calculate technical indicators
            df = calculate_technical_indicators(df)
            
            # Display success message
            st.success(f"✅ Successfully fetched {len(df)} data points for {selected_stock}")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100
            
            with col1:
                st.metric(
                    label="Current Price",
                    value=f"${current_price:.2f}",
                    delta=f"{price_change_pct:.2f}%"
                )
            
            with col2:
                st.metric(
                    label="Volume",
                    value=f"{df['Volume'].iloc[-1]:,.0f}"
                )
            
            with col3:
                st.metric(
                    label="52W High",
                    value=f"${df['High'].max():.2f}"
                )
            
            with col4:
                st.metric(
                    label="52W Low",
                    value=f"${df['Low'].min():.2f}"
                )
            
            # Run forecast
            with st.spinner("Running AI forecast..."):
                forecast, metrics, forecast_error = run_prophet_forecast(df, forecast_days)
                
                if forecast_error:
                    st.warning(f"⚠️ {forecast_error}")
                    forecast = None
                
                # Create and display main chart
                fig = create_main_chart(
                    df, 
                    forecast, 
                    show_confidence_interval, 
                    show_volume, 
                    show_technical_indicators
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast results
                if forecast is not None:
                    st.header("📈 Forecast Results")
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    mae, rmse, mape = metrics
                    
                    with col1:
                        st.metric("Mean Absolute Error", f"{mae:.2f}")
                    with col2:
                        st.metric("Root Mean Square Error", f"{rmse:.2f}")
                    with col3:
                        st.metric("Mean Absolute Percentage Error", f"{mape:.2f}%")
                    
                    # Forecast summary
                    forecast_summary = forecast.tail(forecast_days)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                    forecast_summary.columns = ['Date', 'Predicted Price', 'Lower Bound', 'Upper Bound']
                    forecast_summary['Date'] = forecast_summary['Date'].dt.date
                    
                    # Format currency
                    for col in ['Predicted Price', 'Lower Bound', 'Upper Bound']:
                        forecast_summary[col] = forecast_summary[col].apply(lambda x: f"${x:.2f}")
                    
                    st.subheader("📅 Detailed Forecast")
                    st.dataframe(forecast_summary, use_container_width=True)
                    
                    # Download button
                    csv = forecast_summary.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Forecast Data",
                        data=csv,
                        file_name=f"{ticker}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
                    # Price prediction summary
                    current_price = df['Close'].iloc[-1]
                    predicted_price = forecast['yhat'].iloc[-1]
                    price_direction = "📈 UP" if predicted_price > current_price else "📉 DOWN"
                    price_change_forecast = ((predicted_price - current_price) / current_price) * 100
                    
                    st.markdown(f"""
                    <div class="forecast-summary">
                        <h3>📊 Forecast Summary</h3>
                        <p><strong>Current Price:</strong> ${current_price:.2f}</p>
                        <p><strong>Predicted Price ({forecast_days} days):</strong> ${predicted_price:.2f}</p>
                        <p><strong>Expected Change:</strong> {price_direction} {abs(price_change_forecast):.2f}%</p>
                        <p><strong>Forecast Accuracy:</strong> MAPE {mape:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>📈 FinSight Pro - Professional Stock Forecasting Platform</p>
    <p><small>Disclaimer: This tool is for educational purposes only. Not financial advice.</small></p>
</div>
""", unsafe_allow_html=True)
