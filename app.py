import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Page setup
st.set_page_config(page_title="📈 Stock Forecasting App", layout="wide")
st.title("📈 Simple Stock Forecasting App")

# Sidebar
st.sidebar.header("Settings")

# Stock selection
stocks = {
    "TCS (NSE)": "TCS.NS",
    "Reliance (NSE)": "RELIANCE.NS",
    "Infosys (NSE)": "INFY.NS",
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Microsoft": "MSFT"
}

selected_stock = st.sidebar.selectbox("Select Stock:", list(stocks.keys()))
ticker = stocks[selected_stock]

period = st.sidebar.selectbox("Period:", ["1mo", "3mo", "6mo", "1y"], index=2)
forecast_days = st.sidebar.slider("Forecast Days:", 5, 30, 10)

def fetch_and_clean_data(ticker, period):
    """Fetch and clean stock data"""
    try:
        # Download data
        data = yf.download(ticker, period=period, progress=False)
        
        if data.empty:
            return None, f"No data found for {ticker}"
        
        # Reset index to make Date a column
        data.reset_index(inplace=True)
        
        # Ensure we have the right columns
        if 'Date' not in data.columns:
            data.reset_index(inplace=True)
        
        # Select only required columns
        data = data[['Date', 'Close']].copy()
        
        # Clean the data
        data['Date'] = pd.to_datetime(data['Date'])
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        
        # Remove any NaN values
        data = data.dropna()
        
        # Sort by date
        data = data.sort_values('Date').reset_index(drop=True)
        
        if len(data) < 10:
            return None, f"Insufficient data: only {len(data)} points available"
        
        return data, None
        
    except Exception as e:
        return None, f"Error: {str(e)}"

def simple_prophet_forecast(data, forecast_days):
    """Simple Prophet forecast with robust error handling"""
    try:
        # Prepare data for Prophet
        prophet_data = data[['Date', 'Close']].copy()
        prophet_data.columns = ['ds', 'y']
        
        # Ensure proper data types
        prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
        prophet_data['y'] = pd.to_numeric(prophet_data['y'], errors='coerce')
        
        # Remove any remaining NaN values
        prophet_data = prophet_data.dropna().reset_index(drop=True)
        
        # Check for sufficient data
        if len(prophet_data) < 10:
            return None, None, "Need at least 10 data points"
        
        # Create simple Prophet model
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            uncertainty_samples=0,  # Disable uncertainty for simplicity
            interval_width=0.8
        )
        
        # Fit the model
        model.fit(prophet_data)
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=forecast_days)
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Calculate simple accuracy metrics
        try:
            # Get historical predictions
            hist_forecast = forecast.iloc[:-forecast_days]
            actual = prophet_data['y'].values
            predicted = hist_forecast['yhat'].values
            
            # Ensure same length
            min_len = min(len(actual), len(predicted))
            actual = actual[:min_len]
            predicted = predicted[:min_len]
            
            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            
            return forecast, (mae, rmse), None
            
        except:
            return forecast, (0, 0), None
        
    except Exception as e:
        return None, None, f"Forecast error: {str(e)}"

def create_simple_chart(data, forecast, forecast_days):
    """Create a simple chart"""
    fig = go.Figure()
    
    # Plot historical data
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Close'],
        mode='lines',
        name='Historical',
        line=dict(color='blue')
    ))
    
    # Plot forecast
    if forecast is not None:
        forecast_data = forecast.tail(forecast_days)
        fig.add_trace(go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
    
    fig.update_layout(
        title=f'{selected_stock} - Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_white',
        height=500
    )
    
    return fig

# Main app
if st.sidebar.button("Run Forecast"):
    with st.spinner("Fetching data..."):
        data, error = fetch_and_clean_data(ticker, period)
        
        if error:
            st.error(f"❌ {error}")
        else:
            st.success(f"✅ Fetched {len(data)} data points")
            
            # Show current price info
            current_price = data['Close'].iloc[-1]
            st.metric("Current Price", f"${current_price:.2f}")
            
            # Run forecast
            with st.spinner("Generating forecast..."):
                forecast, metrics, forecast_error = simple_prophet_forecast(data, forecast_days)
                
                if forecast_error:
                    st.error(f"❌ {forecast_error}")
                else:
                    # Create chart
                    fig = create_simple_chart(data, forecast, forecast_days)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show metrics
                    if metrics and metrics[0] > 0:
                        mae, rmse = metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("MAE", f"{mae:.2f}")
                        with col2:
                            st.metric("RMSE", f"{rmse:.2f}")
                    
                    # Show forecast table
                    if forecast is not None:
                        st.subheader("Forecast Data")
                        forecast_table = forecast[['ds', 'yhat']].tail(forecast_days).copy()
                        forecast_table.columns = ['Date', 'Predicted Price']
                        forecast_table['Date'] = forecast_table['Date'].dt.date
                        forecast_table['Predicted Price'] = forecast_table['Predicted Price'].round(2)
                        st.dataframe(forecast_table)
                        
                        # Download button
                        csv = forecast_table.to_csv(index=False)
                        st.download_button(
                            "Download Forecast",
                            csv,
                            file_name=f"{ticker}_forecast.csv",
                            mime="text/csv"
                        )

# Instructions
st.sidebar.markdown("---")
st.sidebar.markdown("""
### How to use:
1. Select a stock from the dropdown
2. Choose the historical period
3. Set forecast days (5-30)
4. Click 'Run Forecast'
""")

st.sidebar.markdown("---")
st.sidebar.markdown("⚠️ **Disclaimer**: This is for educational purposes only. Not financial advice.")
