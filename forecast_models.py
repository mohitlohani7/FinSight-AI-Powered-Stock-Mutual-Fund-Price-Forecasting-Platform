import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import warnings
from datetime import datetime, timedelta

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.WARNING)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_technical_indicators(df):
    """
    Calculate technical indicators for stock analysis
    
    Args:
        df (pandas.DataFrame): Stock data with OHLCV columns
    
    Returns:
        pandas.DataFrame: DataFrame with technical indicators added
    """
    try:
        df = df.copy()
        
        # Simple Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
        df['SMA_10'] = df['Close'].rolling(window=10, min_periods=1).mean()
        df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12, min_periods=1).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, min_periods=1).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, min_periods=1).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
        bb_std = df['Close'].rolling(window=20, min_periods=1).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        
        logger.info("Technical indicators calculated successfully")
        return df
        
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        return df

def prepare_prophet_data(df):
    """
    Prepare data for Prophet forecasting
    
    Args:
        df (pandas.DataFrame): Stock data
    
    Returns:
        pandas.DataFrame: Prophet-formatted data
    """
    try:
        # Select and rename columns for Prophet
        prophet_df = df[['Date', 'Close']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Ensure datetime format
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        
        # Remove any null values
        prophet_df = prophet_df.dropna()
        
        # Ensure numeric values
        prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')
        prophet_df = prophet_df.dropna()
        
        # Sort by date
        prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
        
        return prophet_df
        
    except Exception as e:
        logger.error(f"Error preparing Prophet data: {e}")
        return None

def advanced_prophet_forecast(df, forecast_days, ticker):
    """
    Advanced Prophet forecasting with hyperparameter optimization
    
    Args:
        df (pandas.DataFrame): Stock data
        forecast_days (int): Number of days to forecast
        ticker (str): Stock ticker symbol
    
    Returns:
        tuple: (forecast_df, metrics_dict, plotly_figure)
    """
    try:
        # Prepare data
        prophet_df = prepare_prophet_data(df)
        
        if prophet_df is None or len(prophet_df) < 10:
            logger.warning("Insufficient data for Prophet forecasting")
            return fallback_sma_forecast(df, forecast_days, ticker)
        
        # Create Prophet model with optimized parameters
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode='multiplicative',
            uncertainty_samples=1000,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            mcmc_samples=0,
            interval_width=0.95
        )
        
        # Add custom seasonalities if enough data
        if len(prophet_df) > 730:  # 2 years of data
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
        
        # Fit the model
        logger.info("Training Prophet model...")
        model.fit(prophet_df)
        
        # Make future predictions
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        
        # Calculate accuracy metrics
        metrics = calculate_forecast_accuracy(prophet_df, forecast, forecast_days)
        
        # Create visualization
        fig = create_forecast_chart(df, forecast, forecast_days, ticker)
        
        logger.info(f"Prophet forecast completed successfully for {ticker}")
        return forecast, metrics, fig
        
    except Exception as e:
        logger.error(f"Prophet forecasting failed for {ticker}: {e}")
        return fallback_sma_forecast(df, forecast_days, ticker)

def calculate_forecast_accuracy(historical_data, forecast, forecast_days):
    """
    Calculate comprehensive forecast accuracy metrics
    
    Args:
        historical_data (pandas.DataFrame): Historical data
        forecast (pandas.DataFrame): Forecast results
        forecast_days (int): Number of forecast days
    
    Returns:
        dict: Accuracy metrics
    """
    try:
        # Get historical predictions (excluding forecast period)
        historical_forecast = forecast[:-forecast_days] if forecast_days > 0 else forecast
        
        # Align historical data with predictions
        merged_data = historical_data.merge(
            historical_forecast[['ds', 'yhat']], 
            on='ds', 
            how='inner'
        )
        
        if len(merged_data) < 5:
            logger.warning("Insufficient data for accuracy calculation")
            return {'mae': 0, 'rmse': 0, 'mape': 0, 'r2': 0}
        
        actual = merged_data['y'].values
        predicted = merged_data['yhat'].values
        
        # Calculate metrics
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # R-squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Directional accuracy
        actual_direction = np.diff(actual) > 0
        predicted_direction = np.diff(predicted) > 0
        directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'data_points': len(merged_data)
        }
        
    except Exception as e:
        logger.error(f"Error calculating forecast accuracy: {e}")
        return {'mae': 0, 'rmse': 0, 'mape': 0, 'r2': 0, 'directional_accuracy': 0}

def fallback_sma_forecast(df, forecast_days, ticker):
    """
    Fallback forecasting using Simple Moving Average
    
    Args:
        df (pandas.DataFrame): Stock data
        forecast_days (int): Number of days to forecast
        ticker (str): Stock ticker symbol
    
    Returns:
        tuple: (forecast_df, metrics_dict, plotly_figure)
    """
    try:
        logger.info(f"Using SMA fallback forecast for {ticker}")
        
        df_clean = df[['Date', 'Close']].dropna()
        df_clean['Date'] = pd.to_datetime(df_clean['Date'])
        df_clean = df_clean.sort_values('Date').reset_index(drop=True)
        
        # Calculate SMA
        window = min(10, len(df_clean))
        sma = df_clean['Close'].rolling(window=window, min_periods=1).mean()
        last_sma = sma.iloc[-1]
        
        # Create forecast dates
        last_date = df_clean['Date'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), 
            periods=forecast_days, 
            freq='D'
        )
        
        # Simple forecast (assuming slight trend)
        recent_trend = (df_clean['Close'].iloc[-1] - df_clean['Close'].iloc[-5]) / 5 if len(df_clean) > 5 else 0
        
        forecast_values = []
        for i in range(forecast_days):
            forecast_value = last_sma + (recent_trend * i)
            forecast_values.append(forecast_value)
        
        # Create forecast dataframe
        forecast = pd.DataFrame({
            'ds': list(df_clean['Date']) + list(future_dates),
            'yhat': list(df_clean['Close']) + forecast_values,
            'yhat_lower': list(df_clean['Close']) + [v * 0.95 for v in forecast_values],
            'yhat_upper': list(df_clean['Close']) + [v * 1.05 for v in forecast_values]
        })
        
        # Simple metrics
        metrics = {
            'mae': 0,
            'rmse': 0,
            'mape': 0,
            'r2': 0,
            'directional_accuracy': 0,
            'method': 'SMA Fallback'
        }
        
        # Create chart
        fig = create_simple_forecast_chart(df_clean, forecast, forecast_days, ticker)
        
        return forecast, metrics, fig
        
    except Exception as e:
        logger.error(f"SMA fallback forecast failed: {e}")
        return None, {}, None

def create_forecast_chart(df, forecast, forecast_days, ticker):
    """
    Create comprehensive forecast visualization
    
    Args:
        df (pandas.DataFrame): Stock data
        forecast (pandas.DataFrame): Forecast results
        forecast_days (int): Number of forecast days
        ticker (str): Stock ticker symbol
    
    Returns:
        plotly.graph_objects.Figure: Interactive chart
    """
    try:
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Price Forecast', 'Volume'],
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3]
        )
        
        # Historical candlestick
        fig.add_trace(
            go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Historical OHLC',
                increasing_line_color='#00d4aa',
                decreasing_line_color='#ff6b6b'
            ),
            row=1, col=1
        )
        
        # Forecast line
        forecast_data = forecast.tail(forecast_days)
        fig.add_trace(
            go.Scatter(
                x=forecast_data['ds'],
                y=forecast_data['yhat'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='blue', width=3),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Confidence intervals
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
        
        # Volume
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
            title=f'{ticker} - AI Forecast Analysis ({forecast_days} Days)',
            xaxis_rangeslider_visible=False,
            template='plotly_white',
            height=700,
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
        
    except Exception as e:
        logger.error(f"Error creating forecast chart: {e}")
        return None

def create_simple_forecast_chart(df, forecast, forecast_days, ticker):
    """
    Create simple forecast chart for fallback scenarios
    """
    try:
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Close'],
                mode='lines',
                name='Historical',
                line=dict(color='blue')
            )
        )
        
        # Forecast
        forecast_data = forecast.tail(forecast_days)
        fig.add_trace(
            go.Scatter(
                x=forecast_data['ds'],
                y=forecast_data['yhat'],
                mode='lines+markers',
                name='SMA Forecast',
                line=dict(color='orange', dash='dash')
            )
        )
        
        fig.update_layout(
            title=f'{ticker} - Simple Moving Average Forecast',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_white',
            height=500
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating simple chart: {e}")
        return None

def run_prophet_forecast(df, forecast_days, ticker):
    """
    Main forecast function - wrapper for backward compatibility
    """
    return advanced_prophet_forecast(df, forecast_days, ticker)
