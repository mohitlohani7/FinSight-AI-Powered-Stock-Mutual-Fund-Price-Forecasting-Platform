import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

def fallback_sma_forecast(df, forecast_days, ticker):
    """
    Fallback simple moving average forecast when Prophet fails.
    """
    df = df.copy()
    df = df[['Date', 'Close']].dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    sma = df['Close'].rolling(window=5, min_periods=1).mean()
    last_sma = sma.iloc[-1]

    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast = pd.DataFrame({
        'ds': future_dates,
        'yhat': [last_sma] * forecast_days,
        'yhat_lower': [last_sma * 0.98] * forecast_days,
        'yhat_upper': [last_sma * 1.02] * forecast_days
    })

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Actual'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='SMA Forecast', line=dict(color='orange')))
    fig.update_layout(title=f"{ticker} - SMA Forecast ({forecast_days} Days)", template="plotly_white")

    return forecast, (0, 0), fig


def run_prophet_forecast(df, forecast_days, ticker):
    """
    Main forecast function: tries Prophet, falls back to SMA if Prophet fails.
    """
    try:
        if df is None or df.empty or len(df) < 2:
            return fallback_sma_forecast(df, forecast_days, ticker)

        prophet_df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')
        prophet_df.dropna(inplace=True)

        model = Prophet()
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)

        # Accuracy
        hist_pred = model.predict(prophet_df)[['ds', 'yhat']].set_index('ds')
        actual = prophet_df.set_index('ds')['y']
        merged = hist_pred.join(actual, how='inner').dropna()
        mae = mean_absolute_error(merged['y'], merged['yhat'])
        rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Actual'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
        fig.update_layout(title=f"{ticker} Forecast ({forecast_days} Days)", template='plotly_white')

        return forecast, (mae, rmse), fig

    except Exception as e:
        print(f"⚠️ Prophet failed: {e}. Using SMA fallback.")
        return fallback_sma_forecast(df, forecast_days, ticker)
