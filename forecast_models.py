import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os

def forecast_with_prophet(file_path):
    print(f"📊 Reading data from: {file_path}")
    df = pd.read_csv(file_path)

    df = df.rename(columns={"Date": "ds", "Close": "y"})
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.dropna(subset=["ds", "y"])

    # Prophet Model
    model = Prophet()
    model.fit(df)

    # Forecast
    future = model.make_future_dataframe(periods=10)
    forecast = model.predict(future)

    # Save forecast
    os.makedirs("forecast_output", exist_ok=True)
    forecast_file = f"forecast_output/forecast_{os.path.basename(file_path).split('_')[0]}.csv"
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(forecast_file, index=False)

    # 📌 Accuracy on training data
    df_forecast = forecast.set_index('ds')[['yhat']].join(df.set_index('ds')[['y']])
    df_forecast = df_forecast.dropna()
    mae = mean_absolute_error(df_forecast['y'], df_forecast['yhat'])
    rmse = np.sqrt(mean_squared_error(df_forecast['y'], df_forecast['yhat']))
    print(f"📏 MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # ✅ Plot 1: Candlestick with Volume
    df_ohlc = pd.read_csv(file_path)
    df_ohlc['Date'] = pd.to_datetime(df_ohlc['Date'])

    fig = go.Figure(data=[
        go.Candlestick(
            x=df_ohlc['Date'],
            open=df_ohlc['Open'],
            high=df_ohlc['High'],
            low=df_ohlc['Low'],
            close=df_ohlc['Close'],
            name="Candlestick"
        ),
        go.Bar(x=df_ohlc['Date'], y=df_ohlc['Volume'], name="Volume", marker_color="lightblue", yaxis='y2')
    ])

    fig.update_layout(
        title='📊 Stock OHLC + Volume',
        yaxis=dict(title='Price'),
        yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False),
        xaxis_rangeslider_visible=False,
        template='plotly_white'
    )
    fig.show()

    # ✅ Plot 2: Forecast Plot
    print("📈 Forecast plot")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines',
                              line_color='lightgray', name='Upper Bound'))
    fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty',
                              mode='lines', line_color='lightgray', name='Lower Bound'))
    fig2.update_layout(title='📈 Forecast with Confidence Interval', template='plotly_white')
    fig2.show()

    # ✅ Print last 10-day forecast
    print("\n📅 Final 10-day Forecast:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10).round(2))

    # ✅ Print accuracy
    print(f"\n📏 Accuracy on training data:")
    print(f"MAE (Mean Absolute Error): {mae:.2f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
