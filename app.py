import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------- Streamlit Config ---------
st.set_page_config(page_title="📈 FinSight - Real-Time Stock Forecasting", layout="wide")
st.title("🔮 FinSight: Real-Time AI Stock Forecasting (All Stocks)")

# --------- Sidebar ---------
st.sidebar.header("🛠️ Forecast Settings")

ticker = st.sidebar.text_input("Enter Ticker (e.g. RELIANCE.NS, AAPL, INFY.NS):", "RELIANCE.NS")
period = st.sidebar.selectbox("Select Historical Data Period:", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
forecast_days = st.sidebar.slider("🔮 Days to Forecast Ahead", min_value=5, max_value=30, value=10)

if st.sidebar.button("📥 Fetch & Forecast"):
    st.info(f"Fetching real-time data for `{ticker}` over `{period}`...")
    
    try:
        # --------- Download Stock Data ---------
        df = yf.download(ticker, period=period)
        df.reset_index(inplace=True)

        if df.empty:
            st.warning("⚠️ No data returned. Please check the ticker symbol.")
        else:
            st.success("✅ Real-time data fetched successfully!")

            # --------- OHLC + Volume Chart ---------
            st.subheader("📊 Historical Candlestick + Volume")
            fig = go.Figure(data=[
                go.Candlestick(
                    x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name="OHLC"
                ),
                go.Bar(
                    x=df['Date'], y=df['Volume'],
                    name='Volume', yaxis='y2',
                    marker_color='lightblue'
                )
            ])
            fig.update_layout(
                yaxis_title="Price",
                title=f"{ticker} - OHLC + Volume",
                yaxis2=dict(overlaying='y', side='right', showgrid=False),
                xaxis_rangeslider_visible=False,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

            # --------- Prophet Forecasting ---------
            st.subheader(f"📈 {forecast_days}-Day Forecast using Prophet")
            prophet_df = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
            prophet_df.dropna(inplace=True)

            model = Prophet()
            model.fit(prophet_df)

            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)

            fig2 = plot_plotly(model, forecast)
            st.plotly_chart(fig2, use_container_width=True)

            # --------- Accuracy Metrics ---------
            df_forecast = forecast.set_index('ds')[['yhat']].join(prophet_df.set_index('ds')[['y']])
            df_forecast.dropna(inplace=True)

            mae = mean_absolute_error(df_forecast['y'], df_forecast['yhat'])
            rmse = np.sqrt(mean_squared_error(df_forecast['y'], df_forecast['yhat']))

            st.success(f"📏 Forecast Accuracy:\n- MAE: `{mae:.2f}`\n- RMSE: `{rmse:.2f}`")

            # --------- Forecast Table ---------
            st.subheader("📅 Forecast Table (Final Days)")
            forecast_view = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days).round(2)
            st.dataframe(forecast_view)

            # --------- Download Option ---------
            csv = forecast_view.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📁 Download Forecast as CSV",
                data=csv,
                file_name=f'{ticker}_forecast.csv',
                mime='text/csv',
            )

    except Exception as e:
        st.error(f"❌ Error occurred: {e}")
