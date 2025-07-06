import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------- Streamlit Config --------
st.set_page_config(page_title="📈 FinSight Forecast", layout="wide")
st.title("🔮 FinSight: Real-Time AI Stock Forecasting")

# -------- Sidebar UI --------
st.sidebar.header("🛠️ Forecast Settings")

stock_dict = {
    "Reliance (NSE)": "RELIANCE.NS",
    "TCS (NSE)": "TCS.NS",
    "Infosys (NSE)": "INFY.NS",
    "ICICI Bank (NSE)": "ICICIBANK.NS",
    "HDFC Bank (NSE)": "HDFCBANK.NS",
    "Adani Ent. (NSE)": "ADANIENT.NS",
    "Apple Inc.": "AAPL",
    "Tesla": "TSLA",
    "Amazon": "AMZN",
    "Google": "GOOGL",
    "Microsoft": "MSFT",
    "Custom Ticker (Type Below)": None
}

selected_stock = st.sidebar.selectbox("📌 Choose a Stock", list(stock_dict.keys()))
custom_input = st.sidebar.text_input("Or type any valid Yahoo Finance Ticker:", "RELIANCE.NS")
ticker = stock_dict[selected_stock] if stock_dict[selected_stock] else custom_input.strip()

period = st.sidebar.selectbox("🕒 Historical Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
forecast_days = st.sidebar.slider("🔮 Forecast Days", 5, 30, 10)

# -------- Fetch & Forecast --------
if st.sidebar.button("📥 Fetch & Forecast"):
    st.info(f"Fetching data for `{ticker}` | Period: `{period}`...")

    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True)
        df.reset_index(inplace=True)

        if df.empty or df['Close'].isnull().all():
            st.error("⚠️ No valid stock data found. Please try a different ticker or time range.")
        else:
            st.success("✅ Real-time data downloaded!")

            # -------- Candlestick + Volume Plot --------
            st.subheader("📊 OHLC + Volume Chart")

            fig = go.Figure()

            fig.add_trace(go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Candlestick'
            ))

            fig.add_trace(go.Bar(
                x=df['Date'],
                y=df['Volume'],
                name='Volume',
                marker_color='lightblue',
                yaxis='y2'
            ))

            fig.update_layout(
                title=f"{ticker} - Candlestick with Volume",
                xaxis=dict(title='Date'),
                yaxis=dict(title='Price'),
                yaxis2=dict(
                    title='Volume',
                    overlaying='y',
                    side='right',
                    showgrid=False
                ),
                legend=dict(x=0, y=1.2, orientation="h"),
                xaxis_rangeslider_visible=False,
                template="plotly_white",
                height=600
            )

            st.plotly_chart(fig, use_container_width=True)

            # -------- Prophet Forecast --------
            st.subheader(f"📈 Forecast for Next {forecast_days} Days")

            prophet_df = df[['Date', 'Close']].rename(columns={"Date": "ds", "y": "y"})
            prophet_df.dropna(inplace=True)

            model = Prophet()
            model.fit(prophet_df)

            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)

            fig2 = plot_plotly(model, forecast)
            st.plotly_chart(fig2, use_container_width=True)

            # -------- Forecast Accuracy --------
            df_forecast = forecast.set_index('ds')[['yhat']].join(prophet_df.set_index('ds')[['y']])
            df_forecast.dropna(inplace=True)

            if not df_forecast.empty:
                mae = mean_absolute_error(df_forecast['y'], df_forecast['yhat'])
                rmse = np.sqrt(mean_squared_error(df_forecast['y'], df_forecast['yhat']))
                st.success(f"📏 Forecast Accuracy:\n- MAE: `{mae:.2f}`\n- RMSE: `{rmse:.2f}`")
            else:
                st.warning("⚠️ Accuracy metrics could not be calculated.")

            # -------- Forecast Table & Download --------
            st.subheader("📅 Forecast Table")
            forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days).round(2)
            st.dataframe(forecast_table)

            csv = forecast_table.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📁 Download Forecast CSV",
                data=csv,
                file_name=f"{ticker}_forecast.csv",
                mime='text/csv'
            )

    except Exception as e:
        st.error(f"❌ Error occurred: {e}")
