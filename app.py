import streamlit as st
import pandas as pd
import os
from data_collection import fetch_full_stock_data
from forecast_models import run_prophet_forecast
import plotly.graph_objects as go

# --------- Streamlit Config ---------
st.set_page_config(page_title="📈 FinSight Forecast", layout="wide")
st.title("🔮 FinSight: Real‑Time AI Stock Forecasting")

# --------- Load Stock List ---------
st.sidebar.header("📘 Select Stock")
try:
    stock_df = pd.read_csv("stock_list.csv")
    if stock_df.empty or 'Company Name' not in stock_df.columns or 'Ticker' not in stock_df.columns:
        raise ValueError("stock_list.csv must have 'Company Name' & 'Ticker' columns.")
    selected = st.sidebar.selectbox("Choose a company:", stock_df["Company Name"])
    ticker = stock_df.loc[stock_df["Company Name"] == selected, "Ticker"].iat[0]
except Exception as e:
    st.sidebar.error(f"❌ Could not load stock list: {e}")
    st.stop()

# --------- Forecast Settings ---------
st.sidebar.header("⚙️ Forecast Settings")
period = st.sidebar.selectbox("Historical period:", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
forecast_days = st.sidebar.slider("Days to forecast ahead:", 5, 30, 10)

# --------- Fetch & Forecast ---------
if st.sidebar.button("📥 Fetch & Forecast"):
    st.info(f"Fetching `{ticker}` for period `{period}`…")

    # 1) Download data
    df = fetch_full_stock_data(ticker, period=period)
    if df is None or df.empty:
        st.error("❌ Could not fetch data. Check your ticker or network and try again.")
        st.stop()
    st.success("✅ Data fetched successfully!")

    # Save raw data
    os.makedirs("data", exist_ok=True)
    df.to_csv(f"data/{ticker.replace('.', '_')}.csv", index=False)

    # 2) Historical OHLC + Volume chart
    st.subheader("📊 Historical OHLC + Volume")
    fig1 = go.Figure(data=[
        go.Candlestick(
            x=df["Date"], open=df["Open"],
            high=df["High"], low=df["Low"],
            close=df["Close"], name="OHLC"
        ),
        go.Bar(
            x=df["Date"], y=df["Volume"],
            yaxis="y2", name="Volume",
            marker_color="lightblue"
        )
    ])
    fig1.update_layout(
        yaxis2=dict(overlaying="y", side="right", showgrid=False),
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=600
    )
    st.plotly_chart(fig1, use_container_width=True)

    # 3) Forecast with Prophet
    st.subheader(f"📈 {forecast_days}-Day Forecast")
    forecast, (mae, rmse), fig2 = run_prophet_forecast(df, forecast_days, ticker)

    if forecast is None:
        st.warning("⚠️ Not enough historical data to forecast (need at least 2 data points).")
    else:
        st.plotly_chart(fig2, use_container_width=True)
        st.success(f"📏 Forecast Accuracy — MAE: {mae:.2f}, RMSE: {rmse:.2f}")

        # 4) Forecast table + download
        st.subheader("📅 Forecast Table")
        table = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_days).round(2)
        st.dataframe(table)

        csv = table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📁 Download Forecast CSV",
            data=csv,
            file_name=f"{ticker}_forecast.csv",
            mime="text/csv"
        )
