import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Bullish Quant Dashboard", layout="wide")


def flatten_download(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def get_close_series(df):
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.squeeze()


def compute_rsi(close, window=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(df, window=14):
    high = df["High"]
    low = df["Low"]
    close = get_close_series(df)

    if isinstance(high, pd.DataFrame):
        high = high.iloc[:, 0]
    if isinstance(low, pd.DataFrame):
        low = low.iloc[:, 0]

    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window).mean()


def score_stock(df, spy):
    df = flatten_download(df).copy()
    close = get_close_series(df)

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    rsi = compute_rsi(close)
    atr = compute_atr(df)

    ret_3m = close.iloc[-1] / close.iloc[-60] - 1
    spy_ret = spy.iloc[-1] / spy.iloc[-60] - 1
    rs = ret_3m - spy_ret

    trend = 0
    if float(close.iloc[-1]) > float(sma50.iloc[-1]):
        trend += 1
    if float(close.iloc[-1]) > float(sma200.iloc[-1]):
        trend += 1

    dist_high = close.iloc[-1] / close.rolling(20).max().iloc[-1] - 1

    score = (
        trend * 30
        + (rs * 100) * 0.5
        + (-abs(dist_high) * 100) * 0.3
        + (float(rsi.iloc[-1]) / 100) * 20
    )

    support = float(sma50.iloc[-1])

    return {
        "score": float(score),
        "price": float(close.iloc[-1]),
        "support": support,
        "rsi": float(rsi.iloc[-1]),
        "rs": float(rs),
        "trend": int(trend),
        "dist_high": float(dist_high),
        "atr": float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else np.nan,
    }


def thesis_state(row):
    health = 50

    if row["trend"] == 2:
        health += 20
    if row["rs"] > 0:
        health += 15
    if row["price"] > row["support"]:
        health += 10
    if row["dist_high"] < -0.08:
        health -= 10

    health = max(0, min(100, health))

    if health > 70:
        state = "Active"
    elif health > 50:
        state = "At Risk"
    else:
        state = "Invalid"

    return state, health


st.title("Bullish Quant Stock Dashboard")

tickers = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "COST", "AMD",
    "NFLX", "ADBE", "INTC", "CSCO", "QCOM", "PEP", "TXN", "INTU", "AMAT", "PYPL"
]

spy_df = yf.download("SPY", period="1y", auto_adjust=True, progress=False)
spy_df = flatten_download(spy_df)
spy = get_close_series(spy_df)

data = []
prices = {}

for t in tickers:
    df = yf.download(t, period="1y", auto_adjust=True, progress=False)
    df = flatten_download(df)

    if len(df) < 200:
        continue

    try:
        result = score_stock(df, spy)
        state, health = thesis_state(result)

        result["ticker"] = t
        result["state"] = state
        result["health"] = health

        data.append(result)
        prices[t] = df
    except Exception:
        continue

if not data:
    st.error("No stock data loaded successfully.")
    st.stop()

ranked_df = pd.DataFrame(data).sort_values("score", ascending=False)

st.subheader("Top Bullish Setups")
st.dataframe(
    ranked_df[["ticker", "score", "state", "health", "price"]].reset_index(drop=True),
    use_container_width=True
)

selected = st.selectbox("Select Stock", ranked_df["ticker"])

row = ranked_df[ranked_df["ticker"] == selected].iloc[0]
chart_df = prices[selected].copy()
close = get_close_series(chart_df)

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=chart_df.index,
    open=chart_df["Open"],
    high=chart_df["High"],
    low=chart_df["Low"],
    close=chart_df["Close"],
    name=selected
))

fig.add_hline(y=row["support"], line_dash="dash", annotation_text="Support")

st.plotly_chart(fig, use_container_width=True)

st.subheader("Thesis")
st.write(f"**State:** {row['state']}")
st.write(f"**Health:** {row['health']}/100")
st.write("### What must happen")
st.write("- Hold above support")
st.write("- Continue outperforming SPY")
st.write("- Maintain trend")
st.write("### Exit if")
st.write("- Breaks support")
st.write("- Loses trend")
st.write("- Weak relative strength")