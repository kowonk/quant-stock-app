import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Bullish Quant Dashboard", layout="wide")

# -----------------------------
# Helper functions
# -----------------------------
def compute_rsi(close, window=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_atr(df, window=14):
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window).mean()

def score_stock(df, spy):
    close = df["Close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    rsi = compute_rsi(close)
    atr = compute_atr(df)

    # Returns
    ret_3m = close.iloc[-1] / close.iloc[-60] - 1
    spy_ret = spy.iloc[-1] / spy.iloc[-60] - 1
    rs = ret_3m - spy_ret

    # Trend score
    trend = 0
    if close.iloc[-1] > sma50.iloc[-1]:
        trend += 1
    if close.iloc[-1] > sma200.iloc[-1]:
        trend += 1

    # Pullback score
    dist_high = close.iloc[-1] / close.rolling(20).max().iloc[-1] - 1

    score = (
        trend * 30 +
        (rs * 100) * 0.5 +
        (-abs(dist_high) * 100) * 0.3 +
        (rsi.iloc[-1] / 100) * 20
    )

    support = sma50.iloc[-1]

    return {
        "score": score,
        "price": close.iloc[-1],
        "support": support,
        "rsi": rsi.iloc[-1],
        "rs": rs,
        "trend": trend,
        "dist_high": dist_high,
        "atr": atr.iloc[-1]
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

# -----------------------------
# UI
# -----------------------------
st.title("Bullish Quant Stock Dashboard")

tickers = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO","COST","AMD",
    "NFLX","ADBE","INTC","CSCO","QCOM","PEP","TXN","INTU","AMAT","PYPL"
]

spy = yf.download("SPY", period="1y")["Close"]

data = []
prices = {}

for t in tickers:
    df = yf.download(t, period="1y")
    if len(df) < 200:
        continue

    result = score_stock(df, spy)
    state, health = thesis_state(result)

    result["ticker"] = t
    result["state"] = state
    result["health"] = health

    data.append(result)
    prices[t] = df

df = pd.DataFrame(data).sort_values("score", ascending=False)

st.subheader("Top Bullish Setups")
st.dataframe(df[["ticker","score","state","health","price"]])

selected = st.selectbox("Select Stock", df["ticker"])

row = df[df["ticker"] == selected].iloc[0]
chart_df = prices[selected]

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=chart_df.index,
    open=chart_df["Open"],
    high=chart_df["High"],
    low=chart_df["Low"],
    close=chart_df["Close"]
))

fig.add_hline(y=row["support"], line_dash="dash")

st.plotly_chart(fig)

st.subheader("Thesis")

st.write(f"State: {row['state']}")
st.write(f"Health: {row['health']}/100")

st.write("### What must happen")
st.write("- Hold above support")
st.write("- Continue outperforming SPY")
st.write("- Maintain trend")

st.write("### Exit if")
st.write("- Breaks support")
st.write("- Loses trend")
st.write("- Weak relative strength")