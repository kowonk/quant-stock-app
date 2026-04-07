import warnings
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Quant Stock Thesis Platform", layout="wide")

# =========================================================
# CONFIG
# =========================================================
LOOKBACK_DAYS = 520
MIN_HISTORY = 220
DEFAULT_TOP_N = 10
DEFAULT_HOLD_DAYS = 20
UNIVERSE_MAP = {
    "Magnificent / Mega Cap": [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "AMD", "NFLX",
        "COST", "QCOM", "ADBE", "INTU", "AMAT", "TXN", "CSCO", "PEP", "LIN", "ISRG",
    ],
    "Large Cap Growth": [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "AMD", "NFLX",
        "ADBE", "INTU", "AMAT", "QCOM", "NOW", "CRM", "PANW", "ANET", "SHOP", "UBER",
        "ORCL", "MU", "KLAC", "LRCX", "MELI", "CRWD", "PLTR", "TTD", "SNOW", "CDNS",
    ],
    "Broad Liquid List": [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "AMD", "NFLX",
        "ADBE", "INTU", "AMAT", "QCOM", "NOW", "CRM", "PANW", "ANET", "SHOP", "UBER",
        "ORCL", "MU", "KLAC", "LRCX", "MELI", "CRWD", "PLTR", "TTD", "SNOW", "CDNS",
        "JPM", "GS", "V", "MA", "AXP", "BKNG", "LLY", "UNH", "ISRG", "CAT",
        "DE", "GE", "ETN", "PH", "HON", "XOM", "CVX", "COST", "WMT", "HD",
    ],
}
SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Information Technology": "XLK",
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Financial Services": "XLF",
    "Financial": "XLF",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Real Estate": "XLRE",
}


# =========================================================
# HELPERS
# =========================================================
def clamp(x: float, low: float = 0.0, high: float = 100.0) -> float:
    return float(max(low, min(high, x)))


def flatten_df(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def get_close_series(df: pd.DataFrame) -> pd.Series:
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.squeeze()


def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def score_from_cross_section(s: pd.Series) -> pd.Series:
    return (50 + 15 * zscore(s)).clip(0, 100)


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = get_close_series(df)
    if isinstance(high, pd.DataFrame):
        high = high.iloc[:, 0]
    if isinstance(low, pd.DataFrame):
        low = low.iloc[:, 0]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def rolling_slope(series: pd.Series, window: int = 20) -> pd.Series:
    x = np.arange(window)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def _calc(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        y_mean = values.mean()
        return float(((x - x_mean) * (values - y_mean)).sum() / denom)

    return series.rolling(window).apply(_calc, raw=True)


def max_drawdown(close: pd.Series) -> float:
    peak = close.cummax()
    dd = close / peak - 1
    return float(dd.min()) if len(dd) else np.nan


def normalize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = flatten_df(df).copy()
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[cols].dropna().copy()


# =========================================================
# DATA LOADERS
# =========================================================
@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_prices(tickers: List[str], period_years: int = 2) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    start = date.today() - timedelta(days=365 * period_years + 40)
    for ticker in tickers:
        try:
            raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
            df = normalize_price_columns(raw)
            if len(df) >= MIN_HISTORY:
                out[ticker] = df
        except Exception:
            continue
    return out


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_benchmark_panel(period_years: int = 2) -> Dict[str, pd.DataFrame]:
    tickers = ["SPY", "QQQ", "XLK", "XLC", "XLY", "XLP", "XLF", "XLV", "XLI", "XLE", "XLU", "XLB", "XLRE"]
    return load_prices(tickers, period_years)


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def get_ticker_meta(ticker: str) -> Dict[str, str]:
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {}
    return {
        "shortName": info.get("shortName", ticker),
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", "Unknown"),
    }


# =========================================================
# FEATURE ENGINE
# =========================================================
def market_regime_score(bench: Dict[str, pd.DataFrame]) -> Tuple[float, Dict[str, float]]:
    spy = get_close_series(bench["SPY"])
    qqq = get_close_series(bench["QQQ"])

    def regime_one(close: pd.Series) -> Dict[str, float]:
        return {
            "above50": float(close.iloc[-1] > close.rolling(50).mean().iloc[-1]),
            "above200": float(close.iloc[-1] > close.rolling(200).mean().iloc[-1]),
            "ret20": float(close.iloc[-1] / close.iloc[-21] - 1),
            "vol20": float(close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)),
        }

    spyf = regime_one(spy)
    qqqf = regime_one(qqq)

    sector_breadth = []
    for etf in ["XLK", "XLC", "XLY", "XLP", "XLF", "XLV", "XLI", "XLE", "XLU", "XLB", "XLRE"]:
        if etf in bench:
            s = get_close_series(bench[etf])
            sector_breadth.append(float(s.iloc[-1] > s.rolling(50).mean().iloc[-1]))
    breadth = float(np.mean(sector_breadth)) if sector_breadth else 0.5

    score = 0.0
    score += 15 * spyf["above50"]
    score += 15 * spyf["above200"]
    score += 15 * qqqf["above50"]
    score += 15 * qqqf["above200"]
    score += 15 * clamp((spyf["ret20"] + 0.05) / 0.12 * 100) / 100
    score += 15 * clamp((qqqf["ret20"] + 0.05) / 0.12 * 100) / 100
    score += 10 * breadth
    avg_vol = np.mean([spyf["vol20"], qqqf["vol20"]])
    score += 10 * clamp((0.35 - avg_vol) / 0.25 * 100) / 100

    diag = {
        "SPY 20D Return": spyf["ret20"],
        "QQQ 20D Return": qqqf["ret20"],
        "Breadth": breadth,
        "Avg Index Vol": avg_vol,
    }
    return clamp(score), diag


def feature_row(ticker: str, df: pd.DataFrame, bench: Dict[str, pd.DataFrame], regime: float) -> Dict[str, float]:
    close = get_close_series(df)
    vol = df["Volume"]
    spy = get_close_series(bench["SPY"]).reindex(close.index).ffill()
    qqq = get_close_series(bench["QQQ"]).reindex(close.index).ffill()

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    rsi14 = compute_rsi(close)
    atr14 = compute_atr(df)
    ret = close.pct_change()
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    vol60 = ret.rolling(60).std() * np.sqrt(252)

    ret1m = close.iloc[-1] / close.iloc[-22] - 1
    ret3m = close.iloc[-1] / close.iloc[-63] - 1
    ret6m = close.iloc[-1] / close.iloc[-126] - 1
    rs_spy = ret3m - (spy.iloc[-1] / spy.iloc[-63] - 1)
    rs_qqq = ret3m - (qqq.iloc[-1] / qqq.iloc[-63] - 1)

    rel_line = (close / close.iloc[0]) / (spy / spy.iloc[0])
    rs_slope = rolling_slope(rel_line, 20).iloc[-1]

    avg_dollar_vol = float((close * vol).rolling(50).mean().iloc[-1])
    recent_high20 = float(close.rolling(20).max().iloc[-1])
    dist_high20 = float(close.iloc[-1] / recent_high20 - 1)
    support = float(np.nanmax([sma20.iloc[-1], sma50.iloc[-1]]))
    support_atr = float((close.iloc[-1] - support) / atr14.iloc[-1]) if pd.notna(atr14.iloc[-1]) and atr14.iloc[-1] > 0 else np.nan
    downside = float((close.iloc[-1] - support) / close.iloc[-1]) if support > 0 else np.nan
    upside = float((recent_high20 - close.iloc[-1]) / close.iloc[-1]) if close.iloc[-1] > 0 else np.nan
    rr_proxy = upside / downside if downside not in [0, np.nan] else np.nan

    up_vol = vol[ret > 0].rolling(20).mean().iloc[-1] if (ret > 0).any() else np.nan
    down_vol = vol[ret < 0].rolling(20).mean().iloc[-1] if (ret < 0).any() else np.nan
    participation = float(up_vol / down_vol) if pd.notna(up_vol) and pd.notna(down_vol) and down_vol != 0 else 1.0

    return {
        "Ticker": ticker,
        "Close": float(close.iloc[-1]),
        "SMA20": float(sma20.iloc[-1]),
        "SMA50": float(sma50.iloc[-1]),
        "SMA200": float(sma200.iloc[-1]),
        "ATR14": float(atr14.iloc[-1]) if pd.notna(atr14.iloc[-1]) else np.nan,
        "RSI14": float(rsi14.iloc[-1]),
        "1M": float(ret1m),
        "3M": float(ret3m),
        "6M": float(ret6m),
        "RS_SPY": float(rs_spy),
        "RS_QQQ": float(rs_qqq),
        "RS_SLOPE": float(rs_slope) if pd.notna(rs_slope) else 0.0,
        "AVG_DOLLAR_VOL": avg_dollar_vol,
        "DIST_HIGH20": dist_high20,
        "SUPPORT": support,
        "SUPPORT_ATR": support_atr,
        "UPSIDE": upside,
        "DOWNSIDE": downside,
        "RR_PROXY": float(rr_proxy) if pd.notna(rr_proxy) else 0.0,
        "PARTICIPATION": participation,
        "VOL20": float(vol20.iloc[-1]) if pd.notna(vol20.iloc[-1]) else np.nan,
        "VOL60": float(vol60.iloc[-1]) if pd.notna(vol60.iloc[-1]) else np.nan,
        "MAXDD_6M": max_drawdown(close.iloc[-126:]) if len(close) >= 126 else np.nan,
        "REGIME": regime,
    }


def rank_features(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    ranked = ranked[ranked["AVG_DOLLAR_VOL"] > 15_000_000].copy()

    ranked["Trend Score"] = (
        score_from_cross_section(ranked["Close"] / ranked["SMA50"] - 1)
        + score_from_cross_section(ranked["Close"] / ranked["SMA200"] - 1)
        + score_from_cross_section(ranked["3M"])
    ) / 3

    ranked["Relative Strength Score"] = (
        score_from_cross_section(ranked["RS_SPY"])
        + score_from_cross_section(ranked["RS_QQQ"])
        + score_from_cross_section(ranked["RS_SLOPE"])
    ) / 3

    ranked["Pullback Score"] = (
        score_from_cross_section(-abs(ranked["DIST_HIGH20"] + 0.03))
        + score_from_cross_section(-abs(ranked["SUPPORT_ATR"] - 1.0))
        + score_from_cross_section(-abs(ranked["RSI14"] - 58))
    ) / 3

    ranked["Participation Score"] = (
        score_from_cross_section(np.log(ranked["AVG_DOLLAR_VOL"]))
        + score_from_cross_section(ranked["PARTICIPATION"])
    ) / 2

    ranked["Risk/Reward Score"] = (
        score_from_cross_section(ranked["RR_PROXY"])
        + score_from_cross_section(-ranked["VOL20"].fillna(ranked["VOL20"].median()))
        + score_from_cross_section(-ranked["MAXDD_6M"].abs())
    ) / 3

    ranked["Regime Score"] = ranked["REGIME"].clip(0, 100)
    ranked["Overall Score"] = (
        0.20 * ranked["Regime Score"]
        + 0.25 * ranked["Trend Score"]
        + 0.20 * ranked["Relative Strength Score"]
        + 0.15 * ranked["Pullback Score"]
        + 0.10 * ranked["Participation Score"]
        + 0.10 * ranked["Risk/Reward Score"]
    )

    ranked["Pred Win %"] = (
        0.30
        + 0.0024 * ranked["Trend Score"]
        + 0.0020 * ranked["Relative Strength Score"]
        + 0.0012 * ranked["Regime Score"]
        + 0.0010 * ranked["Pullback Score"]
        - 0.06 * ranked["VOL20"].fillna(0.3)
    ).clip(0.35, 0.78)

    ranked["Entry Status"] = np.select(
        [
            (ranked["Pullback Score"] >= 70) & (ranked["DIST_HIGH20"] > -0.06),
            (ranked["Trend Score"] >= 70) & (ranked["Pullback Score"] >= 55),
            (ranked["DIST_HIGH20"] > -0.015),
        ],
        ["Ready Now", "Watch Trigger", "Too Extended"],
        default="Avoid",
    )

    ranked = ranked.sort_values(["Overall Score", "Pred Win %"], ascending=False).reset_index(drop=True)
    return ranked


def thesis_state(row: pd.Series) -> Tuple[str, float, List[str], List[str], List[str]]:
    health = 50.0
    valid, warn, invalid = [], [], []

    if row["Close"] > row["SMA50"]:
        valid.append("Price remains above the 50-day trend line.")
        health += 10
    else:
        warn.append("Price is below the 50-day trend line.")
        health -= 12

    if row["Close"] > row["SMA200"]:
        valid.append("Long-term structure is still above the 200-day trend line.")
        health += 10
    else:
        invalid.append("A decisive loss of the 200-day trend line damages the larger trend thesis.")
        health -= 18

    if row["RS_SPY"] > 0 and row["RS_QQQ"] > 0:
        valid.append("The stock is outperforming both SPY and QQQ over the intermediate window.")
        health += 12
    else:
        warn.append("Relative strength versus SPY/QQQ is no longer clean.")
        health -= 10

    if pd.notna(row["SUPPORT_ATR"]) and row["SUPPORT_ATR"] >= 0:
        valid.append("Price is still above modeled support on a volatility-adjusted basis.")
        health += 8
    else:
        invalid.append("Price has fallen below modeled support on an ATR-adjusted basis.")
        health -= 18

    if row["Regime Score"] >= 65:
        valid.append("Broad market regime remains constructive for bullish swing trades.")
        health += 10
    elif row["Regime Score"] < 50:
        invalid.append("Broad market regime is hostile enough to negate many bullish setups.")
        health -= 18
    else:
        warn.append("Market regime is neutral rather than strongly supportive.")
        health -= 2

    if row["DIST_HIGH20"] > -0.01:
        warn.append("The stock is getting somewhat extended versus its recent 20-day high.")
        health -= 5

    health = clamp(health)
    if health >= 72:
        state = "Active"
    elif health >= 55:
        state = "At Risk"
    else:
        state = "Invalid"

    if not invalid:
        invalid.append("Exit on a decisive close below support or clear breakdown in relative strength.")
    return state, health, valid, warn, invalid


# =========================================================
# BACKTEST ENGINE
# =========================================================
def snapshot_score_at_date(ticker: str, df: pd.DataFrame, bench: Dict[str, pd.DataFrame], asof: pd.Timestamp, regime_score: float) -> Optional[Dict[str, float]]:
    hist = df.loc[:asof].copy()
    if len(hist) < MIN_HISTORY:
        return None
    try:
        row = feature_row(ticker, hist, bench, regime_score)
        return row
    except Exception:
        return None


def build_backtest_snapshots(prices: Dict[str, pd.DataFrame], bench: Dict[str, pd.DataFrame], rebalance_every: int = 20) -> pd.DataFrame:
    common_end = min(df.index.max() for df in prices.values())
    common_start = max(df.index.min() for df in prices.values())
    dates = pd.bdate_range(common_start, common_end)
    dates = [d for i, d in enumerate(dates) if i > MIN_HISTORY and i % rebalance_every == 0]

    records = []
    for dt in dates:
        try:
            regime = market_regime_score({k: v.loc[:dt] for k, v in bench.items() if len(v.loc[:dt]) >= MIN_HISTORY})[0]
        except Exception:
            continue
        rows = []
        for ticker, df in prices.items():
            row = snapshot_score_at_date(ticker, df, bench, dt, regime)
            if row is not None:
                rows.append(row)
        if not rows:
            continue
        ranked = rank_features(pd.DataFrame(rows)).head(DEFAULT_TOP_N).copy()
        ranked["Snapshot Date"] = dt
        records.append(ranked)

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def evaluate_forward_returns(snapshots: pd.DataFrame, prices: Dict[str, pd.DataFrame], hold_days: int = DEFAULT_HOLD_DAYS) -> pd.DataFrame:
    if snapshots.empty:
        return snapshots
    out = snapshots.copy()
    fwd_returns = []
    max_up = []
    max_down = []
    for _, row in out.iterrows():
        ticker = row["Ticker"]
        dt = pd.Timestamp(row["Snapshot Date"])
        close = get_close_series(prices[ticker])
        future = close.loc[close.index > dt].head(hold_days)
        if future.empty:
            fwd_returns.append(np.nan)
            max_up.append(np.nan)
            max_down.append(np.nan)
            continue
        entry = close.loc[:dt].iloc[-1]
        fwd_returns.append(float(future.iloc[-1] / entry - 1))
        max_up.append(float(future.max() / entry - 1))
        max_down.append(float(future.min() / entry - 1))
    out[f"Fwd {hold_days}D Return"] = fwd_returns
    out[f"Max Up {hold_days}D"] = max_up
    out[f"Max Down {hold_days}D"] = max_down
    out["Hit"] = out[f"Fwd {hold_days}D Return"] > 0
    return out.dropna(subset=[f"Fwd {hold_days}D Return"])


# =========================================================
# VISUALS
# =========================================================
def make_price_chart(df: pd.DataFrame, ticker: str, support: Optional[float] = None) -> go.Figure:
    chart = df.tail(220).copy()
    close = get_close_series(chart)
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=chart.index,
        open=chart["Open"],
        high=chart["High"],
        low=chart["Low"],
        close=chart["Close"],
        name=ticker,
    ))
    fig.add_trace(go.Scatter(x=chart.index, y=sma20, mode="lines", name="20D MA"))
    fig.add_trace(go.Scatter(x=chart.index, y=sma50, mode="lines", name="50D MA"))
    fig.add_trace(go.Scatter(x=chart.index, y=sma200, mode="lines", name="200D MA"))
    if support is not None and pd.notna(support):
        fig.add_hline(y=float(support), line_dash="dash", annotation_text="Support")
    fig.update_layout(height=540, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def metric_card(label: str, value: str):
    st.metric(label, value)


# =========================================================
# APP
# =========================================================
st.title("Quant Stock Thesis Platform")
st.caption(
    "A more comprehensive, interactive stock research dashboard with scanner, ticker lookup, thesis lifecycle monitoring, signal history, and forward-return backtesting."
)

with st.sidebar:
    st.header("Controls")
    universe_name = st.selectbox("Universe", list(UNIVERSE_MAP.keys()), index=2)
    top_n = st.slider("Top ideas shown", 5, 20, DEFAULT_TOP_N)
    hold_days = st.selectbox("Backtest forward window", [5, 10, 20, 40, 60], index=2)
    rebalance_days = st.selectbox("Backtest rebalance interval", [5, 10, 20], index=2)
    custom_lookup = st.text_input("Lookup ticker", value="AAPL").upper().strip()
    run_backtest = st.checkbox("Run backtest / signal history", value=True)
    st.markdown("---")
    st.write("**Model style**")
    st.write("Bullish only. End-of-day data. Quant-style cross-sectional scoring. Thesis-health state machine.")

universe = UNIVERSE_MAP[universe_name]
bench = load_benchmark_panel(period_years=2)
prices = load_prices(sorted(set(universe + ([custom_lookup] if custom_lookup else []))), period_years=2)

if not prices:
    st.error("No price data loaded.")
    st.stop()

regime, regime_diag = market_regime_score(bench)
rows = []
for ticker in universe:
    if ticker not in prices:
        continue
    try:
        rows.append(feature_row(ticker, prices[ticker], bench, regime))
    except Exception:
        continue

if not rows:
    st.error("No valid feature rows built.")
    st.stop()

ranked = rank_features(pd.DataFrame(rows)).head(top_n).copy()
state_values = []
health_values = []
for _, row in ranked.iterrows():
    state, health, _, _, _ = thesis_state(row)
    state_values.append(state)
    health_values.append(health)
ranked["Thesis State"] = state_values
ranked["Thesis Health"] = health_values

# =========================================================
# OVERVIEW
# =========================================================
left, right = st.columns([1.4, 1])
with left:
    st.subheader("Top Ranked Bullish Setups")
    show = ranked[[
        "Ticker", "Overall Score", "Pred Win %", "Regime Score", "Trend Score",
        "Relative Strength Score", "Pullback Score", "Entry Status", "Thesis State", "Thesis Health", "Close"
    ]].copy()
    show["Pred Win %"] = show["Pred Win %"].map(lambda x: f"{x:.1%}")
    for c in ["Overall Score", "Regime Score", "Trend Score", "Relative Strength Score", "Pullback Score", "Thesis Health", "Close"]:
        show[c] = show[c].round(2)
    st.dataframe(show, use_container_width=True, hide_index=True)

with right:
    st.subheader("Market Regime")
    c1, c2 = st.columns(2)
    with c1:
        metric_card("Regime Score", f"{regime:.1f} / 100")
    with c2:
        metric_card("Breadth", f"{regime_diag['Breadth']:.0%}")
    st.write(
        f"SPY 20D return: **{regime_diag['SPY 20D Return']:.2%}**  \
QQQ 20D return: **{regime_diag['QQQ 20D Return']:.2%}**  \
Avg index realized vol: **{regime_diag['Avg Index Vol']:.2%}**"
    )
    st.info("This regime score acts as a gate. Weak regime should reduce confidence, shorten holds, and increase invalidations.")

# =========================================================
# DETAIL TABS
# =========================================================
detail_tab, scanner_tab, backtest_tab = st.tabs(["Ticker Lookup & Thesis", "Scanner Diagnostics", "Backtest & Signal History"])

with detail_tab:
    lookup = custom_lookup if custom_lookup in prices else ranked.iloc[0]["Ticker"]
    if lookup not in prices:
        st.warning("Ticker not available from current data pull.")
    else:
        base_rows = ranked[ranked["Ticker"] == lookup]
        if base_rows.empty:
            detail_features = feature_row(lookup, prices[lookup], bench, regime)
            detail_row = rank_features(pd.DataFrame([detail_features])).iloc[0]
        else:
            detail_row = base_rows.iloc[0]
        state, health, valid, warn, invalid = thesis_state(detail_row)
        meta = get_ticker_meta(lookup)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Overall Score", f"{detail_row['Overall Score']:.1f}")
        m2.metric("Pred. Win %", f"{detail_row['Pred Win %']:.1%}")
        m3.metric("Entry Status", detail_row["Entry Status"])
        m4.metric("Thesis State", state)
        m5.metric("Health", f"{health:.1f} / 100")

        st.subheader(f"{lookup} — {meta['shortName']}")
        st.write(f"**Sector:** {meta['sector']}  \
**Industry:** {meta['industry']}  \
**Price:** {detail_row['Close']:.2f}  \
**Modeled Support:** {detail_row['SUPPORT']:.2f}")

        st.plotly_chart(make_price_chart(prices[lookup], lookup, detail_row["SUPPORT"]), use_container_width=True)

        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.subheader("Quant Factor Breakdown")
            factor_df = pd.DataFrame({
                "Bucket": ["Regime", "Trend", "Relative Strength", "Pullback", "Participation", "Risk / Reward"],
                "Score": [
                    detail_row["Regime Score"],
                    detail_row["Trend Score"],
                    detail_row["Relative Strength Score"],
                    detail_row["Pullback Score"],
                    detail_row["Participation Score"],
                    detail_row["Risk/Reward Score"],
                ],
            })
            st.dataframe(factor_df.style.format({"Score": "{:.1f}"}), use_container_width=True, hide_index=True)

            st.subheader("What must happen for this thesis to stay right")
            must_happen = [
                f"Price should hold above modeled support near {detail_row['SUPPORT']:.2f}.",
                "Relative strength vs SPY and QQQ should remain positive.",
                "The broad market regime should stay constructive for longs.",
                "The stock should start working within the next several sessions or weeks, not stall indefinitely.",
                "Pullbacks should remain orderly rather than expand into deep, high-volume damage.",
            ]
            for item in must_happen:
                st.write(f"- {item}")

        with col2:
            st.subheader("Thesis Lifecycle")
            if state == "Active":
                st.success(f"State: {state}")
            elif state == "At Risk":
                st.warning(f"State: {state}")
            else:
                st.error(f"State: {state}")

            st.write("**Why it is valid**")
            for item in valid:
                st.write(f"- {item}")
            st.write("**Warnings**")
            for item in warn:
                st.write(f"- {item}")
            st.write("**Invalidation / exit triggers**")
            for item in invalid:
                st.write(f"- {item}")

        st.subheader("Raw Quant Metrics")
        raw_cols = ["1M", "3M", "6M", "RS_SPY", "RS_QQQ", "RS_SLOPE", "RSI14", "VOL20", "VOL60", "DIST_HIGH20", "UPSIDE", "DOWNSIDE", "RR_PROXY", "AVG_DOLLAR_VOL"]
        raw = detail_row[raw_cols].to_frame("Value")
        display = raw.copy()
        for idx in display.index:
            val = display.loc[idx, "Value"]
            if idx == "AVG_DOLLAR_VOL":
                display.loc[idx, "Value"] = f"${val:,.0f}"
            elif idx in ["RSI14", "RR_PROXY", "RS_SLOPE"]:
                display.loc[idx, "Value"] = f"{val:.2f}"
            else:
                display.loc[idx, "Value"] = f"{val:.2%}"
        st.dataframe(display, use_container_width=True)

with scanner_tab:
    st.subheader("Scanner Diagnostics")
    st.write("This view lets you inspect the whole ranked set and compare factor buckets across current names.")
    diag = ranked[[
        "Ticker", "Overall Score", "Trend Score", "Relative Strength Score", "Pullback Score",
        "Participation Score", "Risk/Reward Score", "Pred Win %", "Entry Status", "Close"
    ]].copy()
    diag["Pred Win %"] = diag["Pred Win %"].map(lambda x: f"{x:.1%}")
    st.dataframe(diag, use_container_width=True, hide_index=True)

    st.subheader("Current Score Distribution")
    chart_df = ranked[["Ticker", "Overall Score", "Trend Score", "Relative Strength Score", "Pullback Score"]].set_index("Ticker")
    st.bar_chart(chart_df)

with backtest_tab:
    st.subheader("Backtest & Signal History")
    if not run_backtest:
        st.info("Enable backtest in the sidebar to compute signal history and forward-return validation.")
    else:
        with st.spinner("Building historical snapshots and evaluating forward returns..."):
            bt_snapshots = build_backtest_snapshots({k: v for k, v in prices.items() if k in universe}, bench, rebalance_every=rebalance_days)
            bt_results = evaluate_forward_returns(bt_snapshots, {k: v for k, v in prices.items() if k in universe}, hold_days=hold_days)

        if bt_results.empty:
            st.warning("No backtest results available yet.")
        else:
            top_only = bt_results.copy()
            avg_return = top_only[f"Fwd {hold_days}D Return"].mean()
            hit_rate = top_only["Hit"].mean()
            median_return = top_only[f"Fwd {hold_days}D Return"].median()
            avg_up = top_only[f"Max Up {hold_days}D"].mean()
            avg_down = top_only[f"Max Down {hold_days}D"].mean()

            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("Hit Rate", f"{hit_rate:.1%}")
            b2.metric("Avg Return", f"{avg_return:.2%}")
            b3.metric("Median Return", f"{median_return:.2%}")
            b4.metric("Avg Max Up", f"{avg_up:.2%}")
            b5.metric("Avg Max Down", f"{avg_down:.2%}")

            st.write("**Interpretation**")
            st.write(
                "This is the historical signal history for the top-ranked names generated on each rebalance date. "
                "It gives you the basic quant validation loop: hit rate, return profile, and path risk after signal generation."
            )

            st.subheader("Signal History Table")
            history_show = top_only[[
                "Snapshot Date", "Ticker", "Overall Score", "Pred Win %", "Entry Status", f"Fwd {hold_days}D Return", f"Max Up {hold_days}D", f"Max Down {hold_days}D", "Hit"
            ]].copy()
            history_show["Pred Win %"] = history_show["Pred Win %"].map(lambda x: f"{x:.1%}")
            for c in [f"Fwd {hold_days}D Return", f"Max Up {hold_days}D", f"Max Down {hold_days}D"]:
                history_show[c] = history_show[c].map(lambda x: f"{x:.2%}")
            st.dataframe(history_show.sort_values("Snapshot Date", ascending=False), use_container_width=True, hide_index=True)

            st.subheader("Average Forward Return by Ticker")
            avg_by_ticker = (
                top_only.groupby("Ticker")[f"Fwd {hold_days}D Return"].mean().sort_values(ascending=False)
            )
            st.bar_chart(avg_by_ticker)

            st.subheader("Average Forward Return by Entry Status")
            avg_by_status = top_only.groupby("Entry Status")[f"Fwd {hold_days}D Return"].mean().sort_values(ascending=False)
            st.bar_chart(avg_by_status)

st.caption(
    "This version moves into step 2: it adds historical signal tracking, forward-return backtesting, interactive ticker lookup, and a more complete quant-style diagnostic workflow. "
    "The next serious upgrade would be persistent storage, larger universes, true cross-sectional constituent lists, and walk-forward validation with transaction rules."
)
