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
MIN_HISTORY = 220
DEFAULT_TOP_N = 10
DEFAULT_HOLD_DAYS = 20

BUY_QUALITY = 68
WATCH_QUALITY = 62

BUY_TIMING = 58
WATCH_TIMING = 45

BUY_REGIME = 55
WATCH_REGIME = 48

HEALTH_ACTIVE = 70
HEALTH_RISK = 55

UNIVERSE_MAP = {
    "Mega Cap": [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "AMD", "NFLX",
        "ADBE", "INTU", "AMAT", "QCOM", "CSCO", "TXN", "PEP", "COST", "ISRG", "LIN",
    ],
    "Large Cap Growth": [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "AMD", "NFLX",
        "ADBE", "INTU", "AMAT", "QCOM", "NOW", "CRM", "PANW", "ANET", "ORCL", "MU",
        "KLAC", "LRCX", "MELI", "CRWD", "PLTR", "TTD", "SNOW", "CDNS", "SHOP", "UBER",
    ],
    "Broad Liquid List": [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "AMD", "NFLX",
        "ADBE", "INTU", "AMAT", "QCOM", "NOW", "CRM", "PANW", "ANET", "ORCL", "MU",
        "KLAC", "LRCX", "MELI", "CRWD", "PLTR", "TTD", "SNOW", "CDNS", "SHOP", "UBER",
        "JPM", "GS", "V", "MA", "AXP", "BKNG", "LLY", "UNH", "ISRG", "CAT",
        "DE", "GE", "ETN", "PH", "HON", "XOM", "CVX", "COST", "WMT", "HD",
    ],
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


def normalize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = flatten_df(df).copy()
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[keep].dropna().copy()


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

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
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


def nearest_recent_pivot(close: pd.Series, window: int = 20) -> float:
    return float(close.rolling(window).max().iloc[-1])


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
def load_benchmarks(period_years: int = 2) -> Dict[str, pd.DataFrame]:
    bench = ["SPY", "QQQ", "XLK", "XLY", "XLC", "XLF", "XLV", "XLI", "XLE", "XLP", "XLB", "XLU", "XLRE"]
    return load_prices(bench, period_years=period_years)


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
# REGIME
# =========================================================
def market_regime_score(bench: Dict[str, pd.DataFrame]) -> Tuple[float, Dict[str, float]]:
    spy = get_close_series(bench["SPY"])
    qqq = get_close_series(bench["QQQ"])

    def one(close: pd.Series) -> Dict[str, float]:
        return {
            "above50": float(close.iloc[-1] > close.rolling(50).mean().iloc[-1]),
            "above200": float(close.iloc[-1] > close.rolling(200).mean().iloc[-1]),
            "ret20": float(close.iloc[-1] / close.iloc[-21] - 1),
            "vol20": float(close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)),
        }

    spyf = one(spy)
    qqqf = one(qqq)

    breadth_values = []
    for etf in ["XLK", "XLY", "XLC", "XLF", "XLV", "XLI", "XLE", "XLP", "XLB", "XLU", "XLRE"]:
        if etf in bench:
            s = get_close_series(bench[etf])
            breadth_values.append(float(s.iloc[-1] > s.rolling(50).mean().iloc[-1]))
    breadth = float(np.mean(breadth_values)) if breadth_values else 0.5

    avg_vol = np.mean([spyf["vol20"], qqqf["vol20"]])

    score = 0.0
    score += 15 * spyf["above50"]
    score += 15 * spyf["above200"]
    score += 15 * qqqf["above50"]
    score += 15 * qqqf["above200"]
    score += 15 * clamp((spyf["ret20"] + 0.05) / 0.12 * 100) / 100
    score += 15 * clamp((qqqf["ret20"] + 0.05) / 0.12 * 100) / 100
    score += 10 * breadth
    score += 10 * clamp((0.35 - avg_vol) / 0.25 * 100) / 100

    diagnostics = {
        "SPY 20D Return": spyf["ret20"],
        "QQQ 20D Return": qqqf["ret20"],
        "Breadth": breadth,
        "Avg Vol": avg_vol,
    }

    return clamp(score), diagnostics


# =========================================================
# FEATURE ENGINE
# =========================================================
def feature_row(ticker: str, df: pd.DataFrame, bench: Dict[str, pd.DataFrame], regime: float) -> Dict[str, float]:
    close = get_close_series(df)
    vol = df["Volume"]

    spy = get_close_series(bench["SPY"]).reindex(close.index).ffill()
    qqq = get_close_series(bench["QQQ"]).reindex(close.index).ffill()

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    atr14 = compute_atr(df, 14)
    rsi14 = compute_rsi(close, 14)

    ret = close.pct_change()
    ret1m = float(close.iloc[-1] / close.iloc[-22] - 1)
    ret3m = float(close.iloc[-1] / close.iloc[-63] - 1)
    ret6m = float(close.iloc[-1] / close.iloc[-126] - 1)

    rs_spy = float(ret3m - (spy.iloc[-1] / spy.iloc[-63] - 1))
    rs_qqq = float(ret3m - (qqq.iloc[-1] / qqq.iloc[-63] - 1))

    rel_line_spy = (close / close.iloc[0]) / (spy / spy.iloc[0])
    rs_slope_20 = float(rolling_slope(rel_line_spy, 20).iloc[-1]) if len(rel_line_spy) >= 20 else 0.0

    vol20 = float(ret.rolling(20).std().iloc[-1] * np.sqrt(252))
    vol60 = float(ret.rolling(60).std().iloc[-1] * np.sqrt(252))

    recent_high20 = nearest_recent_pivot(close, 20)
    support = float(np.nanmax([sma20.iloc[-1], sma50.iloc[-1]]))
    dist_high20 = float(close.iloc[-1] / recent_high20 - 1)
    support_atr = float((close.iloc[-1] - support) / atr14.iloc[-1]) if pd.notna(atr14.iloc[-1]) and atr14.iloc[-1] > 0 else np.nan

    downside = float((close.iloc[-1] - support) / close.iloc[-1]) if support > 0 else np.nan
    upside = float((recent_high20 - close.iloc[-1]) / close.iloc[-1]) if close.iloc[-1] > 0 else np.nan
    rr_proxy = float(upside / downside) if pd.notna(downside) and downside != 0 else 0.0

    avg_dollar_vol = float((close * vol).rolling(50).mean().iloc[-1])
    avg_vol20 = float(vol.rolling(20).mean().iloc[-1])

    breakout_volume_ratio = float(vol.iloc[-1] / avg_vol20) if avg_vol20 > 0 else 1.0

    up_vol = vol[ret > 0].rolling(20).mean().iloc[-1] if (ret > 0).any() else np.nan
    down_vol = vol[ret < 0].rolling(20).mean().iloc[-1] if (ret < 0).any() else np.nan
    participation = float(up_vol / down_vol) if pd.notna(up_vol) and pd.notna(down_vol) and down_vol != 0 else 1.0

    trend_slope20 = float(rolling_slope(sma20, 20).iloc[-1]) if len(sma20.dropna()) >= 20 else 0.0
    trend_slope50 = float(rolling_slope(sma50, 20).iloc[-1]) if len(sma50.dropna()) >= 20 else 0.0

    maxdd_6m = max_drawdown(close.iloc[-126:]) if len(close) >= 126 else np.nan

    return {
        "Ticker": ticker,
        "Close": float(close.iloc[-1]),
        "SMA20": float(sma20.iloc[-1]),
        "SMA50": float(sma50.iloc[-1]),
        "SMA200": float(sma200.iloc[-1]),
        "ATR14": float(atr14.iloc[-1]) if pd.notna(atr14.iloc[-1]) else np.nan,
        "RSI14": float(rsi14.iloc[-1]),
        "1M": ret1m,
        "3M": ret3m,
        "6M": ret6m,
        "RS_SPY": rs_spy,
        "RS_QQQ": rs_qqq,
        "RS_SLOPE20": rs_slope_20,
        "VOL20": vol20,
        "VOL60": vol60,
        "AVG_DOLLAR_VOL": avg_dollar_vol,
        "BREAKOUT_VOL_RATIO": breakout_volume_ratio,
        "PARTICIPATION": participation,
        "DIST_HIGH20": dist_high20,
        "SUPPORT": support,
        "SUPPORT_ATR": support_atr,
        "UPSIDE": upside,
        "DOWNSIDE": downside,
        "RR_PROXY": rr_proxy,
        "TREND_SLOPE20": trend_slope20,
        "TREND_SLOPE50": trend_slope50,
        "MAXDD_6M": maxdd_6m,
        "PIVOT20": recent_high20,
        "REGIME": regime,
    }


def build_scores(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df = df[df["AVG_DOLLAR_VOL"] > 15_000_000].copy()

    # -------- QUALITY ENGINE --------
    trend_score = (
        score_from_cross_section(df["Close"] / df["SMA50"] - 1)
        + score_from_cross_section(df["Close"] / df["SMA200"] - 1)
        + score_from_cross_section(df["TREND_SLOPE20"])
        + score_from_cross_section(df["TREND_SLOPE50"])
    ) / 4

    momentum_score = (
        score_from_cross_section(df["1M"])
        + score_from_cross_section(df["3M"])
        + score_from_cross_section(df["6M"])
    ) / 3

    rs_score = (
        score_from_cross_section(df["RS_SPY"])
        + score_from_cross_section(df["RS_QQQ"])
        + score_from_cross_section(df["RS_SLOPE20"])
    ) / 3

    path_quality_score = (
        score_from_cross_section(-df["VOL20"].fillna(df["VOL20"].median()))
        + score_from_cross_section(-df["MAXDD_6M"].abs())
        + score_from_cross_section(df["RR_PROXY"])
    ) / 3

    participation_score = (
        score_from_cross_section(np.log(df["AVG_DOLLAR_VOL"]))
        + score_from_cross_section(df["PARTICIPATION"])
        + score_from_cross_section(df["BREAKOUT_VOL_RATIO"])
    ) / 3

    regime_fit_score = df["REGIME"].clip(0, 100)

    df["Trend Score"] = trend_score
    df["Momentum Score"] = momentum_score
    df["Relative Strength Score"] = rs_score
    df["Path Quality Score"] = path_quality_score
    df["Participation Score"] = participation_score
    df["Regime Fit Score"] = regime_fit_score

    df["Quality Score"] = (
        0.25 * df["Trend Score"]
        + 0.15 * df["Momentum Score"]
        + 0.25 * df["Relative Strength Score"]
        + 0.15 * df["Path Quality Score"]
        + 0.10 * df["Participation Score"]
        + 0.10 * df["Regime Fit Score"]
    )

    # -------- TIMING ENGINE --------
    pullback_component = (
        score_from_cross_section(-abs(df["DIST_HIGH20"] + 0.035))
        + score_from_cross_section(-abs(df["SUPPORT_ATR"] - 1.0))
        + score_from_cross_section(-abs(df["RSI14"] - 58))
    ) / 3

    breakout_component = (
        score_from_cross_section(df["BREAKOUT_VOL_RATIO"])
        + score_from_cross_section(df["RS_SLOPE20"])
        + score_from_cross_section(df["Close"] / df["PIVOT20"] - 1)
    ) / 3

    extension_penalty = score_from_cross_section(-abs(df["DIST_HIGH20"]))
    df["Pullback Timing Score"] = pullback_component
    df["Breakout Timing Score"] = breakout_component
    df["Timing Score"] = (0.65 * pullback_component + 0.35 * breakout_component + 0.15 * extension_penalty).clip(0, 100)

    # -------- Predicted win % --------
    df["Pred Win %"] = (
        0.32
        + 0.0020 * df["Quality Score"]
        + 0.0016 * df["Timing Score"]
        - 0.05 * df["VOL20"].fillna(0.3)
    ).clip(0.35, 0.78)

    # -------- Entry state --------
    pullback_buy = (
        (df["Quality Score"] >= BUY_QUALITY)
        & (df["REGIME"] >= BUY_REGIME)
        & (df["RS_SPY"] > 0)
        & (df["RS_QQQ"] > 0)
        & (df["Close"] > df["SMA50"])
        & (df["Close"] > df["SMA200"])
        & (df["SUPPORT_ATR"].between(0.0, 2.0, inclusive="both"))
        & (df["DIST_HIGH20"].between(-0.07, -0.01, inclusive="both"))
    )

    breakout_buy = (
        (df["Quality Score"] >= BUY_QUALITY + 2)
        & (df["REGIME"] >= BUY_REGIME)
        & (df["RS_SPY"] > 0)
        & (df["RS_QQQ"] > 0)
        & (df["Close"] >= df["PIVOT20"])
        & (df["BREAKOUT_VOL_RATIO"] >= 1.15)
        & (df["Close"] > df["SMA50"])
        & (df["Close"] > df["SMA200"])
    )

    watch_trigger = (
        (df["Quality Score"] >= WATCH_QUALITY)
        & (df["Timing Score"] >= WATCH_TIMING)
        & (df["REGIME"] >= WATCH_REGIME)
    )

    too_extended = (
        (df["Quality Score"] >= BUY_QUALITY)
        & (df["DIST_HIGH20"] > -0.01)
        & (df["Close"] > df["SMA50"])
    )

    df["Setup Type"] = np.select(
        [pullback_buy, breakout_buy, too_extended],
        ["Pullback", "Breakout", "Extended Leader"],
        default="Mixed / Weak",
    )

    df["Entry State"] = np.select(
        [pullback_buy | breakout_buy, watch_trigger, too_extended],
        ["Buy Now", "Watch Trigger", "Too Extended"],
        default="No Trade",
    )

    return df.sort_values(["Quality Score", "Timing Score"], ascending=False).reset_index(drop=True)


# =========================================================
# MANAGEMENT ENGINE
# =========================================================
def thesis_health_and_action(row: pd.Series) -> Tuple[str, float, str, List[str], List[str], List[str]]:
    health = 50.0
    valid, warnings_list, invalid = [], [], []

    if row["Close"] > row["SMA50"]:
        valid.append("Price is above the 50-day trend line.")
        health += 10
    else:
        warnings_list.append("Price slipped below the 50-day trend line.")
        health -= 12

    if row["Close"] > row["SMA200"]:
        valid.append("Long-term structure is above the 200-day trend line.")
        health += 10
    else:
        invalid.append("Price below the 200-day trend line damages the larger bullish thesis.")
        health -= 18

    if row["RS_SPY"] > 0 and row["RS_QQQ"] > 0:
        valid.append("The stock is outperforming SPY and QQQ.")
        health += 12
    else:
        warnings_list.append("Relative strength versus the indices is no longer clean.")
        health -= 10

    if pd.notna(row["SUPPORT_ATR"]) and row["SUPPORT_ATR"] >= 0:
        valid.append("Price remains above modeled support on an ATR-adjusted basis.")
        health += 8
    else:
        invalid.append("Price is below modeled support on a volatility-adjusted basis.")
        health -= 18

    if row["REGIME"] >= BUY_REGIME:
        valid.append("Broad market regime remains supportive.")
        health += 10
    elif row["REGIME"] >= WATCH_REGIME:
        warnings_list.append("Market regime is neutral rather than strongly supportive.")
        health -= 3
    else:
        invalid.append("Broad market regime is too weak for a clean bullish thesis.")
        health -= 18

    if row["DIST_HIGH20"] > -0.005:
        warnings_list.append("The stock is getting extended versus its recent pivot.")
        health -= 5

    health = clamp(health)

    if health >= HEALTH_ACTIVE:
        state = "Active"
    elif health >= HEALTH_RISK:
        state = "At Risk"
    else:
        state = "Invalid"

    if state == "Invalid":
        action = "Sell"
    elif row["Entry State"] == "Buy Now" and health >= HEALTH_ACTIVE:
        action = "Buy"
    elif row["Entry State"] == "Watch Trigger" and health >= HEALTH_RISK:
        action = "Wait"
    elif row["Entry State"] == "Too Extended" and health >= HEALTH_RISK:
        action = "Wait"
    elif health >= HEALTH_ACTIVE:
        action = "Hold"
    elif health >= HEALTH_RISK:
        action = "Trim"
    else:
        action = "Sell"

    if not invalid:
        invalid.append("Sell on a decisive break of support, RS failure, or broad regime deterioration.")

    return state, health, action, valid, warnings_list, invalid


# =========================================================
# TRADE PLAN
# =========================================================
def trade_plan(row: pd.Series) -> Dict[str, float]:
    support = float(row["SUPPORT"])
    atr = float(row["ATR14"]) if pd.notna(row["ATR14"]) else 0.0
    close = float(row["Close"])
    pivot = float(row["PIVOT20"])

    if row["Setup Type"] == "Breakout":
        entry_low = pivot
        entry_high = pivot * 1.01
    else:
        entry_low = max(support, close - 0.5 * atr) if atr > 0 else support
        entry_high = close

    stop = max(support - 1.0 * atr, 0) if atr > 0 else support * 0.97
    target1 = close + 1.5 * atr if atr > 0 else close * 1.04
    target2 = close + 3.0 * atr if atr > 0 else close * 1.08
    trigger = pivot

    return {
        "Entry Low": float(entry_low),
        "Entry High": float(entry_high),
        "Stop": float(stop),
        "Target 1": float(target1),
        "Target 2": float(target2),
        "Trigger Price": float(trigger),
    }


# =========================================================
# BACKTEST
# =========================================================
def snapshot_features_at_date(
    ticker: str,
    df: pd.DataFrame,
    bench: Dict[str, pd.DataFrame],
    dt: pd.Timestamp,
    regime_score: float,
) -> Optional[Dict[str, float]]:
    hist = df.loc[:dt].copy()
    if len(hist) < MIN_HISTORY:
        return None
    try:
        return feature_row(ticker, hist, bench, regime_score)
    except Exception:
        return None


def build_backtest_snapshots(
    prices: Dict[str, pd.DataFrame],
    bench: Dict[str, pd.DataFrame],
    rebalance_every: int = 20,
) -> pd.DataFrame:
    common_end = min(df.index.max() for df in prices.values())
    common_start = max(df.index.min() for df in prices.values())
    dates = pd.bdate_range(common_start, common_end)
    dates = [d for i, d in enumerate(dates) if i > MIN_HISTORY and i % rebalance_every == 0]

    records = []
    for dt in dates:
        try:
            bench_slice = {k: v.loc[:dt] for k, v in bench.items() if len(v.loc[:dt]) >= MIN_HISTORY}
            if "SPY" not in bench_slice or "QQQ" not in bench_slice:
                continue
            regime_score = market_regime_score(bench_slice)[0]
        except Exception:
            continue

        rows = []
        for ticker, df in prices.items():
            row = snapshot_features_at_date(ticker, df, bench_slice, dt, regime_score)
            if row is not None:
                rows.append(row)

        if not rows:
            continue

        ranked = build_scores(pd.DataFrame(rows)).copy()

        states = []
        healths = []
        actions = []
        for _, r in ranked.iterrows():
            state, health, action, _, _, _ = thesis_health_and_action(r)
            states.append(state)
            healths.append(health)
            actions.append(action)

        ranked["Thesis State"] = states
        ranked["Thesis Health"] = healths
        ranked["Decision"] = actions
        ranked["Snapshot Date"] = dt
        records.append(ranked)

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def evaluate_forward_returns(
    snapshots: pd.DataFrame,
    prices: Dict[str, pd.DataFrame],
    hold_days: int = DEFAULT_HOLD_DAYS,
) -> pd.DataFrame:
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
    fig.add_trace(
        go.Candlestick(
            x=chart.index,
            open=chart["Open"],
            high=chart["High"],
            low=chart["Low"],
            close=chart["Close"],
            name=ticker,
        )
    )
    fig.add_trace(go.Scatter(x=chart.index, y=sma20, mode="lines", name="20D MA"))
    fig.add_trace(go.Scatter(x=chart.index, y=sma50, mode="lines", name="50D MA"))
    fig.add_trace(go.Scatter(x=chart.index, y=sma200, mode="lines", name="200D MA"))
    if support is not None and pd.notna(support):
        fig.add_hline(y=float(support), line_dash="dash", annotation_text="Support")

    fig.update_layout(height=540, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# =========================================================
# APP
# =========================================================
st.title("Quant Stock Thesis Platform")
st.caption(
    "Bullish-only stock scanner with Quality, Timing, and Management engines. "
    "It gives you buy / wait / hold / trim / sell decisions plus entry, stop, and target levels."
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
    st.write("**Decision thresholds**")
    st.write(f"- Buy if Regime > {BUY_REGIME}, Quality > {BUY_QUALITY}, and Timing > {BUY_TIMING}")
    st.write(f"- Watch if Regime > {WATCH_REGIME}, Quality > {WATCH_QUALITY}, and Timing > {WATCH_TIMING}")
    st.write("- Sell when thesis turns invalid or support / RS breaks")

universe = UNIVERSE_MAP[universe_name]
tickers_to_load = sorted(set(universe + ([custom_lookup] if custom_lookup else [])))

bench = load_benchmarks(period_years=2)
prices = load_prices(tickers_to_load, period_years=2)

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
    st.error("No valid feature rows could be built.")
    st.stop()

ranked = build_scores(pd.DataFrame(rows))

states = []
healths = []
actions = []
setup_valids = []
setup_warns = []
setup_invalids = []
plans = []

for _, row in ranked.iterrows():
    state, health, action, valid, warn, invalid = thesis_health_and_action(row)
    plan = trade_plan(row)
    states.append(state)
    healths.append(health)
    actions.append(action)
    setup_valids.append(valid)
    setup_warns.append(warn)
    setup_invalids.append(invalid)
    plans.append(plan)

ranked["Thesis State"] = states
ranked["Thesis Health"] = healths
ranked["Decision"] = actions
ranked["Why Valid"] = setup_valids
ranked["Warnings"] = setup_warns
ranked["Invalidation"] = setup_invalids
ranked["Entry Low"] = [p["Entry Low"] for p in plans]
ranked["Entry High"] = [p["Entry High"] for p in plans]
ranked["Stop"] = [p["Stop"] for p in plans]
ranked["Target 1"] = [p["Target 1"] for p in plans]
ranked["Target 2"] = [p["Target 2"] for p in plans]
ranked["Trigger Price"] = [p["Trigger Price"] for p in plans]

ranked = ranked.head(top_n).copy()

# =========================================================
# OVERVIEW
# =========================================================
left, right = st.columns([1.4, 1])

with left:
    st.subheader("Top Ranked Setups")
    show = ranked[
        [
            "Ticker", "Quality Score", "Timing Score", "Pred Win %", "Setup Type", "Entry State",
            "Thesis State", "Thesis Health", "Decision", "Trigger Price", "Stop", "Target 1", "Close"
        ]
    ].copy()

    show["Pred Win %"] = show["Pred Win %"].map(lambda x: f"{x:.1%}")
    for col in ["Quality Score", "Timing Score", "Thesis Health", "Trigger Price", "Stop", "Target 1", "Close"]:
        show[col] = show[col].astype(float).round(2)

    st.dataframe(show, use_container_width=True, hide_index=True)

with right:
    st.subheader("Market Regime")
    c1, c2 = st.columns(2)
    c1.metric("Regime Score", f"{regime:.1f} / 100")
    c2.metric("Breadth", f"{regime_diag['Breadth']:.0%}")

    st.write(
        f"SPY 20D return: **{regime_diag['SPY 20D Return']:.2%}**  \n"
        f"QQQ 20D return: **{regime_diag['QQQ 20D Return']:.2%}**  \n"
        f"Avg index vol: **{regime_diag['Avg Vol']:.2%}**"
    )

    st.info("Weak regime should produce fewer Buy signals and more Wait / Sell decisions.")

# =========================================================
# TABS
# =========================================================
detail_tab, scanner_tab, backtest_tab = st.tabs(
    ["Ticker Lookup & Decision", "Scanner Diagnostics", "Backtest & Signal History"]
)

with detail_tab:
    lookup = custom_lookup if custom_lookup in prices else ranked.iloc[0]["Ticker"]

    if lookup in ranked["Ticker"].values:
        detail_row = ranked[ranked["Ticker"] == lookup].iloc[0]
    else:
        single_raw = feature_row(lookup, prices[lookup], bench, regime)
        detail_row = build_scores(pd.DataFrame([single_raw])).iloc[0]
        state, health, action, valid, warn, invalid = thesis_health_and_action(detail_row)
        plan = trade_plan(detail_row)
        detail_row["Thesis State"] = state
        detail_row["Thesis Health"] = health
        detail_row["Decision"] = action
        detail_row["Why Valid"] = valid
        detail_row["Warnings"] = warn
        detail_row["Invalidation"] = invalid
        detail_row["Entry Low"] = plan["Entry Low"]
        detail_row["Entry High"] = plan["Entry High"]
        detail_row["Stop"] = plan["Stop"]
        detail_row["Target 1"] = plan["Target 1"]
        detail_row["Target 2"] = plan["Target 2"]
        detail_row["Trigger Price"] = plan["Trigger Price"]

    meta = get_ticker_meta(lookup)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Quality", f"{detail_row['Quality Score']:.1f}")
    m2.metric("Timing", f"{detail_row['Timing Score']:.1f}")
    m3.metric("Pred. Win %", f"{detail_row['Pred Win %']:.1%}")
    m4.metric("Entry State", detail_row["Entry State"])
    m5.metric("Thesis Health", f"{detail_row['Thesis Health']:.1f} / 100")
    m6.metric("Decision", detail_row["Decision"])

    st.subheader(f"{lookup} — {meta['shortName']}")
    st.write(
        f"**Sector:** {meta['sector']}  \n"
        f"**Industry:** {meta['industry']}  \n"
        f"**Price:** {detail_row['Close']:.2f}  \n"
        f"**Setup Type:** {detail_row['Setup Type']}"
    )

    st.plotly_chart(make_price_chart(prices[lookup], lookup, detail_row["SUPPORT"]), use_container_width=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Trade Plan")
        st.write(f"**Decision:** {detail_row['Decision']}")
        st.write(f"**Entry Zone:** {detail_row['Entry Low']:.2f} to {detail_row['Entry High']:.2f}")
        st.write(f"**Trigger Price:** {detail_row['Trigger Price']:.2f}")
        st.write(f"**Stop:** {detail_row['Stop']:.2f}")
        st.write(f"**Target 1:** {detail_row['Target 1']:.2f}")
        st.write(f"**Target 2:** {detail_row['Target 2']:.2f}")

        st.subheader("What must happen next")
        must_happen = [
            f"Price should stay above modeled support near {detail_row['SUPPORT']:.2f}.",
            "Relative strength versus SPY and QQQ should remain positive.",
            "The stock should begin making progress within the next several sessions to weeks.",
            "Pullbacks should remain controlled rather than expand into deep technical damage.",
        ]
        for item in must_happen:
            st.write(f"- {item}")

        st.subheader("Quant Factor Breakdown")
        factor_df = pd.DataFrame(
            {
                "Bucket": [
                    "Trend", "Momentum", "Relative Strength",
                    "Path Quality", "Participation", "Regime Fit",
                    "Quality", "Timing"
                ],
                "Score": [
                    detail_row["Trend Score"],
                    detail_row["Momentum Score"],
                    detail_row["Relative Strength Score"],
                    detail_row["Path Quality Score"],
                    detail_row["Participation Score"],
                    detail_row["Regime Fit Score"],
                    detail_row["Quality Score"],
                    detail_row["Timing Score"],
                ],
            }
        )
        st.dataframe(factor_df.style.format({"Score": "{:.1f}"}), use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Thesis Lifecycle")

        if detail_row["Thesis State"] == "Active":
            st.success(f"State: {detail_row['Thesis State']}")
        elif detail_row["Thesis State"] == "At Risk":
            st.warning(f"State: {detail_row['Thesis State']}")
        else:
            st.error(f"State: {detail_row['Thesis State']}")

        st.write("**Why it is valid**")
        for item in detail_row["Why Valid"]:
            st.write(f"- {item}")

        st.write("**Warnings**")
        for item in detail_row["Warnings"]:
            st.write(f"- {item}")

        st.write("**Invalidation / sell triggers**")
        for item in detail_row["Invalidation"]:
            st.write(f"- {item}")

with scanner_tab:
    st.subheader("Scanner Diagnostics")
    diag = ranked[
        [
            "Ticker", "Quality Score", "Timing Score", "Trend Score", "Momentum Score",
            "Relative Strength Score", "Path Quality Score", "Participation Score",
            "Regime Fit Score", "Pred Win %", "Setup Type", "Entry State", "Decision"
        ]
    ].copy()

    diag["Pred Win %"] = diag["Pred Win %"].map(lambda x: f"{x:.1%}")
    st.dataframe(diag, use_container_width=True, hide_index=True)

    st.subheader("Score Distribution")
    chart_df = ranked[
        ["Ticker", "Quality Score", "Timing Score", "Trend Score", "Relative Strength Score"]
    ].set_index("Ticker")
    st.bar_chart(chart_df)

with backtest_tab:
    st.subheader("Backtest & Signal History")

    if not run_backtest:
        st.info("Enable backtest in the sidebar to compute historical signal results.")
    else:
        with st.spinner("Building signal history and forward-return validation..."):
            bt_universe_prices = {k: v for k, v in prices.items() if k in universe}
            snapshots = build_backtest_snapshots(bt_universe_prices, bench, rebalance_every=rebalance_days)
            results = evaluate_forward_returns(snapshots, bt_universe_prices, hold_days=hold_days)

        if results.empty:
            st.warning("No backtest results available.")
        else:
            results = results[results["Decision"].isin(["Buy", "Hold", "Wait"])].copy()

            hit_rate = results["Hit"].mean()
            avg_return = results[f"Fwd {hold_days}D Return"].mean()
            median_return = results[f"Fwd {hold_days}D Return"].median()
            avg_up = results[f"Max Up {hold_days}D"].mean()
            avg_down = results[f"Max Down {hold_days}D"].mean()

            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("Hit Rate", f"{hit_rate:.1%}")
            b2.metric("Avg Return", f"{avg_return:.2%}")
            b3.metric("Median Return", f"{median_return:.2%}")
            b4.metric("Avg Max Up", f"{avg_up:.2%}")
            b5.metric("Avg Max Down", f"{avg_down:.2%}")

            st.write("**Signal History Table**")
            history_show = results[
                [
                    "Snapshot Date", "Ticker", "Quality Score", "Timing Score", "Entry State",
                    "Decision", f"Fwd {hold_days}D Return", f"Max Up {hold_days}D",
                    f"Max Down {hold_days}D", "Hit"
                ]
            ].copy()

            for col in [f"Fwd {hold_days}D Return", f"Max Up {hold_days}D", f"Max Down {hold_days}D"]:
                history_show[col] = history_show[col].map(lambda x: f"{x:.2%}")

            st.dataframe(history_show.sort_values("Snapshot Date", ascending=False), use_container_width=True, hide_index=True)

            st.subheader("Average Forward Return by Entry State")
            by_state = results.groupby("Entry State")[f"Fwd {hold_days}D Return"].mean().sort_values(ascending=False)
            st.bar_chart(by_state)

            st.subheader("Average Forward Return by Ticker")
            by_ticker = results.groupby("Ticker")[f"Fwd {hold_days}D Return"].mean().sort_values(ascending=False)
            st.bar_chart(by_ticker)

st.caption(
    "This version is a full-file rewrite. It adds explicit Buy / Wait / Hold / Trim / Sell decisions, "
    "a clearer Entry State engine, trade levels, and a stronger split between stock quality, timing, and management."
)