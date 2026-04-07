import warnings
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Quant Thesis Engine", layout="wide")

# =========================================================
# CONFIG
# =========================================================
MIN_HISTORY = 220
DEFAULT_TOP_N = 10
DEFAULT_HOLD_DAYS = 20
QUALITY_THRESHOLD = 70
TIMING_THRESHOLD = 65
HEALTH_ACTIVE = 72
HEALTH_RISK = 55
BUY_REGIME_THRESHOLD = 55
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


def normalize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = flatten_df(df).copy()
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[cols].dropna().copy()


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


def downside_semivol(returns: pd.Series, window: int = 60) -> float:
    neg = returns[returns < 0]
    if len(neg) < min(10, window // 3):
        return np.nan
    return float(neg.tail(window).std() * np.sqrt(252))


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
    bench = ["SPY", "QQQ", "XLK", "XLC", "XLY", "XLP", "XLF", "XLV", "XLI", "XLE", "XLU", "XLB", "XLRE"]
    return load_prices(bench, period_years)


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

    return clamp(score), {
        "SPY 20D Return": spyf["ret20"],
        "QQQ 20D Return": qqqf["ret20"],
        "Breadth": breadth,
        "Avg Index Vol": avg_vol,
    }


# =========================================================
# FEATURE ENGINE
# =========================================================
def feature_row(ticker: str, df: pd.DataFrame, bench: Dict[str, pd.DataFrame], regime: float) -> Dict[str, float]:
    close = get_close_series(df)
    vol = df["Volume"]
    ret = close.pct_change()

    spy = get_close_series(bench["SPY"]).reindex(close.index).ffill()
    qqq = get_close_series(bench["QQQ"]).reindex(close.index).ffill()

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    atr14 = compute_atr(df)
    rsi14 = compute_rsi(close)

    ret1m = float(close.iloc[-1] / close.iloc[-22] - 1)
    ret3m = float(close.iloc[-1] / close.iloc[-63] - 1)
    ret6m = float(close.iloc[-1] / close.iloc[-126] - 1)
    accel = float(ret1m - ret3m / 3)

    rs_spy = float(ret3m - (spy.iloc[-1] / spy.iloc[-63] - 1))
    rs_qqq = float(ret3m - (qqq.iloc[-1] / qqq.iloc[-63] - 1))

    rel_line_spy = (close / close.iloc[0]) / (spy / spy.iloc[0])
    rs_slope20 = rolling_slope(rel_line_spy, 20).iloc[-1]
    rs_slope60 = rolling_slope(rel_line_spy, 60).iloc[-1] if len(rel_line_spy) >= 60 else np.nan

    avg_dollar_vol = float((close * vol).rolling(50).mean().iloc[-1])
    vol20 = float(ret.rolling(20).std().iloc[-1] * np.sqrt(252))
    vol60 = float(ret.rolling(60).std().iloc[-1] * np.sqrt(252))
    semi60 = downside_semivol(ret, 60)
    dd6m = max_drawdown(close.tail(126)) if len(close) >= 126 else np.nan
    smoothness = float(ret3m / vol20) if pd.notna(vol20) and vol20 != 0 else np.nan

    up_vol = vol[ret > 0].rolling(20).mean().iloc[-1] if (ret > 0).any() else np.nan
    down_vol = vol[ret < 0].rolling(20).mean().iloc[-1] if (ret < 0).any() else np.nan
    up_down_ratio = float(up_vol / down_vol) if pd.notna(up_vol) and pd.notna(down_vol) and down_vol != 0 else 1.0
    breakout_volume_ratio = float(vol.iloc[-1] / vol.rolling(20).mean().iloc[-1]) if pd.notna(vol.rolling(20).mean().iloc[-1]) else 1.0

    recent_high20 = float(close.rolling(20).max().iloc[-1])
    recent_low20 = float(close.rolling(20).min().iloc[-1])
    dist_high20 = float(close.iloc[-1] / recent_high20 - 1)

    support = float(np.nanmax([sma20.iloc[-1], sma50.iloc[-1]]))
    support_atr = float((close.iloc[-1] - support) / atr14.iloc[-1]) if pd.notna(atr14.iloc[-1]) and atr14.iloc[-1] > 0 else np.nan
    downside = float((close.iloc[-1] - support) / close.iloc[-1]) if close.iloc[-1] > 0 else np.nan
    upside = float((recent_high20 - close.iloc[-1]) / close.iloc[-1]) if close.iloc[-1] > 0 else np.nan
    rr_proxy = float(upside / downside) if pd.notna(downside) and downside not in [0, np.nan] else np.nan

    slope20 = rolling_slope(sma20, 20).iloc[-1]
    slope50 = rolling_slope(sma50, 20).iloc[-1]
    ma_alignment = float((sma20.iloc[-1] > sma50.iloc[-1]) and (sma50.iloc[-1] > sma200.iloc[-1]))
    price_above_50 = float(close.iloc[-1] > sma50.iloc[-1])
    price_above_200 = float(close.iloc[-1] > sma200.iloc[-1])

    return {
        "Ticker": ticker,
        "Close": float(close.iloc[-1]),
        "SMA20": float(sma20.iloc[-1]),
        "SMA50": float(sma50.iloc[-1]),
        "SMA200": float(sma200.iloc[-1]),
        "ATR14": float(atr14.iloc[-1]) if pd.notna(atr14.iloc[-1]) else np.nan,
        "RSI14": float(rsi14.iloc[-1]),
        "RET_1M": ret1m,
        "RET_3M": ret3m,
        "RET_6M": ret6m,
        "ACCEL": accel,
        "RS_SPY": rs_spy,
        "RS_QQQ": rs_qqq,
        "RS_SLOPE20": float(rs_slope20) if pd.notna(rs_slope20) else 0.0,
        "RS_SLOPE60": float(rs_slope60) if pd.notna(rs_slope60) else 0.0,
        "AVG_DOLLAR_VOL": avg_dollar_vol,
        "VOL20": vol20,
        "VOL60": vol60,
        "SEMI60": semi60,
        "DD6M": dd6m,
        "SMOOTHNESS": smoothness,
        "UP_DOWN_VOL_RATIO": up_down_ratio,
        "BREAKOUT_VOL_RATIO": breakout_volume_ratio,
        "DIST_HIGH20": dist_high20,
        "RECENT_LOW20": recent_low20,
        "SUPPORT": support,
        "SUPPORT_ATR": support_atr,
        "UPSIDE": upside,
        "DOWNSIDE": downside,
        "RR_PROXY": rr_proxy,
        "SLOPE20": float(slope20) if pd.notna(slope20) else 0.0,
        "SLOPE50": float(slope50) if pd.notna(slope50) else 0.0,
        "MA_ALIGNMENT": ma_alignment,
        "ABOVE50": price_above_50,
        "ABOVE200": price_above_200,
        "REGIME": regime,
    }


def score_engines(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    df = df[df["AVG_DOLLAR_VOL"] > 15_000_000].copy()

    # Quality Engine
    trend_score = (
        score_from_cross_section(df["Close"] / df["SMA50"] - 1)
        + score_from_cross_section(df["Close"] / df["SMA200"] - 1)
        + score_from_cross_section(df["SLOPE20"])
        + score_from_cross_section(df["SLOPE50"])
        + score_from_cross_section(df["MA_ALIGNMENT"])
    ) / 5

    momentum_score = (
        score_from_cross_section(df["RET_1M"])
        + score_from_cross_section(df["RET_3M"])
        + score_from_cross_section(df["RET_6M"])
        + score_from_cross_section(df["ACCEL"])
    ) / 4

    rs_score = (
        score_from_cross_section(df["RS_SPY"])
        + score_from_cross_section(df["RS_QQQ"])
        + score_from_cross_section(df["RS_SLOPE20"])
        + score_from_cross_section(df["RS_SLOPE60"])
    ) / 4

    path_score = (
        score_from_cross_section(-df["VOL20"].fillna(df["VOL20"].median()))
        + score_from_cross_section(-df["SEMI60"].fillna(df["SEMI60"].median()))
        + score_from_cross_section(-df["DD6M"].abs())
        + score_from_cross_section(df["SMOOTHNESS"].fillna(0))
    ) / 4

    participation_score = (
        score_from_cross_section(np.log(df["AVG_DOLLAR_VOL"]))
        + score_from_cross_section(df["UP_DOWN_VOL_RATIO"].fillna(1.0))
        + score_from_cross_section(df["BREAKOUT_VOL_RATIO"].fillna(1.0))
    ) / 3

    regime_fit_score = df["REGIME"].clip(0, 100)

    df["Trend Score"] = trend_score
    df["Momentum Score"] = momentum_score
    df["Relative Strength Score"] = rs_score
    df["Path Quality Score"] = path_score
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

    # Timing Engine
    pullback_geometry = (
        score_from_cross_section(-abs(df["DIST_HIGH20"] + 0.035))
        + score_from_cross_section(-abs(df["SUPPORT_ATR"] - 1.0))
        + score_from_cross_section(-abs(df["RSI14"] - 58))
    ) / 3

    trigger_quality = (
        score_from_cross_section(df["BREAKOUT_VOL_RATIO"].fillna(1.0))
        + score_from_cross_section(df["RS_SLOPE20"])
        + score_from_cross_section(df["UPSIDE"].fillna(0))
    ) / 3

    extension_penalty = score_from_cross_section(-abs(df["DIST_HIGH20"]))
    rr_score = (
        score_from_cross_section(df["RR_PROXY"].fillna(0))
        + score_from_cross_section(-df["DOWNSIDE"].abs().fillna(0))
    ) / 2

    df["Pullback Geometry Score"] = pullback_geometry
    df["Trigger Quality Score"] = trigger_quality
    df["Entry RR Score"] = rr_score
    df["Timing Score"] = (
        0.35 * df["Pullback Geometry Score"]
        + 0.25 * df["Trigger Quality Score"]
        + 0.20 * extension_penalty
        + 0.20 * df["Entry RR Score"]
    )

    df["Entry State"] = np.select(
        [
            (df["REGIME"] > BUY_REGIME_THRESHOLD) & (df["Quality Score"] > QUALITY_THRESHOLD) & (df["Timing Score"] > TIMING_THRESHOLD),
            (df["Quality Score"] > QUALITY_THRESHOLD) & (df["Timing Score"].between(60, TIMING_THRESHOLD, inclusive="left")),
            (df["Quality Score"] > QUALITY_THRESHOLD) & (df["Timing Score"] < 60),
        ],
        ["Buy Now", "Wait for Trigger", "Too Extended"],
        default="No Trade",
    )

    # Management Engine baseline
    df["Pred Win %"] = (
        0.28
        + 0.0024 * df["Quality Score"]
        + 0.0017 * df["Timing Score"]
        + 0.0012 * df["REGIME"]
        - 0.05 * df["VOL20"].fillna(0.3)
    ).clip(0.35, 0.82)

    return df.sort_values(["Quality Score", "Timing Score", "Pred Win %"], ascending=False).reset_index(drop=True)


def management_engine(row: pd.Series, days_in_trade: int = 0, progress: Optional[float] = None) -> Tuple[str, float, str, List[str], List[str], List[str]]:
    health = 50.0
    valid, warnings_list, invalid = [], [], []

    if row["Close"] > row["SMA50"]:
        valid.append("Price remains above the 50-day trend line.")
        health += 10
    else:
        warnings_list.append("Price is below the 50-day trend line.")
        health -= 12

    if row["Close"] > row["SMA200"]:
        valid.append("Long-term structure is still above the 200-day trend line.")
        health += 10
    else:
        invalid.append("A decisive loss of the 200-day trend line damages the larger trend thesis.")
        health -= 18

    if row["RS_SPY"] > 0 and row["RS_QQQ"] > 0:
        valid.append("The stock is still outperforming SPY and QQQ.")
        health += 12
    else:
        warnings_list.append("Relative strength versus SPY/QQQ is no longer clean.")
        health -= 10

    if pd.notna(row["SUPPORT_ATR"]) and row["SUPPORT_ATR"] >= 0:
        valid.append("Price is above modeled support on a volatility-adjusted basis.")
        health += 8
    else:
        invalid.append("Price has fallen below modeled support on an ATR-adjusted basis.")
        health -= 18

    if row["REGIME"] >= BUY_REGIME_THRESHOLD:
        valid.append("Broad market regime remains constructive for long setups.")
        health += 10
    elif row["REGIME"] < 50:
        invalid.append("Broad market regime is hostile enough to negate many bullish setups.")
        health -= 18
    else:
        warnings_list.append("Market regime is neutral rather than strongly supportive.")
        health -= 3

    if row["DIST_HIGH20"] > -0.01:
        warnings_list.append("The stock is somewhat extended versus the recent 20-day high.")
        health -= 5

    if progress is not None and days_in_trade >= 10 and progress < 0.01:
        warnings_list.append("Time-failure warning: the trade has not made enough progress for the time spent in it.")
        health -= 10

    health = clamp(health)
    if health >= HEALTH_ACTIVE:
        state = "Active"
        action = "Hold" if row["Entry State"] != "Buy Now" else "Enter"
    elif health >= HEALTH_RISK:
        state = "At Risk"
        action = "Trim"
    else:
        state = "Invalid"
        action = "Sell"

    if row["Entry State"] == "Buy Now" and health >= HEALTH_ACTIVE:
        action = "Buy"
    if row["Entry State"] == "Wait for Trigger" and health >= HEALTH_ACTIVE:
        action = "Wait"
    if row["Entry State"] == "Too Extended" and health >= HEALTH_ACTIVE:
        action = "Wait"
    if row["Entry State"] == "No Trade":
        action = "Avoid"

    if not invalid:
        invalid.append("Exit on a decisive close below modeled support or a clear breakdown in relative strength.")
    return state, health, action, valid, warnings_list, invalid


def build_trade_levels(row: pd.Series) -> Dict[str, float]:
    entry_low = max(row["SUPPORT"], row["Close"] - 0.5 * row["ATR14"] if pd.notna(row["ATR14"]) else row["Close"])
    entry_high = row["Close"] if row["Entry State"] == "Buy Now" else min(row["Close"] + 0.25 * row["ATR14"], row["Close"] * 1.01) if pd.notna(row["ATR14"]) else row["Close"]
    stop = row["SUPPORT"] - (1.0 * row["ATR14"] if pd.notna(row["ATR14"]) else 0)
    target1 = row["Close"] + (1.5 * row["ATR14"] if pd.notna(row["ATR14"]) else row["Close"] * 0.04)
    target2 = row["Close"] + (3.0 * row["ATR14"] if pd.notna(row["ATR14"]) else row["Close"] * 0.08)
    return {
        "Entry Low": float(entry_low),
        "Entry High": float(entry_high),
        "Stop": float(stop),
        "Target 1": float(target1),
        "Target 2": float(target2),
    }


# =========================================================
# BACKTEST ENGINE
# =========================================================
def snapshot_features_at_date(ticker: str, df: pd.DataFrame, bench: Dict[str, pd.DataFrame], asof: pd.Timestamp, regime_score: float) -> Optional[Dict[str, float]]:
    hist = df.loc[:asof].copy()
    if len(hist) < MIN_HISTORY:
        return None
    try:
        return feature_row(ticker, hist, bench, regime_score)
    except Exception:
        return None


def build_backtest_snapshots(prices: Dict[str, pd.DataFrame], bench: Dict[str, pd.DataFrame], rebalance_every: int = 20) -> pd.DataFrame:
    common_end = min(df.index.max() for df in prices.values())
    common_start = max(df.index.min() for df in prices.values())
    dates = pd.bdate_range(common_start, common_end)
    dates = [d for i, d in enumerate(dates) if i > MIN_HISTORY and i % rebalance_every == 0]

    snapshots = []
    for dt in dates:
        try:
            bench_slice = {k: v.loc[:dt] for k, v in bench.items() if len(v.loc[:dt]) >= MIN_HISTORY}
            regime = market_regime_score(bench_slice)[0]
        except Exception:
            continue
        rows = []
        for ticker, df in prices.items():
            row = snapshot_features_at_date(ticker, df, bench, dt, regime)
            if row is not None:
                rows.append(row)
        if not rows:
            continue
        ranked = score_engines(pd.DataFrame(rows))
        ranked = ranked[ranked["Entry State"].isin(["Buy Now", "Wait for Trigger"])].head(DEFAULT_TOP_N).copy()
        ranked["Snapshot Date"] = dt
        snapshots.append(ranked)

    if not snapshots:
        return pd.DataFrame()
    return pd.concat(snapshots, ignore_index=True)


def evaluate_forward_returns(snapshots: pd.DataFrame, prices: Dict[str, pd.DataFrame], hold_days: int = DEFAULT_HOLD_DAYS) -> pd.DataFrame:
    if snapshots.empty:
        return snapshots
    out = snapshots.copy()
    fwd_returns, max_up, max_down = [], [], []
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
# CHARTING
# =========================================================
def make_price_chart(df: pd.DataFrame, ticker: str, support: Optional[float] = None, target1: Optional[float] = None, target2: Optional[float] = None) -> go.Figure:
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
    if target1 is not None and pd.notna(target1):
        fig.add_hline(y=float(target1), line_dash="dot", annotation_text="Target 1")
    if target2 is not None and pd.notna(target2):
        fig.add_hline(y=float(target2), line_dash="dot", annotation_text="Target 2")
    fig.update_layout(height=550, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# =========================================================
# APP
# =========================================================
st.title("Quant Thesis Engine")
st.caption(
    "A three-engine stock decision platform: Quality Engine for stock selection, Timing Engine for entries, and Management Engine for hold/trim/exit decisions."
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
    st.write("**Buy rule baseline**")
    st.write("Regime > 55, Quality > 70, Timing > 65, then the Management Engine decides Buy / Hold / Trim / Sell.")

universe = UNIVERSE_MAP[universe_name]
all_needed = sorted(set(universe + ([custom_lookup] if custom_lookup else [])))
bench = load_benchmarks(period_years=2)
prices = load_prices(all_needed, period_years=2)

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

ranked = score_engines(pd.DataFrame(rows)).head(top_n).copy()
engine_states, engine_health, engine_actions = [], [], []
for _, row in ranked.iterrows():
    state, health, action, _, _, _ = management_engine(row)
    engine_states.append(state)
    engine_health.append(health)
    engine_actions.append(action)
ranked["Thesis State"] = engine_states
ranked["Thesis Health"] = engine_health
ranked["Action"] = engine_actions
ranked["Decision"] = ranked["Action"]

scanner_tab, detail_tab, backtest_tab, diagnostics_tab = st.tabs([
    "Scanner", "Ticker Lookup & Decision", "Backtest & Signal History", "Diagnostics"
])

with scanner_tab:
    left, right = st.columns([1.45, 1])
    with left:
        st.subheader("Top Ranked Bullish Opportunities")
        show = ranked[[
            "Ticker", "Quality Score", "Timing Score", "Pred Win %", "Entry State",
            "Thesis State", "Thesis Health", "Decision", "Close"
        ]].copy()
        show["Pred Win %"] = show["Pred Win %"].map(lambda x: f"{x:.1%}")
        for c in ["Quality Score", "Timing Score", "Thesis Health", "Close"]:
            show[c] = show[c].round(2)
        st.dataframe(show, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Market Regime")
        c1, c2 = st.columns(2)
        c1.metric("Regime Score", f"{regime:.1f} / 100")
        c2.metric("Breadth", f"{regime_diag['Breadth']:.0%}")
        st.write(
            f"SPY 20D return: **{regime_diag['SPY 20D Return']:.2%}**  \
QQQ 20D return: **{regime_diag['QQQ 20D Return']:.2%}**  \
Average index realized vol: **{regime_diag['Avg Index Vol']:.2%}**"
        )
        st.info("This regime score gates long exposure. Weak regime should mean fewer entries, tighter management, and faster exits.")

with detail_tab:
    lookup = custom_lookup if custom_lookup in prices else ranked.iloc[0]["Ticker"]
    if lookup not in prices:
        st.warning("Ticker not available from current data pull.")
    else:
        current_row = ranked[ranked["Ticker"] == lookup]
        if current_row.empty:
            detail_features = feature_row(lookup, prices[lookup], bench, regime)
            detail_row = score_engines(pd.DataFrame([detail_features])).iloc[0]
        else:
            detail_row = current_row.iloc[0]

        state, health, action, valid, warnings_list, invalid = management_engine(detail_row)
        levels = build_trade_levels(detail_row)
        meta = get_ticker_meta(lookup)

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Quality", f"{detail_row['Quality Score']:.1f}")
        m2.metric("Timing", f"{detail_row['Timing Score']:.1f}")
        m3.metric("Pred. Win %", f"{detail_row['Pred Win %']:.1%}")
        m4.metric("Entry State", detail_row["Entry State"])
        m5.metric("Thesis Health", f"{health:.1f}")
        m6.metric("Decision", action)

        st.subheader(f"{lookup} — {meta['shortName']}")
        st.write(f"**Sector:** {meta['sector']}  \
**Industry:** {meta['industry']}  \
**Price:** {detail_row['Close']:.2f}")
        st.plotly_chart(
            make_price_chart(prices[lookup], lookup, detail_row["SUPPORT"], levels["Target 1"], levels["Target 2"]),
            use_container_width=True,
        )

        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.subheader("Three-Engine Breakdown")
            factor_df = pd.DataFrame({
                "Engine / Bucket": [
                    "Quality - Trend", "Quality - Momentum", "Quality - Relative Strength", "Quality - Path Quality",
                    "Quality - Participation", "Quality - Regime Fit", "Timing - Pullback Geometry", "Timing - Trigger Quality",
                    "Timing - Entry R/R"
                ],
                "Score": [
                    detail_row["Trend Score"], detail_row["Momentum Score"], detail_row["Relative Strength Score"],
                    detail_row["Path Quality Score"], detail_row["Participation Score"], detail_row["Regime Fit Score"],
                    detail_row["Pullback Geometry Score"], detail_row["Trigger Quality Score"], detail_row["Entry RR Score"]
                ]
            })
            st.dataframe(factor_df.style.format({"Score": "{:.1f}"}), use_container_width=True, hide_index=True)

            st.subheader("Buy / Sell Decision Framework")
            st.write(f"- **Buy** only if Regime > {BUY_REGIME_THRESHOLD}, Quality > {QUALITY_THRESHOLD}, Timing > {TIMING_THRESHOLD}.")
            st.write("- **Hold** while thesis health stays strong and support / relative strength remain intact.")
            st.write("- **Trim** when health slips into the At Risk range or after target 1 if extension builds.")
            st.write("- **Sell** when thesis turns Invalid, support breaks, RS breaks, or time-failure logic trips.")

            st.subheader("Trade Levels")
            st.write(f"- Entry Zone: **{levels['Entry Low']:.2f} to {levels['Entry High']:.2f}**")
            st.write(f"- Stop: **{levels['Stop']:.2f}**")
            st.write(f"- Target 1: **{levels['Target 1']:.2f}**")
            st.write(f"- Target 2: **{levels['Target 2']:.2f}**")

            st.subheader("What must happen for this thesis to stay right")
            must_happen = [
                f"Price should hold above modeled support near {detail_row['SUPPORT']:.2f}.",
                "Relative strength versus SPY and QQQ should remain positive.",
                "The broad market regime should remain constructive for long setups.",
                "The stock should start progressing upward within the next several sessions rather than stall.",
                "Pullbacks should remain orderly and not expand into heavy-volume damage.",
            ]
            for item in must_happen:
                st.write(f"- {item}")

        with c2:
            st.subheader("Management Engine")
            if state == "Active":
                st.success(f"State: {state}")
            elif state == "At Risk":
                st.warning(f"State: {state}")
            else:
                st.error(f"State: {state}")
            st.write(f"**Decision:** {action}")
            st.write("**Why it is valid**")
            for item in valid:
                st.write(f"- {item}")
            st.write("**Warnings**")
            for item in warnings_list:
                st.write(f"- {item}")
            st.write("**Invalidation / exit triggers**")
            for item in invalid:
                st.write(f"- {item}")

        st.subheader("Raw Quant Metrics")
        raw_cols = [
            "RET_1M", "RET_3M", "RET_6M", "ACCEL", "RS_SPY", "RS_QQQ", "RS_SLOPE20", "RS_SLOPE60",
            "RSI14", "VOL20", "VOL60", "SEMI60", "DIST_HIGH20", "SUPPORT_ATR", "RR_PROXY", "AVG_DOLLAR_VOL"
        ]
        raw = detail_row[raw_cols].to_frame("Value")
        display = raw.copy()
        for idx in display.index:
            val = display.loc[idx, "Value"]
            if idx == "AVG_DOLLAR_VOL":
                display.loc[idx, "Value"] = f"${val:,.0f}"
            elif idx in ["RSI14", "RR_PROXY", "RS_SLOPE20", "RS_SLOPE60", "SUPPORT_ATR", "ACCEL"]:
                display.loc[idx, "Value"] = f"{val:.2f}"
            else:
                display.loc[idx, "Value"] = f"{val:.2%}"
        st.dataframe(display, use_container_width=True)

with backtest_tab:
    st.subheader("Backtest & Signal History")
    if not run_backtest:
        st.info("Enable backtest in the sidebar to compute historical signal performance.")
    else:
        with st.spinner("Building historical snapshots and evaluating forward returns..."):
            bt_universe = {k: v for k, v in prices.items() if k in universe}
            bt_snapshots = build_backtest_snapshots(bt_universe, bench, rebalance_every=rebalance_days)
            bt_results = evaluate_forward_returns(bt_snapshots, bt_universe, hold_days=hold_days)

        if bt_results.empty:
            st.warning("No backtest results available yet.")
        else:
            avg_return = bt_results[f"Fwd {hold_days}D Return"].mean()
            hit_rate = bt_results["Hit"].mean()
            median_return = bt_results[f"Fwd {hold_days}D Return"].median()
            avg_up = bt_results[f"Max Up {hold_days}D"].mean()
            avg_down = bt_results[f"Max Down {hold_days}D"].mean()

            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("Hit Rate", f"{hit_rate:.1%}")
            b2.metric("Avg Return", f"{avg_return:.2%}")
            b3.metric("Median Return", f"{median_return:.2%}")
            b4.metric("Avg Max Up", f"{avg_up:.2%}")
            b5.metric("Avg Max Down", f"{avg_down:.2%}")

            st.write("**Interpretation**")
            st.write(
                "This is the historical signal history for names that passed the entry engine filters on each rebalance date. "
                "Use it to compare hit rate, forward-return profile, and path risk." 
            )

            history_show = bt_results[[
                "Snapshot Date", "Ticker", "Quality Score", "Timing Score", "Pred Win %", "Entry State",
                f"Fwd {hold_days}D Return", f"Max Up {hold_days}D", f"Max Down {hold_days}D", "Hit"
            ]].copy()
            history_show["Pred Win %"] = history_show["Pred Win %"].map(lambda x: f"{x:.1%}")
            for c in [f"Fwd {hold_days}D Return", f"Max Up {hold_days}D", f"Max Down {hold_days}D"]:
                history_show[c] = history_show[c].map(lambda x: f"{x:.2%}")
            st.dataframe(history_show.sort_values("Snapshot Date", ascending=False), use_container_width=True, hide_index=True)

            st.subheader("Average Forward Return by Entry State")
            st.bar_chart(bt_results.groupby("Entry State")[f"Fwd {hold_days}D Return"].mean().sort_values(ascending=False))

            st.subheader("Average Forward Return by Quality Bucket")
            bt_bucket = bt_results.copy()
            bt_bucket["Quality Bucket"] = pd.cut(bt_bucket["Quality Score"], bins=[0, 60, 70, 80, 100], labels=["<60", "60-70", "70-80", "80+"])
            st.bar_chart(bt_bucket.groupby("Quality Bucket", observed=False)[f"Fwd {hold_days}D Return"].mean())

with diagnostics_tab:
    st.subheader("Diagnostics")
    diag = ranked[[
        "Ticker", "Quality Score", "Timing Score", "Trend Score", "Momentum Score", "Relative Strength Score",
        "Path Quality Score", "Participation Score", "Regime Fit Score", "Pred Win %", "Entry State", "Decision"
    ]].copy()
    diag["Pred Win %"] = diag["Pred Win %"].map(lambda x: f"{x:.1%}")
    st.dataframe(diag, use_container_width=True, hide_index=True)

    st.subheader("Current Score Distribution")
    score_dist = ranked[["Ticker", "Quality Score", "Timing Score", "Trend Score", "Relative Strength Score"]].set_index("Ticker")
    st.bar_chart(score_dist)

st.caption(
    "This version implements the v3 architecture: explicit Quality, Timing, and Management engines; buy/hold/trim/exit actions; structured trade levels; and backtestable entry-state logic. "
    "The next serious upgrade would be persistent signal storage, a larger dynamic universe, and a more realistic execution backtest with stop/target handling."
)
