"""
Roger - Range/Reversal Scanner + Backtester (v0.1)
====================================================

STRATEGY IDEA
--------------
A stock that has tested the same support or resistance level at least
twice without ever closing through it is "respecting a range". If it is
ALSO overbought (RSI > 70) at resistance, or oversold (RSI < 30) at
support, that's the setup: fade the extreme back toward the middle of
the range.

    Testing RESISTANCE + overbought  ->  bearish setup  ->  (buy PUTS)
    Testing SUPPORT    + oversold    ->  bullish setup  ->  (buy CALLS)

FOUR BUILDING BLOCKS (read in this order, that's how the file is laid out)
----------------------------------------------------------------------
  1. rsi()                  - the oscillator
  2. find_active_levels()   - is there a level here that's been touched
                               >=2 times and never broken?
  3. scan_setups()          - walk the series day by day (causal only,
                               no lookahead) and flag qualifying days
  4. backtest()             - turn signals into simulated trades
     performance_metrics()  - Sharpe / drawdown / CAGR / win rate

CAVEAT - PLEASE READ
---------------------
This backtests the UNDERLYING, not a real option. Free data (yfinance)
has no historical strikes/IV, so there is no honest way to backtest
"the put" or "the call" itself. Trading the underlying long/short is a
reasonable proxy for a deep-ITM option or a debit spread, but it does
NOT capture theta decay, IV crush, or the leverage of an option. Once
this shows real directional edge, the move is:
  - Paper trade the actual option structure for a few months, or
  - Buy historical options data (CBOE DataShop, ORATS, Polygon.io) and
    re-run the backtest pricing an actual spread with Black-Scholes.
Treat every number below as "does the DIRECTION have edge", not "here
is what my P&L on puts/calls would have been".
"""

from __future__ import annotations

import sys
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None


# ----------------------------------------------------------------------------
# 1. RSI
# ----------------------------------------------------------------------------

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ----------------------------------------------------------------------------
# 2. Level detection
# ----------------------------------------------------------------------------

def find_active_levels(
    window: pd.DataFrame,
    touch_tol: float = 0.015,
    break_tol: float = 0.01,
    min_touches: int = 2,
) -> tuple[list[float], list[float]]:
    """Given a trailing window of bars ending TODAY, find resistance and
    support levels that have been touched at least `min_touches` times
    and never closed through.

    Heuristic: every day's High is a *candidate resistance level*; it
    counts as touched once by every other day whose High lands within
    `touch_tol` of it. A candidate is thrown out if the window ever
    CLOSED more than `break_tol` above it (i.e. the level actually
    broke). Same idea for Lows / support, mirrored below.

    This is simple on purpose - it's the thing to refine once you've
    seen it work (e.g. requiring touches to be spaced N days apart so
    one noisy spike doesn't masquerade as two touches).
    """
    highs = window["High"].to_numpy()
    lows = window["Low"].to_numpy()
    closes = window["Close"].to_numpy()

    def valid_levels(prices: np.ndarray, is_resistance: bool) -> list[float]:
        levels = []
        for c in prices:
            touches = np.sum(np.abs(prices / c - 1) <= touch_tol)
            if touches < min_touches:
                continue
            broken = (
                np.any(closes > c * (1 + break_tol))
                if is_resistance
                else np.any(closes < c * (1 - break_tol))
            )
            if not broken:
                levels.append(round(float(c), 2))
        return sorted(set(levels))

    return valid_levels(highs, True), valid_levels(lows, False)


# ----------------------------------------------------------------------------
# 3. Scan for setups (causal: only uses data up to and including day t)
# ----------------------------------------------------------------------------

def scan_setups(
    df: pd.DataFrame,
    lookback: int = 60,
    touch_tol: float = 0.015,
    break_tol: float = 0.01,
    min_touches: int = 2,
    rsi_overbought: float = 70,
    rsi_oversold: float = 30,
) -> pd.DataFrame:
    close = df["Close"]
    rsi_series = rsi(close)
    records = []

    for t in range(lookback, len(df)):
        today_rsi = rsi_series.iloc[t]
        if pd.isna(today_rsi):
            continue
        today_close = close.iloc[t]
        window = df.iloc[t - lookback : t + 1]
        resistances, supports = find_active_levels(window, touch_tol, break_tol, min_touches)

        testing_res = next((r for r in resistances if abs(today_close / r - 1) <= touch_tol), None)
        testing_sup = next((s for s in supports if abs(today_close / s - 1) <= touch_tol), None)

        if testing_res is not None and today_rsi >= rsi_overbought:
            records.append(dict(date=df.index[t], close=round(today_close, 2),
                                 setup="RESISTANCE", level=testing_res,
                                 rsi=round(today_rsi, 1), action="PUT"))
        elif testing_sup is not None and today_rsi <= rsi_oversold:
            records.append(dict(date=df.index[t], close=round(today_close, 2),
                                 setup="SUPPORT", level=testing_sup,
                                 rsi=round(today_rsi, 1), action="CALL"))

    return pd.DataFrame(records)


# ----------------------------------------------------------------------------
# 4. Backtest + metrics
# ----------------------------------------------------------------------------

def backtest(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    hold_days: int = 10,
    stop_pct: float = 0.03,
    target_pct: float = 0.05,
) -> tuple[pd.DataFrame, pd.Series]:
    """Simulate one position at a time (no overlapping trades).

    Entry: next bar's OPEN after the signal day (the signal only used
    data through the signal day's close, so this has no lookahead).
    Exit: whichever comes first - stop_pct loss, target_pct gain, or
    hold_days trading days.

    Returns (trades_df, daily_returns) where daily_returns is a full
    length series (0 on flat days) used to compute Sharpe/drawdown.
    """
    dates = df.index
    opens = df["Open"]
    closes = df["Close"]
    idx_of = {d: k for k, d in enumerate(dates)}
    n = len(df)

    daily_returns = pd.Series(0.0, index=dates)
    trades = []
    last_exit_idx = -1

    for _, row in signals.iterrows():
        sig_idx = idx_of[row["date"]]
        if sig_idx <= last_exit_idx:
            continue  # still in a position, skip overlapping signal

        entry_idx = sig_idx + 1
        if entry_idx >= n:
            continue

        direction = 1 if row["action"] == "CALL" else -1
        entry_price = float(opens.iloc[entry_idx])
        max_exit_idx = min(entry_idx + hold_days, n - 1)

        exit_idx = max_exit_idx
        exit_price = float(closes.iloc[max_exit_idx])
        for k in range(entry_idx, max_exit_idx + 1):
            ret = direction * (closes.iloc[k] / entry_price - 1)
            if ret <= -stop_pct or ret >= target_pct or k == max_exit_idx:
                exit_idx, exit_price = k, float(closes.iloc[k])
                break

        # day-by-day mark-to-market for the Sharpe/drawdown series
        for k in range(entry_idx, exit_idx + 1):
            if k == entry_idx:
                daily_returns.iloc[k] = direction * (closes.iloc[k] / entry_price - 1)
            else:
                daily_returns.iloc[k] = direction * (closes.iloc[k] / closes.iloc[k - 1] - 1)

        trade_return = direction * (exit_price / entry_price - 1)
        trades.append(dict(
            entry_date=dates[entry_idx], exit_date=dates[exit_idx], action=row["action"],
            level=row["level"], entry=round(entry_price, 2), exit=round(exit_price, 2),
            return_pct=round(trade_return * 100, 2), days_held=exit_idx - entry_idx,
        ))
        last_exit_idx = exit_idx

    return pd.DataFrame(trades), daily_returns


def performance_metrics(daily_returns: pd.Series, trades: pd.DataFrame, freq: int = 252) -> dict:
    equity = (1 + daily_returns).cumprod()
    years = len(daily_returns) / freq
    total_return = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(freq)
              if daily_returns.std() > 0 else np.nan)
    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()

    wins = trades.loc[trades["return_pct"] > 0, "return_pct"]
    losses = trades.loc[trades["return_pct"] <= 0, "return_pct"]
    win_rate = len(wins) / len(trades) if len(trades) else np.nan
    profit_factor = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else np.nan

    def r(x, d=2):
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)

    return dict(
        total_return_pct=r(total_return * 100),
        cagr_pct=r(cagr * 100),
        sharpe=r(sharpe),
        max_drawdown_pct=r(max_dd * 100),
        num_trades=len(trades),
        win_rate_pct=r(win_rate * 100, 1),
        avg_win_pct=r(wins.mean()) if len(wins) else None,
        avg_loss_pct=r(losses.mean()) if len(losses) else None,
        profit_factor=r(profit_factor),
    )


# ----------------------------------------------------------------------------
# Universe scanning
# ----------------------------------------------------------------------------

def get_sp500_tickers() -> list[str]:
    """Scrapes the current S&P 500 list from Wikipedia. Needs `lxml`
    installed (pip install lxml) for pandas to parse the HTML table.

    Wikipedia returns HTTP 403 to requests with no User-Agent header
    (which is what pandas sends by default), so fetch the page with
    `requests` and a browser-like header first, then hand the HTML
    text to pandas instead of the URL."""
    import io
    import requests

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    table = pd.read_html(io.StringIO(resp.text))[0]
    return sorted(table["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist())


def run_universe(
    tickers: list[str],
    period: str = "3y",
    min_trades: int = 3,
    **kwargs,
) -> pd.DataFrame:
    """Run scan_setups + backtest across many tickers and return one summary
    row per ticker, sorted by Sharpe (best first). Tickers with fewer than
    `min_trades` closed trades are dropped - too few trades to trust a
    Sharpe/win-rate number computed from them."""
    scan_kwargs = {k: v for k, v in kwargs.items() if k in
                   ("lookback", "touch_tol", "break_tol", "min_touches",
                    "rsi_overbought", "rsi_oversold")}
    bt_kwargs = {k: v for k, v in kwargs.items() if k in
                 ("hold_days", "stop_pct", "target_pct")}

    rows = []
    for i, tkr in enumerate(tickers, 1):
        try:
            df = yf.download(tkr, period=period, interval="1d", auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            if len(df) < 150:
                continue
            signals = scan_setups(df, **scan_kwargs)
            if signals.empty:
                continue
            trades, daily_returns = backtest(df, signals, **bt_kwargs)
            if len(trades) < min_trades:
                continue
            m = performance_metrics(daily_returns, trades)
            m["ticker"] = tkr
            rows.append(m)
        except Exception:
            continue  # bad ticker / no data / delisted - skip and keep going
        if i % 25 == 0:
            print(f"[Roger]  ... {i}/{len(tickers)} scanned")

    if not rows:
        return pd.DataFrame()
    summary = pd.DataFrame(rows)
    summary = summary[["ticker"] + [c for c in summary.columns if c != "ticker"]]
    return summary.sort_values("sharpe", ascending=False, na_position="last")


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------

def run(ticker: str = "AAPL", period: str = "3y", **kwargs) -> None:
    if yf is None:
        raise RuntimeError("pip install yfinance")
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()

    print(f"[Roger] {ticker}: {len(df)} bars, scanning for setups ...")
    signals = scan_setups(df, **{k: v for k, v in kwargs.items() if k in
                          ("lookback", "touch_tol", "break_tol", "min_touches",
                           "rsi_overbought", "rsi_oversold")})
    print(f"[Roger] {len(signals)} setups found")
    if signals.empty:
        return

    trades, daily_returns = backtest(df, signals, **{k: v for k, v in kwargs.items() if k in
                                     ("hold_days", "stop_pct", "target_pct")})
    metrics = performance_metrics(daily_returns, trades)

    print(f"\n--- Trades ({len(trades)}) ---")
    print(trades.to_string(index=False))
    print(f"\n--- Performance ---")
    for k, v in metrics.items():
        print(f"{k:>18}: {v}")

    out = f"roger_backtest_{ticker}_{datetime.now():%Y%m%d}.csv"
    trades.to_csv(out, index=False)
    print(f"\n[Roger] trades saved to {out}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "SP500"

    if arg.upper() in ("SP500", "ALL", "UNIVERSE"):
        tickers = get_sp500_tickers()
        print(f"[Roger] scanning all {len(tickers)} S&P 500 tickers - this takes a while ...")
        summary = run_universe(tickers)
        if summary.empty:
            print("[Roger] no tickers produced enough qualifying trades to report.")
        else:
            out = f"roger_universe_scan_{datetime.now():%Y%m%d}.csv"
            summary.to_csv(out, index=False)
            print(f"\n--- Top by Sharpe ({len(summary)} tickers qualified) ---")
            print(summary.head(30).to_string(index=False))
            print(f"\n[Roger] full summary saved to {out}")

    elif "," in arg:
        tickers = [t.strip().upper() for t in arg.split(",")]
        print(f"[Roger] scanning basket: {tickers}")
        summary = run_universe(tickers, min_trades=1)
        if summary.empty:
            print("[Roger] no tickers in this basket produced qualifying trades.")
        else:
            print(summary.to_string(index=False))

    else:
        run(arg)