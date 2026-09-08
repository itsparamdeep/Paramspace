"""
VWAP Tester (v0.1)
====================

WHAT IS VWAP
-------------
Volume Weighted Average Price: the running average price paid so far
TODAY, weighted by how much volume traded at each price. It resets to
zero at the start of every trading session - yesterday's VWAP means
nothing once a new day starts. That's why this needs INTRADAY bars
(1-5 minute), not daily bars like the Roger scanner - a daily bar has
no "so far today" to average over.

    VWAP_t = sum(typical_price_i * volume_i for i <= t) / sum(volume_i for i <= t)
    typical_price = (High + Low + Close) / 3

The running standard deviation around VWAP (same cumulative idea) gives
you bands - "price is 2 standard deviations above VWAP right now" - the
basis for the mean-reversion mode below.

TWO MODES (pass mode='reversion' or mode='trend')
---------------------------------------------------
  reversion : price has stretched >= band_k standard deviations away
              from VWAP -> fade it back toward VWAP.
  trend     : price has just crossed from one side of the VWAP line to
              the other -> trade in that new direction.

Both modes: one position at a time, enter on the NEXT bar's open after
the signal bar's close (no lookahead), and ALWAYS force-flatten at the
last bar of the trading day - day trades don't carry overnight risk,
and tomorrow's VWAP starts over from zero anyway.

DATA CAVEAT - READ BEFORE TRUSTING THE NUMBERS
-------------------------------------------------
Yahoo/yfinance caps intraday history: 5-minute bars go back about 60
days, 1-minute bars only about 7 days. That's a genuinely small sample
- a handful of weeks of trading days. Treat any Sharpe/win-rate here as
a rough first read, not a verdict; the small-sample-fluke problem from
the daily scanner applies here even more.
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
# Config - edit these directly
# ----------------------------------------------------------------------------

INTERVAL = "5m"    # "1m" (~7d history), "5m"/"15m"/"30m" (~60d), "60m" (~2y)
PERIOD = "60d"     # how far back to pull - must match what INTERVAL allows


# ----------------------------------------------------------------------------
# VWAP + bands
# ----------------------------------------------------------------------------

def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Adds vwap, vwap_std, and vwap_z (how many std devs price is from
    VWAP right now) - all reset at the start of each trading day.

    Uses the identity Var(X) = E[X^2] - E[X]^2 (volume-weighted version)
    so the running std can be built from three cumulative sums, same as
    VWAP itself - no need to re-scan past bars on every step.
    """
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"]
    day = df.index.normalize()

    cum_vol = vol.groupby(day).cumsum()
    cum_tpv = (tp * vol).groupby(day).cumsum()
    cum_tp2v = (tp ** 2 * vol).groupby(day).cumsum()

    vwap = cum_tpv / cum_vol
    var = (cum_tp2v / cum_vol) - vwap ** 2
    std = np.sqrt(var.clip(lower=0))

    out = df.copy()
    out["vwap"] = vwap
    out["vwap_std"] = std
    out["vwap_z"] = (df["Close"] - vwap) / std.replace(0, np.nan)
    return out


# ----------------------------------------------------------------------------
# Signals
# ----------------------------------------------------------------------------

def scan_vwap_setups(
    df: pd.DataFrame,
    mode: str = "reversion",
    band_k: float = 2.0,
    min_bars_since_open: int = 6,
) -> pd.DataFrame:
    """mode='reversion': fire when |vwap_z| >= band_k (price stretched
    too far from VWAP).
    mode='trend': fire on a clean cross of price through the VWAP line.

    `min_bars_since_open` skips the first few bars of each day - VWAP
    is unstable right at the open (barely any volume has accumulated
    yet, so it swings around a lot)."""
    d = compute_vwap(df)
    day = d.index.normalize()
    bar_in_day = pd.Series(range(len(d)), index=d.index).groupby(day).cumcount()

    records = []
    prev_day = None
    prev_side = None  # 'above' / 'below' VWAP - trend mode only, reset each day

    for i in range(len(d)):
        row = d.iloc[i]
        today = day[i]
        if today != prev_day:
            prev_day = today
            prev_side = None

        if bar_in_day.iloc[i] < min_bars_since_open or pd.isna(row["vwap_z"]):
            continue

        if mode == "reversion":
            if row["vwap_z"] >= band_k:
                records.append(dict(date=d.index[i], close=round(row["Close"], 4),
                                     vwap=round(row["vwap"], 4), z=round(row["vwap_z"], 2),
                                     action="SHORT"))
            elif row["vwap_z"] <= -band_k:
                records.append(dict(date=d.index[i], close=round(row["Close"], 4),
                                     vwap=round(row["vwap"], 4), z=round(row["vwap_z"], 2),
                                     action="LONG"))

        elif mode == "trend":
            side = "above" if row["Close"] > row["vwap"] else "below"
            if prev_side is not None and side != prev_side:
                action = "LONG" if side == "above" else "SHORT"
                records.append(dict(date=d.index[i], close=round(row["Close"], 4),
                                     vwap=round(row["vwap"], 4), z=round(row["vwap_z"], 2),
                                     action=action))
            prev_side = side

        else:
            raise ValueError("mode must be 'reversion' or 'trend'")

    return pd.DataFrame(records)


# ----------------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------------

def backtest_intraday(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    stop_pct: float = 0.005,
    target_pct: float = 0.01,
    max_hold_bars: int = 30,
) -> tuple[pd.DataFrame, pd.Series]:
    """One position at a time. Enters at the NEXT bar's open after the
    signal bar's close (no lookahead). Exits at whichever comes first:
    stop_pct loss, target_pct gain, max_hold_bars elapsed, or the LAST
    bar of the trading day (forced flatten - no overnight holds)."""
    day = df.index.normalize()
    dates = df.index
    opens = df["Open"]
    closes = df["Close"]
    idx_of = {t: k for k, t in enumerate(dates)}
    day_arr = day.to_numpy()
    n = len(df)

    bar_returns = pd.Series(0.0, index=dates)
    trades = []
    last_exit_idx = -1

    for _, row in signals.iterrows():
        sig_idx = idx_of[row["date"]]
        if sig_idx <= last_exit_idx:
            continue

        entry_idx = sig_idx + 1
        if entry_idx >= n or day_arr[entry_idx] != day_arr[sig_idx]:
            continue  # signal too near end of day - no room to enter+exit same session

        direction = 1 if row["action"] == "LONG" else -1
        entry_price = float(opens.iloc[entry_idx])
        today = day_arr[entry_idx]
        day_end_idx = int(np.max(np.where(day_arr == today)[0]))
        max_exit_idx = min(entry_idx + max_hold_bars, day_end_idx)

        exit_idx, exit_price = max_exit_idx, float(closes.iloc[max_exit_idx])
        for k in range(entry_idx, max_exit_idx + 1):
            ret = direction * (closes.iloc[k] / entry_price - 1)
            if ret <= -stop_pct or ret >= target_pct or k == max_exit_idx:
                exit_idx, exit_price = k, float(closes.iloc[k])
                break

        for k in range(entry_idx, exit_idx + 1):
            if k == entry_idx:
                bar_returns.iloc[k] = direction * (closes.iloc[k] / entry_price - 1)
            else:
                bar_returns.iloc[k] = direction * (closes.iloc[k] / closes.iloc[k - 1] - 1)

        trade_return = direction * (exit_price / entry_price - 1)
        trades.append(dict(
            entry_time=dates[entry_idx], exit_time=dates[exit_idx], action=row["action"],
            entry=round(entry_price, 4), exit=round(exit_price, 4),
            return_pct=round(trade_return * 100, 3), bars_held=exit_idx - entry_idx,
        ))
        last_exit_idx = exit_idx

    return pd.DataFrame(trades), bar_returns


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def bars_per_year(interval: str) -> float:
    """Rough annualization factor: (bars in a 6.5hr session) x 252 days.
    This is what turns per-bar Sharpe into an 'annualized' number - treat
    it as a scale, not a promise about a real year of data."""
    session_minutes = 390
    minutes = {"1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "90m": 90}
    return (session_minutes / minutes.get(interval, 5)) * 252


def performance_metrics(bar_returns: pd.Series, trades: pd.DataFrame, freq: float) -> dict:
    equity = (1 + bar_returns).cumprod()
    total_return = equity.iloc[-1] - 1
    sharpe = (bar_returns.mean() / bar_returns.std() * np.sqrt(freq)
              if bar_returns.std() > 0 else np.nan)
    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()

    wins = trades.loc[trades["return_pct"] > 0, "return_pct"]
    losses = trades.loc[trades["return_pct"] <= 0, "return_pct"]
    win_rate = len(wins) / len(trades) if len(trades) else np.nan
    profit_factor = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else np.nan

    def r(x, dec=2):
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), dec)

    return dict(
        total_return_pct=r(total_return * 100),
        sharpe_annualized=r(sharpe),
        max_drawdown_pct=r(max_dd * 100),
        num_trades=len(trades),
        win_rate_pct=r(win_rate * 100, 1),
        avg_win_pct=r(wins.mean(), 3) if len(wins) else None,
        avg_loss_pct=r(losses.mean(), 3) if len(losses) else None,
        profit_factor=r(profit_factor),
    )


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------

def run(ticker: str = "AAPL", mode: str = "reversion", **kwargs) -> None:
    if yf is None:
        raise RuntimeError("pip install yfinance")

    df = yf.download(ticker, period=PERIOD, interval=INTERVAL, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()

    print(f"[VWAPTester] {ticker}: {len(df)} bars ({INTERVAL} over {PERIOD}), mode={mode}")
    signals = scan_vwap_setups(df, mode=mode, **{k: v for k, v in kwargs.items() if k in
                               ("band_k", "min_bars_since_open")})
    print(f"[VWAPTester] {len(signals)} setups found")
    if signals.empty:
        return

    trades, bar_returns = backtest_intraday(df, signals, **{k: v for k, v in kwargs.items() if k in
                                            ("stop_pct", "target_pct", "max_hold_bars")})
    metrics = performance_metrics(bar_returns, trades, bars_per_year(INTERVAL))

    print(f"\n--- Trades ({len(trades)}) ---")
    print(trades.to_string(index=False))
    print(f"\n--- Performance ---")
    for k, v in metrics.items():
        print(f"{k:>18}: {v}")

    out = f"vwap_backtest_{ticker}_{mode}_{datetime.now():%Y%m%d}.csv"
    trades.to_csv(out, index=False)
    print(f"\n[VWAPTester] trades saved to {out}")


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    mode = sys.argv[2] if len(sys.argv) > 2 else "reversion"
    run(ticker, mode)
