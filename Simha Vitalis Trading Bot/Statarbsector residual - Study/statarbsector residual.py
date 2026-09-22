"""
Sector-Residual Statistical Arbitrage (v0.1)
==============================================

The retail-buildable core of Epstein, Wang, Choi and Pelger (2025),
"Attention Factors for Statistical Arbitrage", which itself builds on
Avellaneda and Lee (2010), "Statistical arbitrage in the US equities
market". The paper's neural net learns factors whose loadings look like
industry sectors and whose signal comes from past returns. This file
uses sectors directly and a classical mean-reversion model for the
signal. Less powerful, fully transparent, runs on free data.

THE MACHINE
------------
For every stock, every day, using only the last WINDOW trading days:

  1. FACTOR MODEL
       r_stock = alpha + beta * r_sector_etf + eps
     beta is the hedge ratio. eps is what the sector does NOT explain.

  2. RESIDUAL PROCESS
     X = cumulative sum of eps over the window. Fit an AR(1):
       X[t+1] = a + b * X[t] + noise
     which is a discrete Ornstein-Uhlenbeck process. From it:
       kappa     = -ln(b)            reversion speed per day
       half_life = ln(2) / kappa     days to close half the gap
       m         = a / (1 - b)       equilibrium level
       sigma_eq  = sqrt(var(noise) / (1 - b^2))
       s_score   = (X[t] - m) / sigma_eq
     s tells you how stretched the residual is, in units of its own
     equilibrium noise. Only stocks with half_life <= MAX_HALF_LIFE
     are tradeable: if it reverts too slowly, it isn't arbitrage.

  3. TRADING RULES (Avellaneda-Lee thresholds)
       open  long  when s < -ENTRY_S        close when s > -EXIT_LONG_S
       open  short when s >  ENTRY_S        close when s <  EXIT_SHORT_S
     plus a MAX_HOLD_DAYS safety exit. Each stock position is hedged
     with beta dollars of its sector ETF, so the position is a bet on
     the residual only, not on the sector or the market.

  4. PORTFOLIO
     Fixed slot size per position, MAX_POSITIONS_PER_SIDE slots per
     side, unused slots sit in cash. Candidates ranked by |s|.

TIMING (no lookahead)
----------------------
Signal computed from closes through day t. Position taken at the close
of day t, earns the return from t to t+1. This assumes you can act at
the close; it's the same convention as the papers. If that bothers you,
the shift is one line in backtest().

LONG-ONLY MODE
---------------
LONG_ONLY = True drops the short side and the hedge. That's what a
$1,000 account without a margin/shorting facility can actually run.
It is NOT market neutral any more; expect it to carry SPY beta and to
look worse in a selloff. Run both and compare; the gap is the price of
not being able to short.

COSTS
------
COST_PER_SIDE on every dollar of turnover (stock and hedge legs) plus
SHORT_BORROW_APR on all short notional. Large caps are cheap to borrow;
0.5% is a fair average.

HONESTY
--------
Survivorship bias: this uses TODAY's S&P 500 members. Stocks that got
kicked out are missing, which flatters results a little, mostly on the
long side. The paper uses CRSP with delistings. Sample: START to today.
"""

from __future__ import annotations

import io
import sys
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

START = "2019-01-01"
WINDOW = 60                  # days for beta and OU fit
ENTRY_S = 1.25
EXIT_LONG_S = 0.50           # close long when s > -0.50
EXIT_SHORT_S = 0.75          # close short when s < 0.75
MAX_HALF_LIFE = 20           # days; slower reversion is not traded
MAX_HOLD_DAYS = 60
MAX_POSITIONS_PER_SIDE = 15
LONG_ONLY = False
HEDGE = True                 # ignored when LONG_ONLY
COST_PER_SIDE = 0.0005       # 5 bps per dollar traded, each side
SHORT_BORROW_APR = 0.005
ACCOUNT_SIZE = 1000.0
BENCHMARK = "SPY"

SECTOR_ETF = {
    "Information Technology": "XLK", "Financials": "XLF", "Health Care": "XLV",
    "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Energy": "XLE",
    "Industrials": "XLI", "Materials": "XLB", "Utilities": "XLU",
    "Real Estate": "XLRE", "Communication Services": "XLC",
}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def get_sp500_with_sectors() -> pd.DataFrame:
    import requests
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}, timeout=15)
    resp.raise_for_status()
    t = pd.read_html(io.StringIO(resp.text))[0]
    out = pd.DataFrame({
        "ticker": t["Symbol"].astype(str).str.replace(".", "-", regex=False),
        "sector": t["GICS Sector"].astype(str),
    })
    out["etf"] = out["sector"].map(SECTOR_ETF)
    return out.dropna().reset_index(drop=True)


def load_prices(tickers: list[str], start: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("pip install yfinance")
    px = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    px = px.dropna(axis=1, thresh=int(len(px) * 0.95)).ffill().dropna()
    return px


# ----------------------------------------------------------------------------
# Signals: rolling factor regression + OU fit, vectorised across stocks
# ----------------------------------------------------------------------------

def compute_signals(R: np.ndarray, E: np.ndarray, window: int = WINDOW,
                    max_half_life: float = MAX_HALF_LIFE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R: (T, N) stock daily returns. E: (T, N) the matching sector ETF
    return for each stock column. Returns S (s-scores), BETA, HL, each
    (T, N), NaN where not computable or half-life too slow."""
    T, N = R.shape
    S = np.full((T, N), np.nan)
    BETA = np.full((T, N), np.nan)
    HL = np.full((T, N), np.nan)

    for i in range(window - 1, T):
        r = R[i - window + 1: i + 1]
        e = E[i - window + 1: i + 1]

        # 1. factor regression, one OLS per stock, all at once
        rm, em = r - r.mean(0), e - e.mean(0)
        beta = (rm * em).sum(0) / np.where((em * em).sum(0) == 0, np.nan, (em * em).sum(0))
        alpha = r.mean(0) - beta * e.mean(0)
        eps = r - alpha - beta * e

        # 2. OU fit on the cumulative residual
        X = eps.cumsum(0)
        x0, x1 = X[:-1], X[1:]
        x0m, x1m = x0 - x0.mean(0), x1 - x1.mean(0)
        denom = (x0m * x0m).sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            b = (x0m * x1m).sum(0) / np.where(denom == 0, np.nan, denom)
            a = x1.mean(0) - b * x0.mean(0)
            noise = x1 - a - b * x0
            var_noise = noise.var(0)
            valid = (b > 0) & (b < 1)
            kappa = np.where(valid, -np.log(np.clip(b, 1e-9, 1 - 1e-9)), np.nan)
            half_life = np.log(2) / kappa
            m = a / (1 - b)
            sigma_eq = np.sqrt(var_noise / (1 - b ** 2))
            s = (X[-1] - m) / sigma_eq

        ok = valid & (half_life <= max_half_life) & np.isfinite(s)
        S[i] = np.where(ok, s, np.nan)
        BETA[i] = beta
        HL[i] = np.where(ok, half_life, np.nan)

    return S, BETA, HL


# ----------------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------------

def backtest(R: np.ndarray, E: np.ndarray, S: np.ndarray, BETA: np.ndarray,
             dates: pd.DatetimeIndex, tickers: list[str],
             long_only: bool = LONG_ONLY, hedge: bool = HEDGE) -> tuple[pd.Series, pd.DataFrame, dict]:
    T, N = R.shape
    hedge = hedge and not long_only
    slots_total = MAX_POSITIONS_PER_SIDE if long_only else 2 * MAX_POSITIONS_PER_SIDE
    slot = 1.0 / slots_total

    pos = np.zeros(N, dtype=int)         # -1, 0, +1
    beta_held = np.zeros(N)
    entry_day = np.full(N, -1)
    trade_pnl = np.zeros(N)              # running P&L of the open trade, in slot-notional units
    daily = np.zeros(T)
    turnover = np.zeros(T)
    trades = []

    for i in range(WINDOW, T - 1):
        s = S[i]
        old_pos = pos.copy()
        old_beta = beta_held.copy()

        # ---- exits
        for j in np.where(pos != 0)[0]:
            sj = s[j]
            too_old = (i - entry_day[j]) >= MAX_HOLD_DAYS
            if pos[j] == 1 and ((not np.isnan(sj) and sj > -EXIT_LONG_S) or too_old):
                trades.append(dict(ticker=tickers[j], side="LONG", entry=dates[entry_day[j]].date(),
                                   exit=dates[i].date(), days=i - entry_day[j],
                                   return_pct=round(trade_pnl[j] * 100, 3),
                                   reason="time" if too_old else "signal"))
                pos[j] = 0
            elif pos[j] == -1 and ((not np.isnan(sj) and sj < EXIT_SHORT_S) or too_old):
                trades.append(dict(ticker=tickers[j], side="SHORT", entry=dates[entry_day[j]].date(),
                                   exit=dates[i].date(), days=i - entry_day[j],
                                   return_pct=round(trade_pnl[j] * 100, 3),
                                   reason="time" if too_old else "signal"))
                pos[j] = 0

        # ---- entries, best |s| first, up to the free slots on each side
        free_long = MAX_POSITIONS_PER_SIDE - int((pos == 1).sum())
        cand_long = [j for j in np.where((pos == 0) & (s < -ENTRY_S))[0]]
        cand_long.sort(key=lambda j: s[j])
        for j in cand_long[:max(free_long, 0)]:
            pos[j], beta_held[j], entry_day[j], trade_pnl[j] = 1, BETA[i, j], i, 0.0

        if not long_only:
            free_short = MAX_POSITIONS_PER_SIDE - int((pos == -1).sum())
            cand_short = [j for j in np.where((pos == 0) & (s > ENTRY_S))[0]]
            cand_short.sort(key=lambda j: -s[j])
            for j in cand_short[:max(free_short, 0)]:
                pos[j], beta_held[j], entry_day[j], trade_pnl[j] = -1, BETA[i, j], i, 0.0

        # ---- costs on what changed (stock leg + hedge leg), charged into tomorrow
        d_stock = np.abs(pos - old_pos) * slot
        d_hedge = np.abs(pos * beta_held - old_pos * old_beta) * slot if hedge else 0.0
        turn = float(d_stock.sum() + np.sum(d_hedge))
        turnover[i + 1] = turn
        cost = turn * COST_PER_SIDE

        # ---- P&L over i -> i+1
        active = pos != 0
        if hedge:
            leg = pos * slot * (R[i + 1] - beta_held * E[i + 1])
            short_notional = slot * ((pos == -1).sum() + np.sum(np.where(pos == 1, beta_held, 0.0)))
        else:
            leg = pos * slot * R[i + 1]
            short_notional = slot * (pos == -1).sum()
        leg = np.where(active, leg, 0.0)
        borrow = short_notional * SHORT_BORROW_APR / 252
        daily[i + 1] = leg.sum() - cost - borrow
        trade_pnl[active] += leg[active] / slot

    ret = pd.Series(daily, index=dates)
    ret = ret.loc[dates[WINDOW + 1]:]
    info = dict(avg_gross_exposure=None, turnover_annual=round(float(turnover.sum()) / (len(ret) / 252), 2))
    return ret, pd.DataFrame(trades), info


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def metrics(r: pd.Series, bench: pd.Series, trades: pd.DataFrame) -> dict:
    eq = (1 + r).cumprod()
    years = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / years) - 1
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    b = bench.reindex(r.index).fillna(0)
    beta = np.cov(r, b)[0, 1] / b.var() if b.var() > 0 else np.nan
    corr = r.corr(b)
    wins = trades["return_pct"] > 0 if len(trades) else pd.Series(dtype=bool)
    return dict(
        cagr_pct=round(cagr * 100, 2),
        sharpe=round(sharpe, 2),
        max_drawdown_pct=round(dd * 100, 2),
        beta_to_spy=round(beta, 3),
        corr_to_spy=round(corr, 3),
        num_trades=len(trades),
        win_rate_pct=round(wins.mean() * 100, 1) if len(trades) else None,
        avg_trade_pct=round(trades["return_pct"].mean(), 3) if len(trades) else None,
        avg_days_held=round(trades["days"].mean(), 1) if len(trades) else None,
        pct_days_positive=round((r > 0).mean() * 100, 1),
        days=len(r),
    )


def monte_carlo_daily(r: pd.Series, n_sims: int = 3000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    draws = rng.choice(r.to_numpy(), size=(n_sims, len(r)), replace=True)
    paths = np.cumprod(1 + draws, axis=1)
    final = paths[:, -1]
    mdd = (paths / np.maximum.accumulate(paths, axis=1) - 1).min(axis=1)
    years = len(r) / 252
    p = lambda x, q: float(np.percentile(x, q))
    return dict(
        p5_cagr_pct=round((p(final, 5) ** (1 / years) - 1) * 100, 2),
        median_cagr_pct=round((p(final, 50) ** (1 / years) - 1) * 100, 2),
        p95_cagr_pct=round((p(final, 95) ** (1 / years) - 1) * 100, 2),
        median_max_dd_pct=round(p(mdd, 50) * 100, 2),
        worst5pct_max_dd_pct=round(p(mdd, 5) * 100, 2),
        prob_lose_money_pct=round(float(np.mean(final < 1)) * 100, 1),
    )


# ----------------------------------------------------------------------------
# Current signals readout
# ----------------------------------------------------------------------------

def signals_now(S: np.ndarray, HL: np.ndarray, BETA: np.ndarray, tickers: list[str],
                etfs: list[str], date, n: int = 10) -> None:
    s, hl, beta = S[-1], HL[-1], BETA[-1]
    df = pd.DataFrame({"ticker": tickers, "etf": etfs, "s": s, "half_life": hl, "beta": beta}).dropna()
    longs = df[df["s"] < -ENTRY_S].sort_values("s").head(n)
    shorts = df[df["s"] > ENTRY_S].sort_values("s", ascending=False).head(n)
    print(f"\n--- Signals as of {date.date()} (|s| > {ENTRY_S}, half-life <= {MAX_HALF_LIFE}d) ---")
    print("LONG candidates (residual stretched below its sector):")
    print(longs.round(2).to_string(index=False) if len(longs) else "  none")
    print("SHORT candidates (residual stretched above its sector):")
    print(shorts.round(2).to_string(index=False) if len(shorts) else "  none")
    print("Hedge: for each LONG, short `beta` dollars of its etf per dollar of stock (and vice versa).")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def run(mode: str = "full") -> None:
    members = get_sp500_with_sectors()
    etf_list = sorted(set(SECTOR_ETF.values()))
    print(f"[StatArb] {len(members)} S&P 500 members, downloading prices from {START} ...")
    px = load_prices(members["ticker"].tolist() + etf_list + [BENCHMARK], START)

    stocks = [t for t in members["ticker"] if t in px.columns]
    etf_of = dict(zip(members["ticker"], members["etf"]))
    etfs = [etf_of[t] for t in stocks]
    rets = px.pct_change().iloc[1:]
    R = rets[stocks].to_numpy()
    E = rets[etfs].to_numpy()
    dates = rets.index
    print(f"[StatArb] {len(stocks)} stocks x {len(dates)} days. Fitting rolling factor + OU models ...")

    S, BETA, HL = compute_signals(R, E)
    print(f"[StatArb] tradeable signals per day (avg): {np.isfinite(S).sum(1)[WINDOW:].mean():.0f}")

    if mode == "now":
        signals_now(S, HL, BETA, stocks, etfs, dates[-1])
        return

    bench = rets[BENCHMARK]
    results = {}
    for label, lo in (("long_short_hedged", False), ("long_only", True)):
        ret, trades, info = backtest(R, E, S, BETA, dates, stocks, long_only=lo)
        m = metrics(ret, bench, trades)
        m["turnover_annual"] = info["turnover_annual"]
        results[label] = (ret, trades, m)

    print(f"\n{'':>22}{'long_short':>14}{'long_only':>12}{'SPY':>10}")
    spy_m = metrics(bench.loc[results['long_short_hedged'][0].index], bench, pd.DataFrame())
    for k in results["long_short_hedged"][2]:
        a = results["long_short_hedged"][2][k]
        b = results["long_only"][2][k]
        c = spy_m.get(k, "")
        print(f"{k:>22}{str(a):>14}{str(b):>12}{str(c):>10}")

    for label, (ret, trades, m) in results.items():
        eq = ACCOUNT_SIZE * (1 + ret).cumprod()
        print(f"\n[{label}] ${ACCOUNT_SIZE:,.0f} -> ${eq.iloc[-1]:,.0f}  "
              f"({ret.index[0].date()} to {ret.index[-1].date()})")
        yearly = ((1 + ret).groupby(ret.index.year).prod() - 1) * 100
        print("  by year: " + ", ".join(f"{y}: {v:+.1f}%" for y, v in yearly.round(1).items()))
        mc = monte_carlo_daily(ret)
        print("  monte carlo: " + ", ".join(f"{k}={v}" for k, v in mc.items()))

    signals_now(S, HL, BETA, stocks, etfs, dates[-1])

    stamp = datetime.now().strftime("%Y%m%d")
    results["long_short_hedged"][1].to_csv(f"statarb_trades_longshort_{stamp}.csv", index=False)
    results["long_only"][1].to_csv(f"statarb_trades_longonly_{stamp}.csv", index=False)
    pd.DataFrame({k: v[0] for k, v in results.items()}).to_csv(f"statarb_daily_returns_{stamp}.csv")
    print(f"\n[StatArb] trade logs and daily returns saved (statarb_*_{stamp}.csv)")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "full")
