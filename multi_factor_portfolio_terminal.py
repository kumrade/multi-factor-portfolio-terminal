#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 Multi-Factor Portfolio Validation Terminal
=============================================================================
 multi_factor_portfolio_terminal.py

 A single-file quantitative terminal that estimates *forward* expected returns
 for NSE equities using a Fama-French / Carhart style four-factor model
 (Market, Size, Value, Momentum) and feeds those expected returns -- together
 with a covariance matrix -- into a Markowitz mean-variance optimiser.

 It answers the question:
   "What return should I expect from each stock using market factors,
    instead of only trusting noisy historical averages?"

 Pipeline
 --------
   NSE prices + NIFTY 50  ->  daily returns
        -> build factors : MKT (excess mkt), SMB (size), HML (value), WML (mom)
        -> OLS regression per stock (alpha, betas, t-stats, p, R^2)
        -> forecast factor returns (mean / EWMA)
        -> forecast stock expected returns (Multi-Factor / CAPM / Historical)
        -> covariance matrix (Sample / EWMA / Factor)
        -> mean-variance optimiser (Max Sharpe / Min Var / Target / Frontier)

 Data
 ----
   * Prices are loaded from wide CSV files or generated in reproducible demo
     mode. Fundamentals are user-supplied and point-in-time dated.

 Run
 ---
   python multi_factor_portfolio_terminal.py --demo
   python multi_factor_portfolio_terminal.py --test
   python multi_factor_portfolio_terminal.py --walk-forward

 Conventions
 -----------
   * Regressions use *excess* daily returns on the LHS. The market factor is
     NIFTY excess return; SMB/HML/WML are self-financing long-short spreads.
   * Expected total return = risk-free + alpha + beta . E[factors].
   * Daily quantities are annualised x252 (arithmetic) for the optimiser.

 Educational quantitative research software.
=============================================================================
"""

import sys
import math
import argparse
import warnings
import threading
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
#  Matplotlib is imported with the Figure API only (no pyplot) so it embeds
#  cleanly in Tk and never needs a display for --test.
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")  # overridden to TkAgg canvas only when GUI is built
from matplotlib.figure import Figure

from scipy.optimize import minimize, linprog
from scipy import stats as sps

TRADING_DAYS = 252

# =============================================================================
#  CONFIG  /  THEME
# =============================================================================

APP_NAME = "Multi-Factor Portfolio Validation Terminal"
VERSION = "2.0"

# Dark theme with a teal accent.
CLR = {
    "bg":        "#0f1115",
    "panel":     "#161a21",
    "panel2":    "#1c2129",
    "grid":      "#232a34",
    "fg":        "#e6e9ef",
    "muted":     "#8b93a3",
    "accent":    "#2dd4bf",   # teal
    "accent2":   "#38bdf8",   # sky
    "good":      "#4ade80",
    "bad":       "#f87171",
    "warn":      "#fbbf24",
    "entry":     "#0b0d11",
    "sel":       "#0e3a37",
}

# Default universe (the 10 liquid NSE names from the spec) --------------------
DEFAULT_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN",
    "ICICIBANK", "LT", "ITC", "AXISBANK", "BHARTIARTL",
]

# Approximate, user-editable fundamentals snapshot (market cap in INR crore).
# These are seed values only -- the Size/Value factors are only as good as the
# numbers the user maintains here. Edit in the Data Loader tab or import a CSV.
SEED_FUNDAMENTALS = {
    #  symbol      mktcap_cr    pb     pe
    "RELIANCE":  (1900000.0,  2.2,  27.0),
    "TCS":       (1400000.0, 14.0,  30.0),
    "HDFCBANK":  (1300000.0,  2.8,  19.0),
    "INFY":      ( 650000.0,  8.0,  25.0),
    "SBIN":      ( 720000.0,  1.6,  10.0),
    "ICICIBANK": ( 880000.0,  3.2,  18.0),
    "LT":        ( 500000.0,  5.5,  35.0),
    "ITC":       ( 560000.0,  7.5,  27.0),
    "AXISBANK":  ( 350000.0,  2.1,  14.0),
    "BHARTIARTL":( 900000.0,  8.5,  60.0),
}

DEFAULT_RF_ANNUAL = 0.065   # ~6.5% India risk-free (editable)


# =============================================================================
#  LOCAL DATA LAYER
# =============================================================================

def load_price_csv(path):
    """Load Date, benchmark and stock closing prices from a wide CSV file."""
    frame = pd.read_csv(path)
    if "Date" not in frame.columns or "NIFTY50" not in frame.columns:
        raise ValueError("CSV requires Date and NIFTY50 columns.")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).set_index("Date").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    benchmark = frame.pop("NIFTY50").dropna().rename("NIFTY50")
    prices = frame.dropna(how="all")
    validate_price_data(prices, benchmark)
    return prices, benchmark


def validate_price_data(prices, benchmark, min_observations=100):
    if prices.columns.duplicated().any():
        raise ValueError("Duplicate stock symbols are not allowed.")
    if len(prices) < min_observations:
        raise ValueError(f"At least {min_observations} price rows are required.")
    if prices.shape[1] < 3:
        raise ValueError("At least three stocks are required for factor sorts.")
    if (prices <= 0).any().any() or (benchmark <= 0).any():
        raise ValueError("Prices must be positive.")
    if prices.isna().mean().max() > 0.10:
        raise ValueError("A stock has more than 10% missing prices.")


# =============================================================================
#  DEMO DATA  (synthetic, factor-structured, internally consistent)
# =============================================================================

def generate_demo_data(symbols, n_days=520, seed=7):
    """Build synthetic prices + NIFTY + fundamentals with genuine factor
    structure, so the *real* factor-construction pipeline is exercised."""
    rng = np.random.default_rng(seed)
    N = len(symbols)
    dates = pd.bdate_range(end=datetime.today().date(), periods=n_days)
    n_days = len(dates)  # bdate_range can drop one when end falls on a weekend

    # latent daily factor premia (per day)
    mkt = rng.normal(0.0004, 0.011, n_days)     # market excess
    smb = rng.normal(0.0001, 0.006, n_days)     # size
    hml = rng.normal(0.00008, 0.005, n_days)    # value
    wml = rng.normal(0.00015, 0.007, n_days)    # momentum

    rf_daily = DEFAULT_RF_ANNUAL / TRADING_DAYS

    # assign each stock latent characteristic loadings
    size_load = rng.uniform(-1, 1, N)   # + => small-cap tilt
    value_load = rng.uniform(-1, 1, N)  # + => value (cheap) tilt
    mom_load = rng.uniform(-0.6, 1, N)

    prices = {}
    fundamentals = {}
    for j, sym in enumerate(symbols):
        b_mkt = rng.uniform(0.7, 1.4)
        b_smb = 0.9 * size_load[j] + rng.normal(0, 0.2)
        b_hml = 0.9 * value_load[j] + rng.normal(0, 0.2)
        b_wml = 0.7 * mom_load[j] + rng.normal(0, 0.2)
        alpha = rng.normal(0.0, 0.00015)
        idio = rng.normal(0, rng.uniform(0.008, 0.016), n_days)
        excess = (alpha + b_mkt * mkt + b_smb * smb
                  + b_hml * hml + b_wml * wml + idio)
        ret = rf_daily + excess
        px = 100.0 * np.cumprod(1.0 + ret)
        prices[sym] = pd.Series(px, index=dates, name=sym)

        # fundamentals consistent with latent characteristics:
        #   small-cap tilt  -> lower market cap
        #   value tilt      -> lower P/B and P/E
        base_cap = 800000.0 * math.exp(-0.9 * size_load[j])
        pb = float(np.clip(4.0 * math.exp(-0.6 * value_load[j]) + rng.normal(0, 0.3), 0.4, 20))
        pe = float(np.clip(22.0 * math.exp(-0.4 * value_load[j]) + rng.normal(0, 2), 5, 80))
        fundamentals[sym] = (round(base_cap, 0), round(pb, 2), round(pe, 2))

    price_df = pd.DataFrame(prices)
    nifty = pd.Series(100.0 * np.cumprod(1.0 + rf_daily + mkt),
                      index=dates, name="NIFTY50")
    fdf = pd.DataFrame(fundamentals, index=["mktcap_cr", "pb", "pe"]).T
    return price_df, nifty, fdf


# =============================================================================
#  QUANT CORE
# =============================================================================

def prices_to_returns(price_df):
    """Simple daily returns, dropping the first NaN row."""
    out = price_df.ffill(limit=2).pct_change(fill_method=None).dropna(how="all")
    return out.replace([np.inf, -np.inf], np.nan)


def validate_fundamentals(fundamentals, symbols):
    required = {"mktcap_cr", "pb", "pe"}
    if not required.issubset(fundamentals.columns):
        raise ValueError(f"Fundamentals require columns: {sorted(required)}")
    f = fundamentals.reindex(symbols).astype(float)
    if f.isna().any().any():
        raise ValueError("Fundamentals are missing for one or more stocks.")
    if (f[list(required)] <= 0).any().any():
        raise ValueError("Market cap, P/B and P/E must all be positive.")
    return f


def _bucket_masks(char_series, low_q=0.30, high_q=0.30):
    """Return (low_names, high_names): bottom low_q and top high_q of a
    characteristic. With small universes falls back to ~1/3 splits."""
    s = char_series.dropna().sort_values()
    n = len(s)
    if n < 3:
        return list(s.index), list(s.index)
    k_low = max(1, int(round(n * low_q)))
    k_high = max(1, int(round(n * high_q)))
    low = list(s.index[:k_low])
    high = list(s.index[-k_high:])
    return low, high


def build_market_factor(nifty_ret, rf_daily):
    return (nifty_ret - rf_daily).rename("MKT")


def build_size_factor(stock_ret, mktcap):
    """SMB = small-cap avg return - large-cap avg return (static cap sort)."""
    small, big = _bucket_masks(mktcap)            # small = low cap, big = high cap
    small = [s for s in small if s in stock_ret.columns]
    big = [s for s in big if s in stock_ret.columns]
    smb = stock_ret[small].mean(axis=1) - stock_ret[big].mean(axis=1)
    return smb.rename("SMB"), small, big


def build_value_factor(stock_ret, pb, pe):
    """HML from a composite value score using book and earnings yields.

    Percentile ranks keep P/B and P/E on comparable scales. A high score means
    high book-to-price and high earnings yield, therefore a cheaper stock.
    """
    book_yield = 1.0 / pb.astype(float)
    earnings_yield = 1.0 / pe.astype(float)
    value_score = 0.5 * book_yield.rank(pct=True) + 0.5 * earnings_yield.rank(pct=True)
    expensive, cheap = _bucket_masks(value_score)
    cheap = [s for s in cheap if s in stock_ret.columns]
    expensive = [s for s in expensive if s in stock_ret.columns]
    hml = stock_ret[cheap].mean(axis=1) - stock_ret[expensive].mean(axis=1)
    return hml.rename("HML"), cheap, expensive, value_score


def build_momentum_factor(stock_ret, formation=126, skip=21, rebalance=21):
    """WML = winners - losers, with a rolling formation window and periodic
    rebalancing. Winners/losers chosen by cumulative return over
    [t-formation-skip, t-skip). Returns a daily WML series (NaN until warm)."""
    ret = stock_ret.copy()
    dates = ret.index
    n = len(dates)
    wml = pd.Series(index=dates, dtype=float, name="WML")

    warm = formation + skip
    if n <= warm + 5:  # not enough history -> degrade gracefully
        skip = 0
        formation = max(20, n // 3)
        warm = formation + skip

    cur_win, cur_los = [], []
    for t in range(n):
        if t >= warm and (t - warm) % rebalance == 0:
            lo = t - formation - skip
            hi = t - skip
            window = ret.iloc[lo:hi]
            cum = (1.0 + window).prod() - 1.0
            cum = cum.dropna()
            if len(cum) >= 3:
                srt = cum.sort_values()
                k = max(1, int(round(len(srt) * 0.30)))
                cur_los = list(srt.index[:k])
                cur_win = list(srt.index[-k:])
        if cur_win and cur_los and t >= warm:
            wml.iloc[t] = (ret.iloc[t][cur_win].mean()
                           - ret.iloc[t][cur_los].mean())
    return wml


def build_factor_table(stock_ret, nifty_ret, fundamentals, rf_daily,
                        mom_formation=126, mom_skip=21):
    """Assemble the aligned MKT/SMB/HML/WML daily factor table."""
    mktcap = fundamentals["mktcap_cr"]
    pb = fundamentals["pb"]
    pe = fundamentals["pe"]

    mkt = build_market_factor(nifty_ret, rf_daily)
    smb, small, big = build_size_factor(stock_ret, mktcap)
    hml, cheap, exp, value_score = build_value_factor(stock_ret, pb, pe)
    wml = build_momentum_factor(stock_ret, mom_formation, mom_skip)

    factors = pd.concat([mkt, smb, hml, wml], axis=1).dropna()
    meta = {
        "size_small": small, "size_big": big,
        "value_cheap": cheap, "value_expensive": exp,
        "value_score": value_score,
    }
    return factors, meta


def run_factor_regression(stock_excess, factors):
    """OLS of one stock's excess return on the four factors.
    Returns dict with alpha, betas, t-stats, p-values, R^2, adj-R^2, resid var."""
    df = pd.concat([stock_excess.rename("y"), factors], axis=1).dropna()
    y = df["y"].values
    Xf = df[factors.columns].values
    n = len(y)
    X = np.column_stack([np.ones(n), Xf])
    k = X.shape[1]
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    covb = sigma2 * XtX_inv
    se = np.sqrt(np.clip(np.diag(covb), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        tvals = np.where(se > 0, beta / se, 0.0)
    pvals = 2.0 * sps.t.sf(np.abs(tvals), dof)
    sst = float(np.sum((y - y.mean()) ** 2))
    ssr = float(resid @ resid)
    r2 = 1.0 - ssr / sst if sst > 0 else 0.0
    adj = 1.0 - (1.0 - r2) * (n - 1) / dof if dof > 0 else r2

    names = ["alpha"] + list(factors.columns)
    return {
        "n": n,
        "coef": dict(zip(names, beta)),
        "t": dict(zip(names, tvals)),
        "p": dict(zip(names, pvals)),
        "r2": r2, "adj_r2": adj,
        "resid_var": sigma2,   # per-day idiosyncratic variance
    }


def regress_all(stock_ret, factors, rf_daily):
    """Run the regression for every stock. Returns dict[symbol] -> result."""
    results = {}
    for sym in stock_ret.columns:
        excess = stock_ret[sym] - rf_daily
        try:
            results[sym] = run_factor_regression(excess, factors)
        except Exception as e:
            results[sym] = {"error": str(e)}
    return results


def forecast_factor_returns(factors, method="mean", ewma_lambda=0.97,
                            shrinkage=0.50, target=None):
    """Forecast the *daily* expected return of each factor."""
    if method == "ewma":
        w = np.array([(1 - ewma_lambda) * ewma_lambda ** i
                      for i in range(len(factors))][::-1])
        w /= w.sum()
        fc = (factors.values * w[:, None]).sum(axis=0)
        estimate = pd.Series(fc, index=factors.columns)
    else:
        estimate = factors.mean()
    if method == "zero":
        estimate = pd.Series(0.0, index=factors.columns)
    elif method == "shrink":
        target = (pd.Series(0.0, index=factors.columns) if target is None
                  else pd.Series(target).reindex(factors.columns).fillna(0.0))
        estimate = (1.0 - shrinkage) * estimate + shrinkage * target
    return estimate


def expected_returns(reg_results, factor_forecast, stock_ret, rf_daily,
                     method="factor", annualize=True, stock_shrinkage=0.0,
                     shrink_target=None, include_alpha=False,
                     annual_bounds=(-0.50, 0.75)):
    """Expected *total* return per stock.
       method: 'factor' | 'capm' | 'historical'."""
    mu = {}
    fc = factor_forecast
    for sym, r in reg_results.items():
        if "error" in r:
            mu[sym] = np.nan
            continue
        if method == "historical":
            daily = stock_ret[sym].mean()
        elif method == "capm":
            beta_mkt = r["coef"].get("MKT", 0.0)
            daily = rf_daily + beta_mkt * fc.get("MKT", 0.0)
        else:  # multi-factor
            # Alpha is an in-sample residual intercept and is not assumed to
            # persist unless the researcher explicitly enables it.
            excess = r["coef"]["alpha"] if include_alpha else 0.0
            for f in factor_forecast.index:
                excess += r["coef"].get(f, 0.0) * fc[f]
            daily = rf_daily + excess
        mu[sym] = daily * TRADING_DAYS if annualize else daily
    result = pd.Series(mu, dtype=float)
    if stock_shrinkage > 0:
        target = (float(result.median()) if shrink_target is None
                  else float(shrink_target))
        result = (1.0 - stock_shrinkage) * result + stock_shrinkage * target
    if annualize and annual_bounds is not None:
        result = result.clip(*annual_bounds)
    return result


def covariance_matrix(stock_ret, reg_results=None, factors=None,
                      method="sample", ewma_lambda=0.94, annualize=True):
    """Return an annualised covariance matrix (DataFrame).
       method: 'sample' | 'ewma' | 'factor'."""
    R = stock_ret.dropna()
    syms = list(R.columns)
    scale = TRADING_DAYS if annualize else 1.0

    if method == "ewma":
        X = (R - R.mean()).values
        T = X.shape[0]
        w = np.array([(1 - ewma_lambda) * ewma_lambda ** i
                      for i in range(T)][::-1])
        w /= w.sum()
        Xw = X * np.sqrt(w)[:, None]
        cov = Xw.T @ Xw
    elif method == "factor" and reg_results is not None and factors is not None:
        fnames = list(factors.columns)
        B = np.array([[reg_results[s]["coef"].get(f, 0.0) for f in fnames]
                      for s in syms])
        Sig_f = np.cov(factors.values, rowvar=False)
        D = np.diag([reg_results[s].get("resid_var", 0.0) for s in syms])
        cov = B @ Sig_f @ B.T + D
    else:  # sample
        cov = np.cov(R.values, rowvar=False)

    cov = nearest_psd(cov * scale)
    return pd.DataFrame(cov, index=syms, columns=syms)


def nearest_psd(matrix, floor=1e-8):
    """Repair a symmetric matrix by flooring its eigenvalues."""
    matrix = 0.5 * (np.asarray(matrix, float) + np.asarray(matrix, float).T)
    values, vectors = np.linalg.eigh(matrix)
    repaired = (vectors * np.maximum(values, floor)) @ vectors.T
    return 0.5 * (repaired + repaired.T)


def covariance_diagnostics(covariance):
    values = np.linalg.eigvalsh(np.asarray(covariance, float))
    positive = values[values > 1e-14]
    condition = float(values.max() / positive.min()) if len(positive) else np.inf
    return {
        "min_eigenvalue": float(values.min()),
        "max_eigenvalue": float(values.max()),
        "condition_number": condition,
        "is_psd": bool(values.min() >= -1e-10),
    }


# =============================================================================
#  MEAN-VARIANCE OPTIMISER
# =============================================================================

def _port_stats(w, mu, cov, rf_annual):
    r = float(w @ mu)
    v = float(np.sqrt(max(w @ cov @ w, 1e-18)))
    sharpe = (r - rf_annual) / v if v > 0 else 0.0
    return r, v, sharpe


def optimise_portfolio(mu, cov, rf_annual, objective="sharpe",
                       target_return=None, allow_short=False, w_max=0.40,
                       w_min=0.0, gross_limit=1.0, previous_weights=None,
                       turnover_limit=None, transaction_cost_bps=0.0,
                       sector_map=None, sector_limits=None, starts=30):
    """Robust constrained optimiser with turnover and exposure controls."""
    mu = np.asarray(mu, dtype=float)
    cov = nearest_psd(cov)
    n = len(mu)
    if not np.isfinite(mu).all() or not np.isfinite(cov).all():
        raise ValueError("Expected returns and covariance must be finite.")
    if not allow_short and w_min * n > 1.0 + 1e-10:
        raise ValueError("Minimum weights exceed 100% in total.")
    if w_max * n < 1.0 - 1e-10:
        raise ValueError("Maximum weights cannot produce a fully invested portfolio.")
    if gross_limit < 1.0:
        raise ValueError("Gross-exposure limit must be at least 1.0.")
    w0 = np.repeat(1.0 / n, n)
    lb = -w_max if allow_short else w_min
    bounds = [(lb, w_max)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    cons.append({"type": "ineq", "fun": lambda w: gross_limit - np.abs(w).sum()})
    previous = (np.asarray(previous_weights, float) if previous_weights is not None
                else None)
    if previous is not None and len(previous) != n:
        raise ValueError("previous_weights length does not match the universe.")
    if previous is not None and turnover_limit is not None:
        cons.append({"type": "ineq",
                     "fun": lambda w: turnover_limit - 0.5 * np.abs(w - previous).sum()})
    if sector_map and sector_limits:
        sector_map = np.asarray(sector_map)
        for sector, limit in sector_limits.items():
            mask = sector_map == sector
            cons.append({"type": "ineq",
                         "fun": lambda w, m=mask, cap=float(limit): cap - w[m].sum()})

    if objective == "sharpe":
        def neg(w):
            r, v, s = _port_stats(w, mu, cov, rf_annual)
            cost = (transaction_cost_bps / 10000.0 * np.abs(w - previous).sum()
                    if previous is not None else 0.0)
            return -(r - cost - rf_annual) / v
        obj = neg
    elif objective == "minvar":
        obj = lambda w: float(w @ cov @ w)
    elif objective == "target":
        obj = lambda w: float(w @ cov @ w)
        cons.append({"type": "eq", "fun": lambda w: float(w @ mu) - target_return})
    else:
        raise ValueError(objective)

    rng = np.random.default_rng(42)
    candidates = [w0]
    if not allow_short:
        candidates.extend(rng.dirichlet(np.ones(n), size=starts))
    best = None
    for start in candidates:
        if any(start[i] < lb - 1e-10 or start[i] > w_max + 1e-10 for i in range(n)):
            continue
        res = minimize(obj, start, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1500, "ftol": 1e-12})
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise ValueError("No feasible optimum found under the selected constraints.")
    w = best.x
    w = np.where(np.abs(w) < 1e-6, 0.0, w)
    if w.sum() != 0:
        w = w / w.sum()
    r, v, s = _port_stats(w, mu, cov, rf_annual)
    turnover = (0.5 * float(np.abs(w - previous).sum()) if previous is not None
                else np.nan)
    cost = (transaction_cost_bps / 10000.0 * 2.0 * turnover
            if np.isfinite(turnover) else 0.0)
    return {"weights": w, "ret": r, "net_ret": r - cost, "vol": v,
            "sharpe": s, "turnover": turnover, "cost": cost,
            "concentration": float(w @ w), "gross_exposure": float(np.abs(w).sum()),
            "ok": True, "message": best.message}


def efficient_frontier(mu, cov, rf_annual, n_points=40,
                       allow_short=False, w_max=0.40, w_min=0.0,
                       gross_limit=1.0):
    mu = np.asarray(mu, float)
    minimum = optimise_portfolio(mu, cov, rf_annual, "minvar",
                                 allow_short=allow_short, w_max=w_max,
                                 w_min=w_min, gross_limit=gross_limit)
    bounds = [(-w_max if allow_short else w_min, w_max)] * len(mu)
    maximum = linprog(-mu, A_eq=np.ones((1, len(mu))), b_eq=[1.0],
                      bounds=bounds, method="highs")
    if not maximum.success:
        raise ValueError("Unable to determine the feasible maximum return.")
    targets = np.linspace(minimum["ret"], float(-maximum.fun), n_points)
    pts = []
    for t in targets:
        try:
            r = optimise_portfolio(mu, cov, rf_annual, "target", t,
                                   allow_short, w_max, w_min, gross_limit)
            if r["ok"]:
                pts.append((r["vol"], r["ret"]))
        except Exception:
            pass
    return np.array(pts) if pts else np.empty((0, 2))


def portfolio_risk_metrics(returns, weights, confidence=0.95):
    """Historical VaR/CVaR, drawdown and realised annualised statistics."""
    series = pd.Series(np.asarray(returns) @ np.asarray(weights),
                       index=getattr(returns, "index", None)).dropna()
    if len(series) < 20:
        raise ValueError("At least 20 portfolio returns are required.")
    wealth = (1.0 + series).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    q = float(series.quantile(1.0 - confidence))
    cvar = float(series[series <= q].mean())
    ann_return = float(series.mean() * TRADING_DAYS)
    ann_vol = float(series.std(ddof=1) * np.sqrt(TRADING_DAYS))
    return {"realised_return": ann_return, "realised_volatility": ann_vol,
            "var": -q, "cvar": -cvar, "max_drawdown": float(drawdown.min())}


def stress_test(weights, covariance, shocks=None):
    """Apply transparent one-period return shocks to a portfolio."""
    shocks = shocks or {"broad_selloff": -0.10, "mild_correction": -0.05}
    volatility = float(np.sqrt(np.asarray(weights) @ np.asarray(covariance)
                               @ np.asarray(weights)))
    return {name: float(level * np.abs(weights).sum()) for name, level in shocks.items()} | {
        "one_sigma_annual": -volatility,
        "two_sigma_annual": -2.0 * volatility,
    }


def risk_contributions(weights, covariance):
    weights = np.asarray(weights, float)
    covariance = np.asarray(covariance, float)
    variance = float(weights @ covariance @ weights)
    if variance <= 1e-18:
        return np.zeros_like(weights)
    component = weights * (covariance @ weights) / variance
    return component


def factor_attribution(weights, symbols, regressions, factor_forecast):
    """Portfolio factor exposures and forecast return contributions."""
    factors = list(factor_forecast.index)
    beta = pd.DataFrame(
        {symbol: {factor: regressions[symbol]["coef"].get(factor, 0.0)
                  for factor in factors}
         for symbol in symbols}
    ).T
    exposure = beta.T @ pd.Series(weights, index=symbols)
    contribution = exposure * factor_forecast * TRADING_DAYS
    return pd.DataFrame({"Exposure": exposure,
                         "Expected Return Contribution": contribution})


def regime_analysis(benchmark_returns, lookback=63):
    """Simple observable trend/volatility regimes without look-ahead."""
    r = benchmark_returns.dropna()
    momentum = (1.0 + r).rolling(lookback).apply(np.prod, raw=True) - 1.0
    volatility = r.rolling(lookback).std() * np.sqrt(TRADING_DAYS)
    vol_cutoff = volatility.expanding(min_periods=lookback).median()
    regime = pd.Series(index=r.index, dtype="object")
    regime[(momentum >= 0) & (volatility <= vol_cutoff)] = "Risk On"
    regime[(momentum >= 0) & (volatility > vol_cutoff)] = "Bull High Vol"
    regime[(momentum < 0) & (volatility <= vol_cutoff)] = "Defensive"
    regime[(momentum < 0) & (volatility > vol_cutoff)] = "Risk Off"
    return pd.DataFrame({"Momentum": momentum, "Volatility": volatility,
                         "Regime": regime}).dropna()


def benchmark_statistics(stock_returns, benchmark_returns, weights):
    portfolio = stock_returns @ np.asarray(weights)
    benchmark = benchmark_returns.reindex(portfolio.index).dropna()
    portfolio = portfolio.reindex(benchmark.index)
    equal = stock_returns.reindex(benchmark.index).mean(axis=1)
    market = pd.DataFrame({"Portfolio": portfolio, "NIFTY50": benchmark,
                           "Equal Weight": equal})
    return pd.DataFrame({
        "Annual Return": market.mean() * TRADING_DAYS,
        "Annual Volatility": market.std() * np.sqrt(TRADING_DAYS),
        "Maximum Drawdown": market.apply(
            lambda s: ((1 + s).cumprod() / (1 + s).cumprod().cummax() - 1).min()),
    })


def walk_forward_backtest(prices, benchmark, fundamentals, rf_annual,
                          train_days=252, rebalance_days=21,
                          ret_method="factor", cov_method="factor",
                          forecast_method="shrink", factor_shrinkage=0.50,
                          stock_shrinkage=0.35, w_max=0.40,
                          transaction_cost_bps=10.0):
    """Expanding-window, out-of-sample portfolio simulation.

    Every rebalance uses only information available through the preceding day.
    The chosen weights are then held through the next out-of-sample block.
    """
    validate_price_data(prices, benchmark, min_observations=train_days + 30)
    fundamentals = validate_fundamentals(fundamentals, prices.columns)
    all_returns = prices.pct_change(fill_method=None).dropna(how="any")
    benchmark_returns = benchmark.pct_change(fill_method=None).reindex(all_returns.index)
    portfolio = pd.Series(index=all_returns.index, dtype=float, name="Portfolio")
    equal = all_returns.mean(axis=1).rename("Equal Weight")
    weights_history, costs = [], []
    previous = np.repeat(1.0 / prices.shape[1], prices.shape[1])

    for start in range(train_days, len(all_returns), rebalance_days):
        stop = min(start + rebalance_days, len(all_returns))
        train_index = all_returns.index[:start]
        train_prices = prices.loc[:train_index[-1]]
        train_benchmark = benchmark.loc[:train_index[-1]]
        engine = FactorEngine(rf_annual=rf_annual, log=lambda *_: None)
        engine.set_data(train_prices, train_benchmark, fundamentals)
        engine.run_all(ret_method=ret_method, cov_method=cov_method,
                       fc_method=forecast_method,
                       factor_shrinkage=factor_shrinkage,
                       stock_shrinkage=stock_shrinkage)
        mu = engine.mu.reindex(prices.columns)
        cov = engine.cov.reindex(index=prices.columns, columns=prices.columns)
        result = optimise_portfolio(
            mu.values, cov.values, rf_annual, objective="sharpe",
            w_max=w_max, previous_weights=previous,
            transaction_cost_bps=transaction_cost_bps,
        )
        block_index = all_returns.index[start:stop]
        block = all_returns.loc[block_index] @ result["weights"]
        trading_cost = result["cost"]
        if len(block):
            block.iloc[0] -= trading_cost
        portfolio.loc[block_index] = block
        weights_history.append(pd.Series(result["weights"], index=prices.columns,
                                         name=block_index[0]))
        costs.append(trading_cost)
        previous = result["weights"]

    output = pd.concat([portfolio, benchmark_returns.rename("NIFTY50"), equal], axis=1)
    output = output.loc[portfolio.first_valid_index():].dropna()
    statistics = pd.DataFrame({
        name: portfolio_risk_metrics(output[[name]].values, [1.0])
        for name in output.columns
    }).T
    return {
        "returns": output,
        "wealth": (1.0 + output).cumprod(),
        "statistics": statistics,
        "weights": pd.DataFrame(weights_history),
        "total_cost": float(np.sum(costs)),
    }


def monte_carlo_weights(k, n=4000, allow_short=False, w_max=1.0, seed=1):
    """Generate feasible random portfolio weights for plotting.

    Important: when short selling is allowed, do NOT draw normal weights and
    divide by their sum. If the sum is close to zero, that creates artificial
    leverage and produces impossible 10,000%+ volatility/return points on the
    efficient-frontier chart.

    This generator respects the same constraints as the optimiser:
        sum(weights) = 1
        long-only:     0 <= w_i <= w_max
        long-short: -w_max <= w_i <= w_max
    """
    rng = np.random.default_rng(seed)

    if not allow_short:
        if w_max * k < 1.0 - 1e-10:
            raise ValueError("w_max is infeasible for this universe size.")
        accepted = []
        attempts = 0
        while len(accepted) < n and attempts < n * 500:
            batch = rng.dirichlet(np.ones(k), size=min(2000, n * 2))
            accepted.extend(batch[batch.max(axis=1) <= w_max + 1e-12])
            attempts += len(batch)
        if len(accepted) < n:
            accepted.extend([np.repeat(1.0 / k, k)] * (n - len(accepted)))
        return np.asarray(accepted[:n])

    lb, ub = -float(w_max), float(w_max)
    weights = []
    attempts = 0
    max_attempts = max(20000, n * 200)

    # Rejection sampler: choose k-1 weights, set the final one so sum = 1.
    # Accept only if the final weight also stays within bounds.
    while len(weights) < n and attempts < max_attempts:
        attempts += 1
        w = rng.uniform(lb, ub, size=k)
        w[-1] = 1.0 - np.sum(w[:-1])
        if lb <= w[-1] <= ub:
            weights.append(w)

    if len(weights) < n:
        # Conservative fallback: include feasible long-only portfolios rather
        # than plotting explosive leveraged points.
        extra = rng.dirichlet(np.ones(k), size=n - len(weights))
        weights.extend(extra)

    return np.asarray(weights[:n])


def monte_carlo_cloud(mu, cov, rf_annual, n=4000, allow_short=False, seed=1, w_max=1.0):
    k = len(mu)
    W = monte_carlo_weights(k, n=n, allow_short=allow_short, w_max=w_max, seed=seed)
    out = np.empty((len(W), 3))
    for i, w in enumerate(W):
        out[i] = _port_stats(w, mu, cov, rf_annual)
    return out  # columns: ret, vol, sharpe


# =============================================================================
#  ANALYSIS ORCHESTRATION  (data -> everything; used by GUI and self-test)
# =============================================================================

class FactorEngine:
    """Holds state and runs the full pipeline. Pure logic, no GUI."""

    def __init__(self, rf_annual=DEFAULT_RF_ANNUAL, log=print):
        self.rf_annual = rf_annual
        self.rf_daily = rf_annual / TRADING_DAYS
        self.log = log
        # data
        self.prices = None
        self.nifty = None
        self.fundamentals = None
        # derived
        self.stock_ret = None
        self.nifty_ret = None
        self.factors = None
        self.meta = None
        self.reg = None
        self.factor_forecast = None
        self.mu = None
        self.cov = None

    def set_data(self, prices, nifty, fundamentals):
        validate_price_data(prices, nifty)
        self.prices = prices.sort_index().astype(float)
        self.nifty = nifty.sort_index().astype(float)
        # align fundamentals to available price columns
        self.fundamentals = validate_fundamentals(fundamentals,
                                                  self.prices.columns)

    def compute_returns(self):
        self.stock_ret = prices_to_returns(self.prices)
        self.nifty_ret = self.nifty.pct_change().reindex(self.stock_ret.index).dropna()
        # align both
        common = self.stock_ret.index.intersection(self.nifty_ret.index)
        self.stock_ret = self.stock_ret.loc[common].dropna(axis=1, how="all")
        self.nifty_ret = self.nifty_ret.loc[common]
        return self.stock_ret, self.nifty_ret

    def build_factors(self, mom_formation=126, mom_skip=21):
        self.factors, self.meta = build_factor_table(
            self.stock_ret, self.nifty_ret, self.fundamentals,
            self.rf_daily, mom_formation, mom_skip)
        return self.factors

    def regress(self):
        self.reg = regress_all(self.stock_ret, self.factors, self.rf_daily)
        return self.reg

    def forecast(self, method="mean", ewma_lambda=0.97, shrinkage=0.50):
        self.factor_forecast = forecast_factor_returns(
            self.factors, method, ewma_lambda, shrinkage)
        return self.factor_forecast

    def expected(self, method="factor", stock_shrinkage=0.0):
        if method == "shrunk":
            method, stock_shrinkage = "factor", max(stock_shrinkage, 0.35)
        self.mu = expected_returns(self.reg, self.factor_forecast,
                                   self.stock_ret, self.rf_daily, method,
                                   stock_shrinkage=stock_shrinkage)
        return self.mu

    def covariance(self, method="sample", ewma_lambda=0.94):
        self.cov = covariance_matrix(self.stock_ret, self.reg, self.factors,
                                     method, ewma_lambda)
        return self.cov

    def run_all(self, ret_method="factor", cov_method="factor",
                fc_method="shrink", mom_formation=126, mom_skip=21,
                factor_shrinkage=0.50, stock_shrinkage=0.35):
        self.compute_returns()
        self.build_factors(mom_formation, mom_skip)
        self.regress()
        self.forecast(fc_method, shrinkage=factor_shrinkage)
        self.expected(ret_method, stock_shrinkage)
        self.covariance(cov_method)
        return self


# =============================================================================
#  SELF-TEST  (headless, no network, no GUI)
# =============================================================================

def self_test():
    print("=" * 70)
    print(" Multi-Factor Portfolio Terminal -- SELF TEST")
    print("=" * 70)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
        return cond

    # 1. demo data
    prices, nifty, fdf = generate_demo_data(DEFAULT_SYMBOLS, n_days=520)
    check("demo prices shape", prices.shape[0] > 400 and prices.shape[1] == 10,
          f"{prices.shape}")
    check("nifty aligned", len(nifty) == len(prices))
    check("fundamentals complete", fdf.shape == (10, 3))

    eng = FactorEngine(rf_annual=DEFAULT_RF_ANNUAL)
    eng.set_data(prices, nifty, fdf)
    eng.run_all(ret_method="factor", cov_method="sample", fc_method="mean")

    # 2. returns
    check("stock returns computed", eng.stock_ret.shape[1] == 10)

    # 3. factors
    f = eng.factors
    check("factor table has 4 cols", list(f.columns) == ["MKT", "SMB", "HML", "WML"])
    check("factors non-degenerate", (f.std() > 1e-6).all(),
          "std=" + ", ".join(f"{c}:{f[c].std():.2e}" for c in f.columns))
    check("WML warmed up (has rows)", len(f) > 200, f"{len(f)} rows")
    check("value score uses every stock",
          len(eng.meta["value_score"].dropna()) == len(DEFAULT_SYMBOLS))

    # 4. regression
    r_sbin = eng.reg["SBIN"]
    check("regression produced betas", "MKT" in r_sbin["coef"])
    mean_r2 = np.mean([v["r2"] for v in eng.reg.values() if "r2" in v])
    check("avg R^2 sensible (>0.3 on factor-built demo)", mean_r2 > 0.3,
          f"avg R2={mean_r2:.3f}")
    mkt_betas = [v["coef"]["MKT"] for v in eng.reg.values() if "coef" in v]
    check("market betas positive-ish", np.mean(mkt_betas) > 0.5,
          f"avg bMKT={np.mean(mkt_betas):.2f}")

    # 5. forecast + expected returns
    fc = eng.factor_forecast
    check("factor forecast has 4 entries", len(fc) == 4)
    shrunk = forecast_factor_returns(eng.factors, "shrink", shrinkage=0.75)
    check("factor shrinkage reduces forecast magnitude",
          np.abs(shrunk).sum() <= np.abs(eng.factors.mean()).sum() + 1e-12)
    mu = eng.mu
    check("expected returns finite", np.isfinite(mu.values).all(),
          "mu range [%.1f%%, %.1f%%]" % (mu.min() * 100, mu.max() * 100))

    # compare methods differ
    mu_hist = eng.expected("historical")
    mu_capm = eng.expected("capm")
    eng.expected("factor")
    check("factor vs historical differ",
          np.abs((mu - mu_hist)).mean() > 1e-4)
    check("capm expected finite", np.isfinite(mu_capm.values).all())

    # 6. covariance PSD
    cov = eng.cov
    eigs = np.linalg.eigvalsh(cov.values)
    check("covariance PSD", eigs.min() > -1e-8, f"min eig={eigs.min():.2e}")
    check("covariance symmetric",
          np.allclose(cov.values, cov.values.T))
    diag = covariance_diagnostics(cov)
    check("covariance diagnostics pass", diag["is_psd"])

    # factor covariance path
    covf = eng.covariance("factor")
    check("factor covariance PSD",
          np.linalg.eigvalsh(covf.values).min() > -1e-6)
    eng.covariance("sample")

    # 7. optimiser
    muv = eng.mu.values
    covv = eng.cov.values
    sh = optimise_portfolio(muv, covv, eng.rf_annual, "sharpe")
    check("max-sharpe weights sum to 1", abs(sh["weights"].sum() - 1) < 1e-6,
          f"sharpe={sh['sharpe']:.3f}")
    check("max-sharpe long-only", (sh["weights"] >= -1e-6).all())
    mv = optimise_portfolio(muv, covv, eng.rf_annual, "minvar")
    check("min-var vol <= max-sharpe vol", mv["vol"] <= sh["vol"] + 1e-6,
          f"minvar_vol={mv['vol']:.3f} sharpe_vol={sh['vol']:.3f}")
    tgt = float(np.median(muv))
    tr = optimise_portfolio(muv, covv, eng.rf_annual, "target", tgt)
    check("target-return hits target", abs(tr["ret"] - tgt) < 1e-3,
          f"got {tr['ret']:.4f} want {tgt:.4f}")

    # 8. frontier
    ef = efficient_frontier(muv, covv, eng.rf_annual, n_points=20)
    check("efficient frontier produced points", len(ef) >= 5, f"{len(ef)} pts")
    mc_short = monte_carlo_cloud(muv, covv, eng.rf_annual, n=500, allow_short=True)
    check("frontier MC scaled normally", np.nanmax(np.abs(mc_short[:, :2])) < 5.0,
          f"max abs ret/vol={np.nanmax(np.abs(mc_short[:, :2])):.2f}")

    # 9. risk analytics and out-of-sample simulation
    risk = portfolio_risk_metrics(eng.stock_ret.values, sh["weights"])
    check("risk metrics finite", all(np.isfinite(v) for v in risk.values()))
    rc = risk_contributions(sh["weights"], eng.cov.values)
    check("risk contributions sum to one", abs(rc.sum() - 1.0) < 1e-6)
    attr = factor_attribution(sh["weights"], list(eng.cov.columns), eng.reg,
                              eng.factor_forecast)
    check("factor attribution covers four factors", len(attr) == 4)
    regimes = regime_analysis(eng.nifty_ret)
    check("regime analysis produces observations", len(regimes) > 100)
    wf = walk_forward_backtest(prices, nifty, fdf, DEFAULT_RF_ANNUAL,
                               train_days=300, rebalance_days=63,
                               transaction_cost_bps=10)
    check("walk-forward produces OOS returns", len(wf["returns"]) > 100,
          f"{len(wf['returns'])} rows")
    check("walk-forward weights obey 40% cap",
          wf["weights"].max().max() <= 0.4001)

    print("-" * 70)
    print("  RESULT:", "ALL TESTS PASSED " if ok else "SOME TESTS FAILED ")
    print("=" * 70)
    return 0 if ok else 1


# =============================================================================
#  GUI
# =============================================================================

def _launch_gui(demo=False, _on_ready=None):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg, NavigationToolbar2Tk)

    # ---- matplotlib global styling to match the dark theme ----------------
    matplotlib.rcParams.update({
        "figure.facecolor": CLR["panel"],
        "axes.facecolor": CLR["panel2"],
        "axes.edgecolor": CLR["grid"],
        "axes.labelcolor": CLR["fg"],
        "text.color": CLR["fg"],
        "xtick.color": CLR["muted"],
        "ytick.color": CLR["muted"],
        "grid.color": CLR["grid"],
        "font.size": 8.5,
    })

    class FactorTerminalApp:
        def __init__(self, root):
            self.root = root
            self.demo = demo
            self.engine = FactorEngine(rf_annual=DEFAULT_RF_ANNUAL,
                                       log=self._log)
            self.busy = False
            self._figs = []  # keep refs

            root.title(APP_NAME)
            root.geometry("1280x820")
            root.minsize(1080, 700)
            root.configure(bg=CLR["bg"])

            self._style()
            self._build_header()
            self._build_body()
            self._build_status()

            self.fund_df = pd.DataFrame(
                SEED_FUNDAMENTALS, index=["mktcap_cr", "pb", "pe"]).T
            self._populate_fund_tree()
            self._log("Ready. " +
                      ("Demo mode." if demo else "Load a price CSV or use demo data."))
            if demo:
                self.root.after(400, self._load_demo)

        # -- styling --------------------------------------------------------
        def _style(self):
            st = ttk.Style()
            try:
                st.theme_use("clam")
            except Exception:
                pass
            c = CLR
            st.configure(".", background=c["bg"], foreground=c["fg"],
                         fieldbackground=c["entry"], bordercolor=c["grid"])
            st.configure("TFrame", background=c["bg"])
            st.configure("Panel.TFrame", background=c["panel"])
            st.configure("Card.TFrame", background=c["panel2"])
            st.configure("TLabel", background=c["bg"], foreground=c["fg"])
            st.configure("Panel.TLabel", background=c["panel"], foreground=c["fg"])
            st.configure("Muted.TLabel", background=c["panel"], foreground=c["muted"])
            st.configure("Head.TLabel", background=c["bg"], foreground=c["accent"],
                         font=("Segoe UI", 11, "bold"))
            st.configure("Big.TLabel", background=c["panel"], foreground=c["accent"],
                         font=("Segoe UI", 15, "bold"))
            st.configure("TButton", background=c["panel2"], foreground=c["fg"],
                         borderwidth=0, focusthickness=0, padding=6)
            st.map("TButton",
                   background=[("active", c["grid"]), ("pressed", c["sel"])])
            st.configure("Accent.TButton", background=c["accent"],
                         foreground="#03201d", font=("Segoe UI", 9, "bold"),
                         padding=7)
            st.map("Accent.TButton",
                   background=[("active", "#5fe6d6"), ("pressed", "#1ba99a")])
            st.configure("TEntry", fieldbackground=c["entry"], foreground=c["fg"],
                         insertcolor=c["fg"], bordercolor=c["grid"])
            st.configure("TCombobox", fieldbackground=c["entry"],
                         background=c["panel2"], foreground=c["fg"],
                         arrowcolor=c["accent"])
            st.map("TCombobox", fieldbackground=[("readonly", c["entry"])],
                   foreground=[("readonly", c["fg"])])
            st.configure("TNotebook", background=c["bg"], borderwidth=0)
            st.configure("TNotebook.Tab", background=c["panel"],
                         foreground=c["muted"], padding=(14, 7),
                         font=("Segoe UI", 9))
            st.map("TNotebook.Tab",
                   background=[("selected", c["panel2"])],
                   foreground=[("selected", c["accent"])])
            st.configure("Treeview", background=c["panel2"],
                         fieldbackground=c["panel2"], foreground=c["fg"],
                         rowheight=22, borderwidth=0)
            st.configure("Treeview.Heading", background=c["panel"],
                         foreground=c["accent"], relief="flat",
                         font=("Segoe UI", 8, "bold"))
            st.map("Treeview.Heading", background=[("active", c["grid"])])
            st.map("Treeview", background=[("selected", c["sel"])],
                   foreground=[("selected", c["fg"])])
            st.configure("TLabelframe", background=c["panel"],
                         foreground=c["accent"], bordercolor=c["grid"])
            st.configure("TLabelframe.Label", background=c["panel"],
                         foreground=c["accent"], font=("Segoe UI", 9, "bold"))
            st.configure("TRadiobutton", background=c["panel"], foreground=c["fg"])
            st.map("TRadiobutton", background=[("active", c["panel"])])
            st.configure("TCheckbutton", background=c["panel"], foreground=c["fg"])
            st.map("TCheckbutton", background=[("active", c["panel"])])
            st.configure("Vertical.TScrollbar", background=c["panel2"],
                         troughcolor=c["bg"], arrowcolor=c["muted"])
            st.configure("Horizontal.TScrollbar", background=c["panel2"],
                         troughcolor=c["bg"], arrowcolor=c["muted"])

        # -- header / status ------------------------------------------------
        def _build_header(self):
            h = tk.Frame(self.root, bg=CLR["bg"], height=54)
            h.pack(fill="x", side="top")
            h.pack_propagate(False)
            tk.Label(h, text="MULTI-FACTOR", fg=CLR["accent"], bg=CLR["bg"],
                     font=("Segoe UI", 18, "bold")).pack(side="left", padx=(16, 2))
            tk.Label(h, text="   Portfolio Validation Terminal",
                     fg=CLR["muted"], bg=CLR["bg"],
                     font=("Segoe UI", 10)).pack(side="left")
            self.mode_lbl = tk.Label(h, text="", fg=CLR["warn"], bg=CLR["bg"],
                                     font=("Segoe UI", 9, "bold"))
            self.mode_lbl.pack(side="right", padx=16)
            tk.Frame(self.root, bg=CLR["accent"], height=2).pack(fill="x")

        def _build_status(self):
            s = tk.Frame(self.root, bg=CLR["panel"], height=26)
            s.pack(fill="x", side="bottom")
            s.pack_propagate(False)
            self.status = tk.Label(s, text="Ready", fg=CLR["muted"],
                                   bg=CLR["panel"], anchor="w",
                                   font=("Consolas", 8))
            self.status.pack(side="left", fill="x", expand=True, padx=10)
            self.prog = ttk.Progressbar(s, mode="determinate", length=180)
            self.prog.pack(side="right", padx=10, pady=4)

        def _set_status(self, txt):
            self.status.config(text=txt)

        def _log(self, msg):
            ts = datetime.now().strftime("%H:%M:%S")
            line = f"[{ts}] {msg}"
            print(line)
            try:
                self.root.after(0, lambda: self._set_status(msg))
                if hasattr(self, "logbox"):
                    self.root.after(0, lambda: self._append_log(line))
            except Exception:
                pass

        def _append_log(self, line):
            self.logbox.config(state="normal")
            self.logbox.insert("end", line + "\n")
            self.logbox.see("end")
            self.logbox.config(state="disabled")

        # -- body / tabs ----------------------------------------------------
        def _build_body(self):
            self.nb = ttk.Notebook(self.root)
            self.nb.pack(fill="both", expand=True, padx=6, pady=6)
            self.tab_data = ttk.Frame(self.nb, style="Panel.TFrame")
            self.tab_fac = ttk.Frame(self.nb, style="Panel.TFrame")
            self.tab_reg = ttk.Frame(self.nb, style="Panel.TFrame")
            self.tab_exp = ttk.Frame(self.nb, style="Panel.TFrame")
            self.tab_cov = ttk.Frame(self.nb, style="Panel.TFrame")
            self.tab_opt = ttk.Frame(self.nb, style="Panel.TFrame")
            self.tab_ef = ttk.Frame(self.nb, style="Panel.TFrame")
            self.tab_bt = ttk.Frame(self.nb, style="Panel.TFrame")
            self.tab_rep = ttk.Frame(self.nb, style="Panel.TFrame")
            self.nb.add(self.tab_data, text="1  Data Loader")
            self.nb.add(self.tab_fac, text="2  Factor Builder")
            self.nb.add(self.tab_reg, text="3  Regression")
            self.nb.add(self.tab_exp, text="4  Expected Return")
            self.nb.add(self.tab_cov, text="5  Covariance")
            self.nb.add(self.tab_opt, text="6  Optimizer")
            self.nb.add(self.tab_ef, text="7  Efficient Frontier")
            self.nb.add(self.tab_bt, text="8  Walk Forward")
            self.nb.add(self.tab_rep, text="9  Report")
            self._tab_data()
            self._tab_factors()
            self._tab_regression()
            self._tab_expected()
            self._tab_cov()
            self._tab_optimizer()
            self._tab_frontier()
            self._tab_backtest()
            self._tab_report()

        # -- embed helper ---------------------------------------------------
        def _embed_fig(self, parent, figsize=(6, 4), toolbar=False):
            fig = Figure(figsize=figsize, dpi=100)
            fig.patch.set_facecolor(CLR["panel"])
            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas.get_tk_widget().configure(bg=CLR["panel"], highlightthickness=0)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            if toolbar:
                tb = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
                tb.configure(bg=CLR["panel"])
                tb.pack(fill="x")
            self._figs.append(fig)
            return fig, canvas

        # ================================================================
        #  TAB 1 : DATA LOADER
        # ================================================================
        def _tab_data(self):
            t = self.tab_data
            left = ttk.Frame(t, style="Panel.TFrame")
            left.pack(side="left", fill="y", padx=8, pady=8)
            right = ttk.Frame(t, style="Panel.TFrame")
            right.pack(side="left", fill="both", expand=True, padx=8, pady=8)

            source = ttk.LabelFrame(left, text="Local research data")
            source.pack(fill="x", pady=(0, 8))
            ttk.Label(source, text="Wide CSV: Date, NIFTY50, stock columns",
                      style="Muted.TLabel").pack(anchor="w", padx=8, pady=7)
            ttk.Button(source, text="Load Price CSV",
                       command=self._load_price_file).pack(fill="x", padx=8,
                                                           pady=(0, 8))

            # universe
            uni = ttk.LabelFrame(left, text="Universe & Window")
            uni.pack(fill="x", pady=(0, 8))
            ttk.Label(uni, text="NSE symbols (one per line):",
                      style="Muted.TLabel").pack(anchor="w", padx=8, pady=(6, 2))
            self.sym_text = tk.Text(uni, width=28, height=9, bg=CLR["entry"],
                                    fg=CLR["fg"], insertbackground=CLR["fg"],
                                    relief="flat", font=("Consolas", 9))
            self.sym_text.pack(padx=8)
            self.sym_text.insert("1.0", "\n".join(DEFAULT_SYMBOLS))
            frm = ttk.Frame(uni, style="Panel.TFrame")
            frm.pack(fill="x", padx=8, pady=6)
            ttk.Label(frm, text="Lookback (yrs)", style="Muted.TLabel").grid(
                row=0, column=0, sticky="w")
            self.v_years = tk.StringVar(value="2")
            ttk.Entry(frm, textvariable=self.v_years, width=6).grid(
                row=0, column=1, padx=6)
            ttk.Label(frm, text="Risk-free % p.a.", style="Muted.TLabel").grid(
                row=1, column=0, sticky="w", pady=(4, 0))
            self.v_rf = tk.StringVar(value=str(DEFAULT_RF_ANNUAL * 100))
            ttk.Entry(frm, textvariable=self.v_rf, width=6).grid(
                row=1, column=1, padx=6, pady=(4, 0))

            # actions
            act = ttk.Frame(left, style="Panel.TFrame")
            act.pack(fill="x", pady=(0, 8))
            ttk.Button(act, text="Load Demo Data",
                       command=self._load_demo).pack(fill="x", pady=2)
            ttk.Button(act, text="Run Full Analysis  \u25B6",
                       style="Accent.TButton",
                       command=self._run_analysis).pack(fill="x", pady=2)

            # fundamentals grid
            fund = ttk.LabelFrame(right, text=(
                "Fundamentals  (user-supplied \u2014 drive Size & Value factors; "
                "double-click a cell to edit)"))
            fund.pack(fill="both", expand=True)
            bar = ttk.Frame(fund, style="Panel.TFrame")
            bar.pack(fill="x", padx=6, pady=4)
            ttk.Button(bar, text="Import CSV", command=self._import_fund_csv
                       ).pack(side="left", padx=2)
            ttk.Button(bar, text="Reset to seed", command=self._reset_fund
                       ).pack(side="left", padx=2)
            ttk.Label(bar, text="CSV cols: symbol,mktcap_cr,pb,pe",
                      style="Muted.TLabel").pack(side="left", padx=10)
            cols = ("symbol", "mktcap_cr", "pb", "pe")
            self.fund_tree = ttk.Treeview(fund, columns=cols, show="headings",
                                          height=16)
            for c, w in zip(cols, (120, 130, 80, 80)):
                self.fund_tree.heading(c, text=c)
                self.fund_tree.column(c, width=w, anchor="e" if c != "symbol"
                                      else "w")
            self.fund_tree.pack(fill="both", expand=True, padx=6, pady=4)
            self.fund_tree.bind("<Double-1>", self._edit_fund_cell)

            # log
            logf = ttk.LabelFrame(right, text="Log")
            logf.pack(fill="x", pady=(8, 0))
            self.logbox = tk.Text(logf, height=6, bg=CLR["entry"],
                                  fg=CLR["muted"], relief="flat",
                                  font=("Consolas", 8), state="disabled")
            self.logbox.pack(fill="x", padx=6, pady=4)

        def _populate_fund_tree(self):
            self.fund_tree.delete(*self.fund_tree.get_children())
            for sym, row in self.fund_df.iterrows():
                mc = row['mktcap_cr']
                pb = row['pb']
                pe = row['pe']
                self.fund_tree.insert("", "end", values=(
                    sym,
                    "" if pd.isna(mc) else f"{mc:.0f}",
                    "" if pd.isna(pb) else f"{pb:.2f}",
                    "" if pd.isna(pe) else f"{pe:.2f}"))

        def _reset_fund(self):
            self.fund_df = pd.DataFrame(
                SEED_FUNDAMENTALS, index=["mktcap_cr", "pb", "pe"]).T
            self._populate_fund_tree()
            self._log("Fundamentals reset to seed values.")

        def _edit_fund_cell(self, event):
            item = self.fund_tree.identify_row(event.y)
            col = self.fund_tree.identify_column(event.x)
            if not item or col == "#1":  # symbol column not editable
                return
            bbox = self.fund_tree.bbox(item, col)
            if not bbox:
                return
            x, y, w, h = bbox
            cname = self.fund_tree["columns"][int(col[1:]) - 1]
            val = self.fund_tree.set(item, cname)
            ent = tk.Entry(self.fund_tree, bg=CLR["entry"], fg=CLR["fg"],
                           insertbackground=CLR["fg"], relief="flat")
            ent.place(x=x, y=y, width=w, height=h)
            ent.insert(0, val)
            ent.focus()

            def commit(_=None):
                try:
                    newv = float(ent.get())
                    self.fund_tree.set(item, cname,
                                       f"{newv:.0f}" if cname == "mktcap_cr"
                                       else f"{newv:.2f}")
                    sym = self.fund_tree.set(item, "symbol")
                    self.fund_df.loc[sym, cname] = newv
                except ValueError:
                    pass
                ent.destroy()
            ent.bind("<Return>", commit)
            ent.bind("<FocusOut>", commit)

        def _import_fund_csv(self):
            path = filedialog.askopenfilename(
                filetypes=[("CSV", "*.csv"), ("All", "*.*")])
            if not path:
                return
            try:
                df = pd.read_csv(path)
                df.columns = [c.strip().lower() for c in df.columns]
                df = df.set_index(df.columns[0])
                need = {"mktcap_cr", "pb", "pe"}
                if not need.issubset(set(df.columns)):
                    raise ValueError("CSV must have columns: symbol,mktcap_cr,pb,pe")
                df.index = [str(i).upper().strip() for i in df.index]
                self.fund_df = df[["mktcap_cr", "pb", "pe"]].astype(float)
                self._populate_fund_tree()
                self._log(f"Imported fundamentals for {len(self.fund_df)} symbols.")
            except Exception as e:
                messagebox.showerror("Import failed", str(e))

        # -- symbols / rf helpers ------------------------------------------
        def _symbols(self):
            raw = self.sym_text.get("1.0", "end").strip().splitlines()
            return [s.strip().upper() for s in raw if s.strip()]

        def _rf(self):
            try:
                return float(self.v_rf.get()) / 100.0
            except ValueError:
                return DEFAULT_RF_ANNUAL

        # -- async runner ---------------------------------------------------
        def _run_async(self, fn, on_done=None, on_err=None):
            if self.busy:
                self._log("Busy \u2014 wait for the current task to finish.")
                return
            self.busy = True
            self.prog.config(mode="indeterminate")
            self.prog.start(12)

            def worker():
                try:
                    res = fn()
                    if on_done:
                        self.root.after(0, lambda: on_done(res))
                except Exception as e:
                    tb = traceback.format_exc()
                    print(tb)
                    self.root.after(0, lambda: (
                        on_err(e) if on_err else
                        messagebox.showerror("Error", str(e))))
                finally:
                    self.root.after(0, self._async_done)
            threading.Thread(target=worker, daemon=True).start()

        def _async_done(self):
            self.busy = False
            self.prog.stop()
            self.prog.config(mode="determinate", value=0)

        # -- data actions ---------------------------------------------------
        def _load_demo(self):
            def job():
                self._log("Generating synthetic factor-structured data ...")
                syms = self._symbols() or DEFAULT_SYMBOLS
                prices, nifty, fdf = generate_demo_data(syms, n_days=520)
                return prices, nifty, fdf
            def done(res):
                prices, nifty, fdf = res
                self.fund_df = fdf
                self._populate_fund_tree()
                self.engine.rf_annual = self._rf()
                self.engine.rf_daily = self.engine.rf_annual / TRADING_DAYS
                self.engine.set_data(prices, nifty, fdf)
                self.mode_lbl.config(text="DEMO DATA", fg=CLR["warn"])
                self._log(f"Demo data ready: {prices.shape[1]} stocks, "
                          f"{prices.shape[0]} days. Running analysis ...")
                self._run_analysis()
            self._run_async(job, done)

        def _load_price_file(self):
            path = filedialog.askopenfilename(
                filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
            if not path:
                return
            try:
                prices, benchmark = load_price_csv(path)
                fundamentals = validate_fundamentals(
                    self.fund_df.reindex(prices.columns), prices.columns)
                self.engine.rf_annual = self._rf()
                self.engine.rf_daily = self.engine.rf_annual / TRADING_DAYS
                self.engine.set_data(prices, benchmark, fundamentals)
                self.mode_lbl.config(text="LOCAL CSV", fg=CLR["good"])
                self._log(f"Loaded {len(prices)} observations for "
                          f"{prices.shape[1]} stocks. Running analysis ...")
                self._run_analysis()
            except Exception as exc:
                messagebox.showerror("Price import failed", str(exc))

        # ================================================================
        #  RUN FULL ANALYSIS
        # ================================================================
        def _run_analysis(self):
            if self.engine.prices is None:
                messagebox.showinfo("No data", "Load a price CSV or demo data first.")
                return
            self.engine.fundamentals = self.fund_df.reindex(
                self.engine.prices.columns)
            if self.engine.fundamentals[["mktcap_cr", "pb", "pe"]].isna().any().any():
                messagebox.showwarning(
                    "Fundamentals", "Market-cap, P/B and P/E are needed for all symbols "
                    "to build Size/Value factors.")
                return
            self.engine.rf_annual = self._rf()
            self.engine.rf_daily = self.engine.rf_annual / TRADING_DAYS

            fc = self.fc_method.get() if hasattr(self, "fc_method") else "mean"
            rm = self.ret_method.get() if hasattr(self, "ret_method") else "factor"
            cm = self.cov_method.get() if hasattr(self, "cov_method") else "sample"

            def job():
                self._log("Computing returns & building factors ...")
                self.engine.compute_returns()
                self.engine.build_factors()
                self._log("Running factor regressions ...")
                self.engine.regress()
                self.engine.forecast(fc)
                self.engine.expected(rm)
                self.engine.covariance(cm)
                return True
            def done(_):
                self._refresh_factor_charts()
                self._refresh_regression()
                self._refresh_expected()
                self._refresh_cov()
                self._log("Analysis complete. Review tabs 2\u20138.")
            self._run_async(job, done)

        # ================================================================
        #  TAB 2 : FACTOR BUILDER
        # ================================================================
        def _tab_factors(self):
            t = self.tab_fac
            top = ttk.Frame(t, style="Panel.TFrame")
            top.pack(fill="x", padx=8, pady=(8, 0))
            ttk.Label(top, text="Factors: MKT (excess market)  \u00b7  "
                      "SMB (size)  \u00b7  HML (value)  \u00b7  WML (momentum)",
                      style="Big.TLabel").pack(side="left")
            self.fac_fig, self.fac_canvas = self._embed_fig(t, (11, 6.5),
                                                            toolbar=True)

        def _refresh_factor_charts(self):
            f = self.engine.factors
            if f is None or f.empty:
                return
            fig = self.fac_fig
            fig.clear()
            gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
            colors = {"MKT": CLR["accent"], "SMB": CLR["accent2"],
                      "HML": CLR["good"], "WML": CLR["warn"]}

            ax1 = fig.add_subplot(gs[0, 0])
            cum = (1 + f).cumprod() - 1
            for c in f.columns:
                ax1.plot(cum.index, cum[c] * 100, color=colors[c], lw=1.3, label=c)
            ax1.set_title("Cumulative factor return (%)", color=CLR["fg"])
            ax1.legend(fontsize=7, facecolor=CLR["panel2"], edgecolor=CLR["grid"],
                       labelcolor=CLR["fg"])
            ax1.grid(alpha=0.25)

            ax2 = fig.add_subplot(gs[0, 1])
            roll = f.rolling(21).mean() * TRADING_DAYS * 100
            for c in f.columns:
                ax2.plot(roll.index, roll[c], color=colors[c], lw=1.1)
            ax2.axhline(0, color=CLR["muted"], lw=0.6)
            ax2.set_title("Rolling 21d factor premium (ann. %)", color=CLR["fg"])
            ax2.grid(alpha=0.25)

            ax3 = fig.add_subplot(gs[1, 0])
            corr = f.corr()
            im = ax3.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
            ax3.set_xticks(range(len(corr)))
            ax3.set_yticks(range(len(corr)))
            ax3.set_xticklabels(corr.columns, fontsize=8)
            ax3.set_yticklabels(corr.columns, fontsize=8)
            for i in range(len(corr)):
                for j in range(len(corr)):
                    ax3.text(j, i, f"{corr.values[i, j]:.2f}", ha="center",
                             va="center", color="black", fontsize=8)
            ax3.set_title("Factor correlation", color=CLR["fg"])
            fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

            ax4 = fig.add_subplot(gs[1, 1])
            prem = f.mean() * TRADING_DAYS * 100
            ax4.bar(prem.index, prem.values,
                    color=[colors[c] for c in prem.index])
            ax4.axhline(0, color=CLR["muted"], lw=0.6)
            ax4.set_title("Mean annualised premium (%)", color=CLR["fg"])
            ax4.grid(alpha=0.25, axis="y")

            self.fac_canvas.draw()

        # ================================================================
        #  TAB 3 : REGRESSION
        # ================================================================
        def _tab_regression(self):
            t = self.tab_reg
            ttk.Label(t, text="Per-stock four-factor OLS  (excess returns; "
                      "t-stats in parentheses)", style="Big.TLabel").pack(
                      anchor="w", padx=8, pady=8)
            cols = ("stock", "alpha_bps", "bMKT", "bSMB", "bHML", "bWML",
                    "R2", "adjR2", "n")
            wrap = ttk.Frame(t, style="Panel.TFrame")
            wrap.pack(fill="both", expand=True, padx=8, pady=4)
            self.reg_tree = ttk.Treeview(wrap, columns=cols, show="headings")
            heads = {"stock": "Stock", "alpha_bps": "\u03b1 (bps/day)",
                     "bMKT": "\u03b2 MKT", "bSMB": "\u03b2 SMB",
                     "bHML": "\u03b2 HML", "bWML": "\u03b2 WML",
                     "R2": "R\u00b2", "adjR2": "adj R\u00b2", "n": "n"}
            for c in cols:
                self.reg_tree.heading(c, text=heads[c])
                self.reg_tree.column(c, width=115 if c != "stock" else 100,
                                     anchor="center" if c != "stock" else "w")
            vs = ttk.Scrollbar(wrap, orient="vertical",
                               command=self.reg_tree.yview)
            self.reg_tree.configure(yscrollcommand=vs.set)
            self.reg_tree.pack(side="left", fill="both", expand=True)
            vs.pack(side="right", fill="y")
            self.reg_tree.tag_configure("sig", foreground=CLR["good"])
            ttk.Label(t, text="Green \u03b1 => statistically significant "
                      "(p < 0.05).", style="Muted.TLabel").pack(
                      anchor="w", padx=8, pady=(0, 8))

        def _refresh_regression(self):
            reg = self.engine.reg
            if not reg:
                return
            self.reg_tree.delete(*self.reg_tree.get_children())

            def cell(r, name):
                b = r["coef"].get(name, float("nan"))
                tt = r["t"].get(name, float("nan"))
                return f"{b:+.2f} ({tt:+.1f})"
            for sym, r in reg.items():
                if "error" in r:
                    self.reg_tree.insert("", "end", values=(
                        sym, "ERR", r["error"][:8], "", "", "", "", "", ""))
                    continue
                a = r["coef"]["alpha"] * 1e4
                at = r["t"]["alpha"]
                tag = "sig" if r["p"]["alpha"] < 0.05 else ""
                self.reg_tree.insert("", "end", tags=(tag,), values=(
                    sym, f"{a:+.1f} ({at:+.1f})",
                    cell(r, "MKT"), cell(r, "SMB"), cell(r, "HML"), cell(r, "WML"),
                    f"{r['r2']:.2f}", f"{r['adj_r2']:.2f}", r["n"]))

        # ================================================================
        #  TAB 4 : EXPECTED RETURN
        # ================================================================
        def _tab_expected(self):
            t = self.tab_exp
            bar = ttk.Frame(t, style="Panel.TFrame")
            bar.pack(fill="x", padx=8, pady=8)
            ttk.Label(bar, text="Expected-return source:",
                      style="Panel.TLabel").pack(side="left")
            self.ret_method = tk.StringVar(value="factor")
            for lab, val in [("Multi-Factor", "factor"), ("Shrunk Factor", "shrunk"),
                             ("CAPM", "capm"), ("Historical", "historical")]:
                ttk.Radiobutton(bar, text=lab, value=val,
                                variable=self.ret_method,
                                command=self._refresh_expected).pack(side="left",
                                                                     padx=6)
            ttk.Label(bar, text="   Factor forecast:",
                      style="Panel.TLabel").pack(side="left")
            self.fc_method = tk.StringVar(value="shrink")
            ttk.Combobox(bar, textvariable=self.fc_method, width=8,
                         state="readonly", values=["mean", "ewma", "shrink", "zero"]).pack(
                         side="left", padx=6)
            ttk.Button(bar, text="Recompute", style="Accent.TButton",
                       command=self._recompute_expected).pack(side="left", padx=10)

            body = ttk.Frame(t, style="Panel.TFrame")
            body.pack(fill="both", expand=True, padx=8, pady=4)
            left = ttk.Frame(body, style="Panel.TFrame")
            left.pack(side="left", fill="both", expand=True)
            cols = ("stock", "hist", "capm", "factor", "diff")
            self.exp_tree = ttk.Treeview(left, columns=cols, show="headings")
            heads = {"stock": "Stock", "hist": "Historical %",
                     "capm": "CAPM %", "factor": "Multi-Factor %",
                     "diff": "Factor \u2212 Hist"}
            for c in cols:
                self.exp_tree.heading(c, text=heads[c])
                self.exp_tree.column(c, width=120, anchor="e" if c != "stock"
                                     else "w")
            self.exp_tree.pack(fill="both", expand=True)
            self.exp_tree.tag_configure("up", foreground=CLR["good"])
            self.exp_tree.tag_configure("dn", foreground=CLR["bad"])

            self.exp_fig, self.exp_canvas = self._embed_fig(body, (5.2, 5))

            ff = ttk.LabelFrame(t, text="Forecast factor premia (annualised)")
            ff.pack(fill="x", padx=8, pady=(0, 8))
            self.ff_lbl = ttk.Label(ff, text="\u2014", style="Panel.TLabel",
                                    font=("Consolas", 9))
            self.ff_lbl.pack(anchor="w", padx=8, pady=6)

        def _recompute_expected(self):
            if self.engine.reg is None:
                return
            self.engine.forecast(self.fc_method.get())
            self.engine.expected(self.ret_method.get())
            self._refresh_expected()

        def _refresh_expected(self):
            eng = self.engine
            if eng.reg is None:
                return
            eng.forecast(self.fc_method.get())
            mu_hist = eng.expected("historical") * 100
            mu_capm = eng.expected("capm") * 100
            mu_fac = eng.expected("factor") * 100
            eng.expected(self.ret_method.get())  # restore chosen

            self.exp_tree.delete(*self.exp_tree.get_children())
            for s in eng.stock_ret.columns:
                diff = mu_fac[s] - mu_hist[s]
                tag = "up" if diff >= 0 else "dn"
                self.exp_tree.insert("", "end", tags=(tag,), values=(
                    s, f"{mu_hist[s]:+.1f}", f"{mu_capm[s]:+.1f}",
                    f"{mu_fac[s]:+.1f}", f"{diff:+.1f}"))

            fc = eng.factor_forecast * TRADING_DAYS * 100
            self.ff_lbl.config(text="   ".join(
                f"{k}: {v:+.2f}%" for k, v in fc.items()))

            mu_shrunk = eng.expected("shrunk") * 100
            eng.expected(self.ret_method.get())
            chosen = {"factor": mu_fac, "shrunk": mu_shrunk, "capm": mu_capm,
                      "historical": mu_hist}[self.ret_method.get()]
            fig = self.exp_fig
            fig.clear()
            ax = fig.add_subplot(111)
            order = chosen.sort_values()
            cols = [CLR["good"] if v >= 0 else CLR["bad"] for v in order.values]
            ax.barh(order.index, order.values, color=cols)
            ax.axvline(0, color=CLR["muted"], lw=0.6)
            ax.set_title(f"Expected return \u2014 {self.ret_method.get()} (%)",
                         color=CLR["fg"], fontsize=9)
            ax.grid(alpha=0.2, axis="x")
            fig.tight_layout()
            self.exp_canvas.draw()

        # ================================================================
        #  TAB 5 : COVARIANCE
        # ================================================================
        def _tab_cov(self):
            t = self.tab_cov
            bar = ttk.Frame(t, style="Panel.TFrame")
            bar.pack(fill="x", padx=8, pady=8)
            ttk.Label(bar, text="Covariance estimator:",
                      style="Panel.TLabel").pack(side="left")
            self.cov_method = tk.StringVar(value="factor")
            for lab, val in [("Sample", "sample"), ("EWMA", "ewma"),
                             ("Factor", "factor")]:
                ttk.Radiobutton(bar, text=lab, value=val,
                                variable=self.cov_method,
                                command=self._refresh_cov).pack(side="left", padx=6)
            ttk.Label(bar, text="   (annualised)", style="Muted.TLabel").pack(
                side="left")
            self.cov_diag_lbl = ttk.Label(bar, text="", style="Muted.TLabel")
            self.cov_diag_lbl.pack(side="right", padx=8)
            self.cov_fig, self.cov_canvas = self._embed_fig(t, (9, 6.5),
                                                            toolbar=True)

        def _refresh_cov(self):
            eng = self.engine
            if eng.stock_ret is None:
                return
            eng.covariance(self.cov_method.get())
            cov = eng.cov
            diag = covariance_diagnostics(cov.values)
            self.cov_diag_lbl.config(
                text=f"min eigen {diag['min_eigenvalue']:.2e}  |  "
                     f"condition {diag['condition_number']:.2e}  |  PSD yes")
            d = np.sqrt(np.diag(cov.values))
            corr = cov.values / np.outer(d, d)
            fig = self.cov_fig
            fig.clear()
            ax = fig.add_subplot(111)
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            syms = list(cov.columns)
            ax.set_xticks(range(len(syms)))
            ax.set_yticks(range(len(syms)))
            ax.set_xticklabels(syms, rotation=60, ha="right", fontsize=7)
            ax.set_yticklabels(syms, fontsize=7)
            if len(syms) <= 16:
                for i in range(len(syms)):
                    for j in range(len(syms)):
                        ax.text(j, i, f"{corr[i, j]:.2f}", ha="center",
                                va="center", fontsize=6,
                                color="black" if abs(corr[i, j]) < 0.6 else "white")
            ax.set_title(f"Correlation ({self.cov_method.get()} covariance)",
                         color=CLR["fg"], fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            self.cov_canvas.draw()

        # ================================================================
        #  TAB 6 : OPTIMIZER
        # ================================================================
        def _tab_optimizer(self):
            t = self.tab_opt
            bar = ttk.Frame(t, style="Panel.TFrame")
            bar.pack(fill="x", padx=8, pady=8)
            ttk.Label(bar, text="Objective:", style="Panel.TLabel").pack(side="left")
            self.opt_obj = tk.StringVar(value="sharpe")
            ttk.Combobox(bar, textvariable=self.opt_obj, width=12, state="readonly",
                         values=["sharpe", "minvar", "target"]).pack(side="left",
                                                                     padx=6)
            ttk.Label(bar, text="Target %", style="Muted.TLabel").pack(side="left")
            self.opt_target = tk.StringVar(value="15")
            ttk.Entry(bar, textvariable=self.opt_target, width=6).pack(side="left",
                                                                       padx=4)
            ttk.Label(bar, text="Max wt %", style="Muted.TLabel").pack(side="left")
            self.opt_wmax = tk.StringVar(value="40")
            ttk.Entry(bar, textvariable=self.opt_wmax, width=6).pack(side="left",
                                                                     padx=4)
            ttk.Label(bar, text="Min wt %", style="Muted.TLabel").pack(side="left")
            self.opt_wmin = tk.StringVar(value="0")
            ttk.Entry(bar, textvariable=self.opt_wmin, width=5).pack(side="left", padx=3)
            ttk.Label(bar, text="Cost bps", style="Muted.TLabel").pack(side="left")
            self.opt_cost = tk.StringVar(value="10")
            ttk.Entry(bar, textvariable=self.opt_cost, width=5).pack(side="left", padx=3)
            self.opt_short = tk.BooleanVar(value=False)
            ttk.Checkbutton(bar, text="Allow short", variable=self.opt_short
                            ).pack(side="left", padx=8)
            ttk.Button(bar, text="Optimize  \u25B6", style="Accent.TButton",
                       command=self._run_optimizer).pack(side="left", padx=10)
            ttk.Label(bar, text="(uses tab-4 return source)",
                      style="Muted.TLabel").pack(side="left")

            body = ttk.Frame(t, style="Panel.TFrame")
            body.pack(fill="both", expand=True, padx=8, pady=4)
            left = ttk.Frame(body, style="Panel.TFrame")
            left.pack(side="left", fill="both", expand=True)
            cols = ("stock", "weight")
            self.opt_tree = ttk.Treeview(left, columns=cols, show="headings",
                                         height=16)
            self.opt_tree.heading("stock", text="Stock")
            self.opt_tree.heading("weight", text="Weight %")
            self.opt_tree.column("stock", width=120, anchor="w")
            self.opt_tree.column("weight", width=110, anchor="e")
            self.opt_tree.pack(fill="both", expand=True)
            self.opt_stats = ttk.Label(left, text="\u2014", style="Panel.TLabel",
                                       font=("Consolas", 10))
            self.opt_stats.pack(anchor="w", pady=6)

            self.opt_fig, self.opt_canvas = self._embed_fig(body, (5.5, 5))

        def _run_optimizer(self):
            eng = self.engine
            if eng.mu is None or eng.cov is None:
                messagebox.showinfo("No model", "Run Full Analysis first.")
                return
            eng.expected(self.ret_method.get())
            mu = eng.mu.reindex(eng.cov.columns)
            if mu.isna().any():
                messagebox.showwarning("Expected returns",
                                       "Some expected returns are NaN.")
                return
            obj = self.opt_obj.get()
            try:
                wmax = float(self.opt_wmax.get()) / 100.0
                wmin = float(self.opt_wmin.get()) / 100.0
                cost_bps = float(self.opt_cost.get())
                tgt = float(self.opt_target.get()) / 100.0
            except ValueError:
                wmax, wmin, cost_bps, tgt = 0.40, 0.0, 10.0, 0.15
            previous = (self.last_opt["weights"] if hasattr(self, "last_opt")
                        else np.repeat(1.0 / len(mu), len(mu)))
            res = optimise_portfolio(mu.values, eng.cov.values, eng.rf_annual,
                                     obj, tgt, self.opt_short.get(), wmax,
                                     wmin=wmin,
                                     gross_limit=1.5 if self.opt_short.get() else 1.0,
                                     previous_weights=previous,
                                     turnover_limit=0.75,
                                     transaction_cost_bps=cost_bps)
            self.last_opt = res
            self.opt_tree.delete(*self.opt_tree.get_children())
            for s, w in zip(eng.cov.columns, res["weights"]):
                if abs(w) > 1e-4:
                    self.opt_tree.insert("", "end", values=(s, f"{w * 100:+.2f}"))
            self.opt_stats.config(text=(
                f"E[R] {res['ret'] * 100:6.2f}%    "
                f"Vol {res['vol'] * 100:6.2f}%    "
                f"Sharpe {res['sharpe']:5.2f}\n"
                f"Turnover {res['turnover']*100:5.1f}%   "
                f"Cost {res['cost']*100:.3f}%   HHI {res['concentration']:.3f}"))
            fig = self.opt_fig
            fig.clear()
            ax = fig.add_subplot(111)
            w = pd.Series(res["weights"], index=eng.cov.columns)
            wp = w[w > 1e-3]
            if len(wp):
                ax.pie(wp.values, labels=wp.index, autopct="%1.0f%%",
                       textprops={"color": CLR["fg"], "fontsize": 8},
                       colors=[matplotlib.cm.viridis(x) for x in
                               np.linspace(0.15, 0.9, len(wp))])
                ax.set_title(f"Allocation \u2014 {obj}", color=CLR["fg"],
                             fontsize=9)
            self.opt_canvas.draw()
            self._log(f"Optimised ({obj}): Sharpe {res['sharpe']:.2f}")

        # ================================================================
        #  TAB 7 : EFFICIENT FRONTIER
        # ================================================================
        def _tab_frontier(self):
            t = self.tab_ef
            bar = ttk.Frame(t, style="Panel.TFrame")
            bar.pack(fill="x", padx=8, pady=8)
            ttk.Button(bar, text="Build Frontier  \u25B6", style="Accent.TButton",
                       command=self._run_frontier).pack(side="left")
            ttk.Label(bar, text="  Monte-Carlo cloud + frontier + max-Sharpe / "
                      "min-var markers (uses tab-4 return source)",
                      style="Muted.TLabel").pack(side="left", padx=8)
            self.ef_fig, self.ef_canvas = self._embed_fig(t, (10, 6.5),
                                                          toolbar=True)

        def _run_frontier(self):
            eng = self.engine
            if eng.mu is None:
                messagebox.showinfo("No model", "Run Full Analysis first.")
                return

            def job():
                eng.expected(self.ret_method.get())
                mu = eng.mu.reindex(eng.cov.columns).values
                cov = eng.cov.values
                short = self.opt_short.get()
                wmax = float(self.opt_wmax.get()) / 100.0
                wmin = float(self.opt_wmin.get()) / 100.0
                gross = 1.5 if short else 1.0
                ef = efficient_frontier(mu, cov, eng.rf_annual, 40, short,
                                        wmax, wmin, gross)
                mc = monte_carlo_cloud(mu, cov, eng.rf_annual, 3500, short,
                                       w_max=wmax)
                sh = optimise_portfolio(mu, cov, eng.rf_annual, "sharpe",
                                        allow_short=short, w_max=wmax,
                                        w_min=wmin, gross_limit=gross)
                mv = optimise_portfolio(mu, cov, eng.rf_annual, "minvar",
                                        allow_short=short, w_max=wmax,
                                        w_min=wmin, gross_limit=gross)
                return ef, mc, sh, mv
            def done(res):
                ef, mc, sh, mv = res
                fig = self.ef_fig
                fig.clear()
                ax = fig.add_subplot(111)
                sc = ax.scatter(mc[:, 1] * 100, mc[:, 0] * 100, c=mc[:, 2],
                                cmap="viridis", s=6, alpha=0.35)
                if len(ef):
                    ax.plot(ef[:, 0] * 100, ef[:, 1] * 100, color=CLR["accent"],
                            lw=2, label="Efficient frontier")
                ax.scatter([sh["vol"] * 100], [sh["ret"] * 100], marker="*",
                           s=280, color=CLR["warn"], edgecolor="black",
                           label=f"Max Sharpe ({sh['sharpe']:.2f})", zorder=5)
                ax.scatter([mv["vol"] * 100], [mv["ret"] * 100], marker="D",
                           s=90, color=CLR["bad"], edgecolor="black",
                           label="Min variance", zorder=5)
                ax.set_xlabel("Volatility (ann. %)")
                ax.set_ylabel("Expected return (ann. %)")
                ax.set_title("Efficient frontier & random portfolios",
                             color=CLR["fg"])
                ax.legend(fontsize=8, facecolor=CLR["panel2"],
                          edgecolor=CLR["grid"], labelcolor=CLR["fg"])
                ax.grid(alpha=0.2)
                fig.colorbar(sc, ax=ax, label="Sharpe")
                fig.tight_layout()
                self.ef_canvas.draw()
                self._log("Efficient frontier built.")
            self._run_async(job, done)

        # ================================================================
        #  TAB 8 : WALK-FORWARD VALIDATION
        # ================================================================
        def _tab_backtest(self):
            t = self.tab_bt
            bar = ttk.Frame(t, style="Panel.TFrame")
            bar.pack(fill="x", padx=8, pady=8)
            ttk.Label(bar, text="Train sessions", style="Panel.TLabel").pack(side="left")
            self.bt_train = tk.StringVar(value="252")
            ttk.Entry(bar, textvariable=self.bt_train, width=6).pack(side="left", padx=4)
            ttk.Label(bar, text="Rebalance sessions", style="Panel.TLabel").pack(side="left")
            self.bt_rebalance = tk.StringVar(value="21")
            ttk.Entry(bar, textvariable=self.bt_rebalance, width=6).pack(side="left", padx=4)
            ttk.Label(bar, text="Cost bps", style="Panel.TLabel").pack(side="left")
            self.bt_cost = tk.StringVar(value="10")
            ttk.Entry(bar, textvariable=self.bt_cost, width=6).pack(side="left", padx=4)
            ttk.Button(bar, text="Run Out-of-Sample Test  \u25B6",
                       style="Accent.TButton", command=self._run_backtest).pack(side="left", padx=10)
            self.bt_stats = ttk.Label(t, text="", style="Panel.TLabel",
                                      font=("Consolas", 9))
            self.bt_stats.pack(fill="x", padx=10)
            self.bt_fig, self.bt_canvas = self._embed_fig(t, (10, 6), toolbar=True)

        def _run_backtest(self):
            eng = self.engine
            if eng.prices is None:
                messagebox.showinfo("No data", "Load data and run the analysis first.")
                return
            try:
                train = int(self.bt_train.get())
                rebalance = int(self.bt_rebalance.get())
                cost = float(self.bt_cost.get())
                wmax = float(self.opt_wmax.get()) / 100.0
            except ValueError:
                messagebox.showerror("Invalid settings", "Backtest settings must be numeric.")
                return

            def job():
                return walk_forward_backtest(
                    eng.prices, eng.nifty, eng.fundamentals, eng.rf_annual,
                    train_days=train, rebalance_days=rebalance,
                    ret_method=self.ret_method.get(),
                    cov_method=self.cov_method.get(),
                    forecast_method=self.fc_method.get(), w_max=wmax,
                    transaction_cost_bps=cost)

            def done(result):
                self.last_backtest = result
                stats = result["statistics"]
                self.bt_stats.config(text="   ".join(
                    f"{name}: return {row['realised_return']*100:.1f}% | "
                    f"vol {row['realised_volatility']*100:.1f}% | "
                    f"MDD {row['max_drawdown']*100:.1f}%"
                    for name, row in stats.iterrows()))
                fig = self.bt_fig
                fig.clear()
                ax = fig.add_subplot(111)
                for name in result["wealth"].columns:
                    ax.plot(result["wealth"].index, result["wealth"][name],
                            label=name, lw=1.8)
                ax.set_title("Walk-forward out-of-sample wealth", color=CLR["fg"])
                ax.set_ylabel("Growth of 1.00")
                ax.grid(alpha=0.2)
                ax.legend(facecolor=CLR["panel2"], edgecolor=CLR["grid"],
                          labelcolor=CLR["fg"])
                fig.tight_layout()
                self.bt_canvas.draw()
                self._log("Walk-forward validation complete.")
            self._run_async(job, done)

        # ================================================================
        #  TAB 9 : REPORT
        # ================================================================
        def _tab_report(self):
            t = self.tab_rep
            bar = ttk.Frame(t, style="Panel.TFrame")
            bar.pack(fill="x", padx=8, pady=8)
            ttk.Button(bar, text="Generate Report", style="Accent.TButton",
                       command=self._build_report).pack(side="left")
            ttk.Button(bar, text="Export .txt", command=lambda: self._export("txt")
                       ).pack(side="left", padx=6)
            ttk.Button(bar, text="Export weights .csv",
                       command=lambda: self._export("csv")).pack(side="left")
            self.report_box = tk.Text(t, bg=CLR["entry"], fg=CLR["fg"],
                                      relief="flat", font=("Consolas", 9),
                                      wrap="none")
            self.report_box.pack(fill="both", expand=True, padx=8, pady=8)

        def _build_report(self):
            eng = self.engine
            if eng.reg is None:
                messagebox.showinfo("No model", "Run Full Analysis first.")
                return
            eng.expected(self.ret_method.get())
            L = []
            L.append("=" * 72)
            L.append(" Multi-Factor Portfolio Validation Report")
            L.append(" " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                     + f"   mode={'DEMO' if self.demo else 'LOCAL'}")
            L.append("=" * 72)
            L.append(f" Universe        : {', '.join(eng.stock_ret.columns)}")
            L.append(f" Observations    : {len(eng.stock_ret)} trading days")
            L.append(f" Risk-free (p.a.): {eng.rf_annual*100:.2f}%")
            L.append(f" Return source   : {self.ret_method.get()}   "
                     f"Forecast: {self.fc_method.get()}   "
                     f"Covariance: {self.cov_method.get()}")
            diag = covariance_diagnostics(eng.cov.values)
            L.append(f" Covariance check: PSD={diag['is_psd']}   "
                     f"min eigen={diag['min_eigenvalue']:.3e}   "
                     f"condition={diag['condition_number']:.3e}")
            L.append("")
            L.append(" Size sort  small: " + ", ".join(eng.meta["size_small"]))
            L.append("            large: " + ", ".join(eng.meta["size_big"]))
            L.append(" Value sort cheap: " + ", ".join(eng.meta["value_cheap"]))
            L.append("        expensive: " + ", ".join(eng.meta["value_expensive"]))
            L.append("")
            fc = eng.factor_forecast * TRADING_DAYS * 100
            L.append(" Forecast factor premia (annualised):")
            for k, v in fc.items():
                L.append(f"     {k:5s}: {v:+.2f}%")
            L.append("")
            L.append(" Regression betas & expected returns")
            L.append(" " + "-" * 70)
            L.append(f" {'Stock':10s}{'aBps':>8s}{'bMKT':>8s}{'bSMB':>8s}"
                     f"{'bHML':>8s}{'bWML':>8s}{'R2':>7s}{'E[R]%':>9s}")
            mu = eng.expected(self.ret_method.get()) * 100
            for s, r in eng.reg.items():
                if "error" in r:
                    continue
                L.append(f" {s:10s}{r['coef']['alpha']*1e4:8.1f}"
                         f"{r['coef']['MKT']:8.2f}{r['coef']['SMB']:8.2f}"
                         f"{r['coef']['HML']:8.2f}{r['coef']['WML']:8.2f}"
                         f"{r['r2']:7.2f}{mu[s]:9.2f}")
            L.append("")
            if hasattr(self, "last_opt"):
                res = self.last_opt
                L.append(" Optimised portfolio (last run)")
                L.append(" " + "-" * 70)
                L.append(f"     E[R] {res['ret']*100:.2f}%   "
                         f"Vol {res['vol']*100:.2f}%   Sharpe {res['sharpe']:.2f}")
                w = pd.Series(res["weights"], index=eng.cov.columns)
                for s, wv in w[w.abs() > 1e-3].sort_values(ascending=False).items():
                    L.append(f"     {s:10s}{wv*100:+7.2f}%")
                risk = portfolio_risk_metrics(eng.stock_ret.values, res["weights"])
                L.append(f"     VaR 95% {risk['var']*100:.2f}%   "
                         f"CVaR 95% {risk['cvar']*100:.2f}%   "
                         f"Max drawdown {risk['max_drawdown']*100:.2f}%")
                L.append(f"     Turnover {res['turnover']*100:.2f}%   "
                         f"Cost {res['cost']*100:.3f}%   "
                         f"Concentration {res['concentration']:.3f}")
                rc = risk_contributions(res["weights"], eng.cov.values)
                L.append("     Risk contribution: " + "  ".join(
                    f"{s} {v*100:.1f}%" for s, v in
                    zip(eng.cov.columns, rc)))
                attribution = factor_attribution(
                    res["weights"], list(eng.cov.columns), eng.reg,
                    eng.factor_forecast)
                L.append("     Portfolio factor exposures:")
                for factor, row in attribution.iterrows():
                    L.append(f"       {factor:5s} beta {row['Exposure']:+.3f}   "
                             f"forecast contribution "
                             f"{row['Expected Return Contribution']*100:+.2f}%")
                regimes = regime_analysis(eng.nifty_ret)
                if len(regimes):
                    current = regimes.iloc[-1]
                    L.append(f"     Current regime: {current['Regime']}   "
                             f"momentum {current['Momentum']*100:+.1f}%   "
                             f"vol {current['Volatility']*100:.1f}%")
            else:
                L.append(" (Run the Optimizer tab to include portfolio weights.)")
            if hasattr(self, "last_backtest"):
                L.append("")
                L.append(" Walk-forward out-of-sample comparison")
                L.append(" " + "-" * 70)
                for name, row in self.last_backtest["statistics"].iterrows():
                    L.append(f" {name:14s} Return {row['realised_return']*100:7.2f}%  "
                             f"Vol {row['realised_volatility']*100:7.2f}%  "
                             f"MDD {row['max_drawdown']*100:7.2f}%")
            L.append("=" * 72)
            self.report_text = "\n".join(L)
            self.report_box.delete("1.0", "end")
            self.report_box.insert("1.0", self.report_text)

        def _export(self, kind):
            eng = self.engine
            if kind == "txt":
                if not hasattr(self, "report_text"):
                    self._build_report()
                path = filedialog.asksaveasfilename(
                    defaultextension=".txt", filetypes=[("Text", "*.txt")])
                if path:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(self.report_text)
                    self._log(f"Report exported: {path}")
            else:
                if not hasattr(self, "last_opt"):
                    messagebox.showinfo("No portfolio", "Run the Optimizer first.")
                    return
                path = filedialog.asksaveasfilename(
                    defaultextension=".csv", filetypes=[("CSV", "*.csv")])
                if path:
                    w = pd.Series(self.last_opt["weights"], index=eng.cov.columns,
                                  name="weight")
                    mu = eng.expected(self.ret_method.get())
                    out = pd.DataFrame({"weight": w,
                                        "expected_return": mu.reindex(w.index)})
                    out.to_csv(path)
                    self._log(f"Weights exported: {path}")

    root = tk.Tk()
    app = FactorTerminalApp(root)
    if _on_ready is not None:
        root.after(1200, lambda: _on_ready(app, root))
    root.mainloop()


# =============================================================================
#  MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Multi-Factor Portfolio Validation Terminal")
    ap.add_argument("--test", action="store_true",
                    help="run headless self-test (no GUI, no network)")
    ap.add_argument("--demo", action="store_true",
                    help="launch GUI preloaded with synthetic demo data")
    ap.add_argument("--walk-forward", action="store_true",
                    help="run a reproducible headless walk-forward demonstration")
    args = ap.parse_args()

    if args.test:
        sys.exit(self_test())
    if args.walk_forward:
        prices, benchmark, fundamentals = generate_demo_data(DEFAULT_SYMBOLS, 756)
        result = walk_forward_backtest(
            prices, benchmark, fundamentals, DEFAULT_RF_ANNUAL,
            train_days=378, rebalance_days=21)
        print("\nWALK-FORWARD OUT-OF-SAMPLE RESULTS\n")
        print(result["statistics"].to_string(float_format=lambda x: f"{x:.4f}"))
        print(f"\nTotal modelled transaction cost: {result['total_cost']:.4%}")
        return
    _launch_gui(demo=args.demo)


if __name__ == "__main__":
    main()
