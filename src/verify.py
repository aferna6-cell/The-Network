"""Statistical gate that decides whether a strategy's out-of-sample returns show
real, multiple-testing-robust edge — before it is ever allowed to trade.

This is the maker/checker split done honestly: the checker is *statistics*, not a
second LLM. An LLM is a fine judge of code correctness and a terrible judge of
whether a Sharpe ratio survived data-snooping. The numbers below get the only
vote; every threshold lives in `config.STRATEGY_GATE`, none is an agent's say-so.

Tests applied:
* Annualised Sharpe ratio.
* Newey-West (HAC) t-stat of the mean return — robust to autocorrelation.
* Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014): the probability the true
  Sharpe is positive AFTER correcting for the number of strategy variants tried.
  This is the multiple-testing correction most "my backtest Sharpe is 2!" claims
  quietly skip — the best of N random strategies has a high Sharpe by luck alone.
* Maximum drawdown.
* Minimum out-of-sample sample size.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

import config

EULER_MASCHERONI = 0.5772156649015329


def sharpe(returns, periods_per_year: float) -> float:
    """Annualised Sharpe ratio (excess-of-zero; risk-free assumed ~0)."""
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else 0.0


def max_drawdown(returns) -> float:
    """Worst peak-to-trough decline of the cumulative return curve (<= 0)."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def _nw_lags(n: int) -> int:
    """Newey-West (1994) automatic lag rule of thumb."""
    return int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def newey_west_tstat(returns, lags: int | None = None) -> float:
    """HAC t-stat for H0: mean return == 0, robust to autocorrelation."""
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n < 3:
        return 0.0
    mean = r.mean()
    e = r - mean
    if lags is None:
        lags = _nw_lags(n)
    var = float(e @ e) / n                       # gamma_0
    for k in range(1, lags + 1):
        cov = float(e[k:] @ e[:-k]) / n
        var += 2.0 * (1.0 - k / (lags + 1.0)) * cov   # Bartlett weight
    if var <= 0:
        return 0.0
    se = math.sqrt(var / n)
    return float(mean / se) if se > 0 else 0.0


def _moments(r: np.ndarray, sd: float, sr: float) -> tuple[float, float]:
    skew = float(((r - r.mean()) ** 3).mean() / sd ** 3)
    kurt = float(((r - r.mean()) ** 4).mean() / sd ** 4)   # non-excess
    return skew, kurt


def probabilistic_sharpe_ratio(returns, sr_benchmark_per_period: float = 0.0) -> float:
    """PSR: P(true per-period Sharpe > benchmark), adjusting for skew & kurtosis."""
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n < 3:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    sr = r.mean() / sd                            # per-period Sharpe
    skew, kurt = _moments(r, sd, sr)
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2))
    z = (sr - sr_benchmark_per_period) * math.sqrt(n - 1) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(sr_std_per_period: float, n_trials: int) -> float:
    """E[max per-period Sharpe] across `n_trials` independent worthless variants."""
    if n_trials <= 1 or sr_std_per_period <= 0:
        return 0.0
    n, e = n_trials, math.e
    return sr_std_per_period * (
        (1 - EULER_MASCHERONI) * norm.ppf(1 - 1.0 / n)
        + EULER_MASCHERONI * norm.ppf(1 - 1.0 / (n * e))
    )


def deflated_sharpe_ratio(returns, n_trials: int,
                          sr_std_per_period: float | None = None) -> float:
    """DSR: PSR with the benchmark set to the expected MAX Sharpe under the null —
    i.e. the Sharpe you'd expect from the luckiest of `n_trials` no-edge variants.

    `sr_std_per_period` is the cross-trial std of the Sharpe estimates. When the
    individual trial Sharpes aren't retained we fall back to the asymptotic SE of
    the Sharpe estimator, a standard and slightly conservative proxy.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n < 3:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    sr = r.mean() / sd
    if sr_std_per_period is None:
        skew, kurt = _moments(r, sd, sr)
        var = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2) / (n - 1)
        sr_std_per_period = math.sqrt(max(1e-12, var))
    sr0 = expected_max_sharpe(sr_std_per_period, n_trials)
    return probabilistic_sharpe_ratio(returns, sr_benchmark_per_period=sr0)


@dataclass
class Check:
    name: str
    value: float
    threshold: float
    passed: bool


@dataclass
class Verdict:
    passed: bool
    checks: list[Check]
    stats: dict

    def render(self) -> str:
        head = "PASS ✅" if self.passed else "FAIL ❌"
        lines = [f"Strategy gate: {head}",
                 f"{'check':<18}{'value':>12}{'threshold':>12}  result"]
        for c in self.checks:
            mark = "ok" if c.passed else "FAIL"
            lines.append(f"{c.name:<18}{c.value:>12.4f}{c.threshold:>12.4f}  {mark}")
        return "\n".join(lines)


def gate(returns, *, periods_per_year: float, n_trials: int,
         thresholds: dict | None = None) -> Verdict:
    """Run every statistical check; pass only if ALL clear their threshold."""
    t = {**config.STRATEGY_GATE, **(thresholds or {})}
    r = np.asarray(returns, dtype=float)
    sr = sharpe(r, periods_per_year)
    nw = newey_west_tstat(r)
    dsr = deflated_sharpe_ratio(r, n_trials)
    mdd = max_drawdown(r)
    n = float(r.size)
    checks = [
        Check("annual_sharpe", sr, t["min_sharpe"], sr >= t["min_sharpe"]),
        Check("newey_west_t", nw, t["min_nw_tstat"], nw >= t["min_nw_tstat"]),
        Check("deflated_sharpe", dsr, t["min_dsr"], dsr >= t["min_dsr"]),
        Check("max_drawdown", mdd, t["max_drawdown"], mdd >= t["max_drawdown"]),
        Check("oos_obs", n, float(t["min_oos_obs"]), n >= t["min_oos_obs"]),
    ]
    passed = all(c.passed for c in checks)
    stats = {"annual_sharpe": sr, "newey_west_t": nw, "deflated_sharpe": dsr,
             "max_drawdown": mdd, "n_obs": int(n), "n_trials": n_trials}
    return Verdict(passed, checks, stats)


# --- Factor-neutral alpha (Jensen's alpha) -----------------------------------
# "Beats SPY" isn't enough: excess return can be disguised exposure to known
# factors (market, momentum). Real alpha is the regression intercept that
# survives AFTER those factors are stripped out — with a Newey-West t-stat,
# because financial residuals are autocorrelated. r = alpha + sum(beta*factor)+e.

def _ols_hac(X: np.ndarray, y: np.ndarray, lags: int | None = None):
    """OLS with Newey-West (HAC) standard errors. Returns (beta, tstats, r2)."""
    n, p = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta
    if lags is None:
        lags = _nw_lags(n)
    xe = X * resid[:, None]
    meat = xe.T @ xe                                  # Σ e_t^2 x_t x_t'
    for k in range(1, lags + 1):
        g = xe[k:].T @ xe[:-k]
        meat += (1.0 - k / (lags + 1.0)) * (g + g.T)  # Bartlett-weighted lags
    # Same plain HAC convention as newey_west_tstat (no n/(n-p) correction), so
    # the no-factor case reduces exactly to that statistic.
    cov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    tstats = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, tstats, r2


@dataclass
class FactorAlpha:
    alpha_per_period: float
    alpha_annualized: float
    alpha_tstat: float
    betas: dict
    r_squared: float
    n_obs: int

    @property
    def significant(self) -> bool:
        """Positive alpha with |t| above the gate's Newey-West bar."""
        return self.alpha_per_period > 0 and self.alpha_tstat >= config.STRATEGY_GATE[
            "min_nw_tstat"]

    def render(self) -> str:
        verdict = "REAL alpha" if self.significant else "not distinguishable from beta"
        lines = [f"Factor-neutral alpha: {verdict}",
                 f"  alpha (annualised): {self.alpha_annualized:+.2%}",
                 f"  alpha t-stat (NW):  {self.alpha_tstat:+.2f}"
                 f"  (need >= {config.STRATEGY_GATE['min_nw_tstat']:.1f})",
                 f"  R^2:                {self.r_squared:.3f}"]
        for name, b in self.betas.items():
            lines.append(f"  beta[{name}]:{'':<8}{b:+.3f}")
        return "\n".join(lines)


def factor_alpha(returns, factors: dict, periods_per_year: float) -> FactorAlpha:
    """Regress strategy returns on factor returns; report the surviving alpha.

    `factors` maps name -> per-period return series (same length as `returns`).
    With no factors this reduces to the mean return with a Newey-West t-stat.
    """
    y = np.asarray(returns, dtype=float)
    names = list(factors)
    columns = [np.ones(y.size)] + [np.asarray(factors[k], dtype=float) for k in names]
    X = np.column_stack(columns)
    beta, tstats, r2 = _ols_hac(X, y)
    alpha = float(beta[0])
    return FactorAlpha(
        alpha_per_period=alpha,
        alpha_annualized=(1.0 + alpha) ** periods_per_year - 1.0,
        alpha_tstat=float(tstats[0]),
        betas={n: float(b) for n, b in zip(names, beta[1:])},
        r_squared=r2, n_obs=int(y.size))
