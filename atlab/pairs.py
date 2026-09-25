"""Pairs-stat-arb strategy v1: cointegration-based statistical arbitrage.

Public recipe (Gatev, Goetzmann & Rouwenhorst 2006; Avellaneda & Lee 2008;
Chan 2013, "Algorithmic Trading"):

  formation:  OLS hedge ratio beta on log prices over a formation window;
              Engle-Granger cointegration screen; half-life filter.
  signal:     spread S_t = logP_A - beta*logP_B; z-score on a rolling
              window (literature standard: 60 bars).
  entry:      LONG the spread when z_t <= -2 (spread cheap vs its history).
  exit:       |z_t| <= 0.5 (spread converged) -- or the charter $5
              stop-loss, which overrides the strategy via the risk engine.

v1 scope limits, stated honestly:
  - LONG-SPREAD ONLY. The canonical trade is symmetric (short the spread
    when z_t >= +2), but this paper engine is long-only: it cannot model
    shorting, borrow fees, or margin. z_t >= +2 therefore yields HOLD.
    A live pairs book needs margin + borrow + recall-risk handling that
    the current charter does not grant.
  - The spread is traded as a synthetic price series P_t = 100*exp(S_t-S_0).
    log P_t = S_t + const, so a z-score on log prices IS the spread z-score.
    Two-leg execution (both-leg fills, borrow costs, recall risk) is
    abstracted away: this is a research model of the signal, not of live
    pair execution.
  - The strategy is deliberately stateless: it proposes ENTER/EXIT from the
    z-score of the latest bar only. Sizing, loss halts, and stops are the
    deterministic risk engine's job, which disposes.

Pure-Python math (statistics module): this repo carries no numpy/pandas/
statsmodels dependency, and the formation math (OLS, ADF, half-life) is
small enough to implement exactly.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics

from .models import DecisionAction, JEVDecision, MarketState, Side, StrategyVersion

PAIRS_STRATEGY_ID = "pairs-stat-arb"
PAIRS_VERSION = "1.0.0"
PAIRS_HYPOTHESIS = (
    "Two cointegrated instruments share a stationary spread; when the "
    "spread's z-score stretches past -2 it reverts toward zero often "
    "enough that buying the spread and exiting on convergence is positive "
    "expectancy at small size, with the risk charter bounding the "
    "downside when the relationship breaks."
)

# MacKinnon approximate 5% critical value for the Engle-Granger residual
# ADF test on a pair (stricter than the univariate -2.86 because the
# residuals were fitted by OLS). Documented in the research notes.
ENGLE_GRANGER_CRITICAL_5PCT = -3.4


def _mean(xs: list[float]) -> float:
    return statistics.fmean(xs)


def _variance(xs: list[float]) -> float:
    # Population variance: the z-score window is the whole population here.
    mu = _mean(xs)
    return sum((x - mu) ** 2 for x in xs) / len(xs)


def _covariance(xs: list[float], ys: list[float]) -> float:
    mx, my = _mean(xs), _mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)


def estimate_hedge_ratio(log_a: list[float], log_b: list[float]) -> float:
    """OLS hedge ratio beta of log-price A on log-price B (formation window).

    beta = Cov(logA, logB) / Var(logB), sanity-bounded to [0.1, 10] per
    practitioner convention (a beta outside that range is not a pair, it
    is a coincidence).
    """
    if len(log_a) != len(log_b) or len(log_a) < 2:
        raise ValueError("PAIRS_NEED_PAIRED_HISTORY")
    var_b = _variance(log_b)
    if var_b <= 0:
        raise ValueError("PAIRS_DEGENERATE_LEG")
    beta = _covariance(log_a, log_b) / var_b
    return min(10.0, max(0.1, beta))


def spread_series(log_a: list[float], log_b: list[float], beta: float) -> list[float]:
    """Stationary spread S_t = logA_t - beta*logB_t (the OLS residual)."""
    if len(log_a) != len(log_b):
        raise ValueError("PAIRS_NEED_PAIRED_HISTORY")
    alpha = _mean(log_a) - beta * _mean(log_b)
    return [a - alpha - beta * b for a, b in zip(log_a, log_b)]


def adf_t_statistic(residuals: list[float], lags: int = 1) -> float:
    """t-statistic of rho in the ADF regression (no deterministic term).

    d(u_t) = rho*u_{t-1} + sum_j gamma_j*d(u_{t-j}) + e_t.
    H0: rho = 0 (unit root -> no cointegration). Reject (t < critical)
    to accept the spread as stationary. Hand-rolled OLS so this module
    stays dependency-free.
    """
    n = len(residuals)
    if n < lags + 3:
        raise ValueError("PAIRS_ADF_NEED_HISTORY")
    # Regressors: u_{t-1}, d(u_{t-1}), ..., d(u_{t-lags}).
    y: list[float] = []
    x_rows: list[list[float]] = []
    for t in range(lags + 1, n):
        y.append(residuals[t] - residuals[t - 1])
        row = [residuals[t - 1]]
        for j in range(1, lags + 1):
            row.append(residuals[t - j] - residuals[t - j - 1])
        x_rows.append(row)
    return _ols_t_stat(y, x_rows, coef_index=0)


def _ols_t_stat(y: list[float], x_rows: list[list[float]], coef_index: int) -> float:
    """OLS t-statistic for one coefficient via normal equations (k <= 3)."""
    k = len(x_rows[0])
    n = len(y)
    # X'X and X'y.
    xtx = [[0.0] * k for _ in range(k)]
    xty = [0.0] * k
    for row, yi in zip(x_rows, y):
        for i in range(k):
            xty[i] += row[i] * yi
            for j in range(k):
                xtx[i][j] += row[i] * row[j]
    beta = _solve(xtx, xty)
    residuals = [yi - sum(b * xi for b, xi in zip(beta, row)) for yi, row in zip(y, x_rows)]
    dof = n - k
    if dof <= 0:
        raise ValueError("PAIRS_ADF_NEED_HISTORY")
    s2 = sum(r * r for r in residuals) / dof
    xtx_inv = _invert(xtx)
    se = math.sqrt(max(s2 * xtx_inv[coef_index][coef_index], 0.0))
    if se == 0:
        return 0.0
    return beta[coef_index] / se


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination for small systems (k <= 3)."""
    n = len(b)
    m = [row[:] + [bi] for row, bi in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("PAIRS_SINGULAR_MATRIX")
        m[col], m[pivot] = m[pivot], m[col]
        piv = m[col][col]
        m[col] = [v / piv for v in m[col]]
        for r in range(n):
            if r != col:
                factor = m[r][col]
                m[r] = [rv - factor * cv for rv, cv in zip(m[r], m[col])]
    return [m[i][n] for i in range(n)]


def _invert(a: list[list[float]]) -> list[list[float]]:
    n = len(a)
    return [_solve(a, [1.0 if i == j else 0.0 for i in range(n)]) for j in range(n)]


def engle_granger_cointegrated(
    log_a: list[float],
    log_b: list[float],
    critical: float = ENGLE_GRANGER_CRITICAL_5PCT,
) -> bool:
    """Engle-Granger two-step screen: OLS in levels, ADF on the residuals."""
    beta = estimate_hedge_ratio(log_a, log_b)
    resid = spread_series(log_a, log_b, beta)
    return adf_t_statistic(resid) < critical


def half_life(spread: list[float]) -> float:
    """Mean-reversion half-life in bars from an AR(1) fit.

    t_half = -ln(2)/ln(b). Practitioner filter: trade only 5..60 bars;
    slower means the pair reverts too slowly to be worth the capital.
    """
    if len(spread) < 3:
        raise ValueError("PAIRS_HALFLIFE_NEED_HISTORY")
    lagged = spread[:-1]
    var_lag = _variance(lagged)
    if var_lag <= 0:
        return math.inf
    b = _covariance(spread[1:], lagged) / var_lag
    if b <= 0 or b >= 1:
        return math.inf
    return -math.log(2.0) / math.log(b)


def spread_zscore(log_prices: list[float], window: int) -> float | None:
    """z-score of the latest log-price level vs the rolling window.

    Because the synthetic spread price satisfies log P_t = S_t + const,
    this is exactly the spread z-score of the canonical recipe.
    """
    if window < 2 or len(log_prices) < window:
        return None
    levels = log_prices[-window:]
    mu = _mean(levels)
    sd = math.sqrt(_variance(levels))
    if sd == 0:
        return None
    return (levels[-1] - mu) / sd


def log_price_levels(returns: list[float]) -> list[float]:
    """Reconstruct log-price levels from simple returns (translation-free).

    levels are defined up to an additive constant, which the z-score
    (mean- and scale-normalized) does not see.
    """
    levels: list[float] = []
    acc = 0.0
    for r in returns:
        if r <= -1.0:
            raise ValueError("PAIRS_INVALID_RETURN")
        acc += math.log1p(r)
        levels.append(acc)
    return levels


def make_pairs_decision(strategy: StrategyVersion, state: MarketState) -> JEVDecision:
    """Propose a pairs-stat-arb decision from the spread z-score.

    Parameters (all in ``strategy.parameters``):
      - ``z_entry``: ENTER/BUY (long the spread) when z <= -z_entry.
      - ``z_exit``:  EXIT/SELL when |z| <= z_exit (spread converged).
      - ``z_window``: rolling window for the z-score (default 60 bars).

    z >= +z_entry is the canonical short-spread entry; this long-only
    engine yields HOLD there instead (see module docstring).
    """
    if strategy.strategy_id != PAIRS_STRATEGY_ID:
        raise ValueError("PAIRS_STRATEGY_ID_MISMATCH")
    z_entry = float(strategy.parameters.get("z_entry", 2.0))
    z_exit = float(strategy.parameters.get("z_exit", 0.5))
    z_window = int(strategy.parameters.get("z_window", 60))
    if z_entry <= 0 or z_exit < 0 or z_window < 2:
        raise ValueError("PAIRS_INVALID_PARAMETERS")

    levels = log_price_levels(state.returns)
    z = spread_zscore(levels, z_window)
    if z is None:
        action, side = DecisionAction.HOLD, None
        rationale = f"Pairs warming up: {len(levels)} bars < {z_window}-bar window; no signal yet."
    elif z <= -z_entry:
        action, side = DecisionAction.ENTER, Side.BUY
        rationale = (
            f"Pairs entry: spread z-score {z:.2f} <= -{z_entry:.2f}; "
            "spread cheap vs its 60-bar history, buying the spread for "
            "the snap-back to zero."
        )
    elif abs(z) <= z_exit:
        action, side = DecisionAction.EXIT, Side.SELL
        rationale = (
            f"Pairs exit: |z| {abs(z):.2f} <= {z_exit:.2f}; spread "
            "converged, taking the round trip off."
        )
    else:
        action, side = DecisionAction.HOLD, None
        if z >= z_entry:
            rationale = (
                f"Pairs hold: z-score {z:.2f} >= +{z_entry:.2f} is the "
                "canonical short-spread entry, but this engine is "
                "long-only; standing aside."
            )
        else:
            rationale = (
                f"Pairs hold: z-score {z:.2f} inside the band "
                f"(-{z_entry:.2f}, +-{z_exit:.2f}/{z_exit:.2f}); no edge."
            )
    confidence = min(1.0, 0.5 + abs(z or 0.0) * 0.2)
    identity = {
        "strategy_id": strategy.strategy_id,
        "strategy_version": strategy.version,
        "symbol": state.symbol,
        "action": action.value,
        "side": side.value if side else None,
        "confidence": confidence,
        "state_fingerprint": state.state_fingerprint,
    }
    decision_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return JEVDecision(
        decision_id=decision_id,
        strategy_id=strategy.strategy_id,
        strategy_version=strategy.version,
        symbol=state.symbol,
        action=action,
        side=side,
        confidence=confidence,
        rationale=rationale,
        state_fingerprint=state.state_fingerprint,
    )
