"""Tests for the pairs-stat-arb strategy v1 (cointegration stat arb)."""

import itertools
import math
import random

import pytest

from atlab.models import DecisionAction, MarketState, Side, StrategyVersion
from atlab.pairs import (
    PAIRS_STRATEGY_ID,
    PAIRS_VERSION,
    adf_t_statistic,
    engle_granger_cointegrated,
    estimate_hedge_ratio,
    half_life,
    make_pairs_decision,
    spread_series,
    spread_zscore,
)
from atlab.strategy import make_decision


def pairs_strategy(**parameters):
    params = {"z_entry": 2.0, "z_exit": 0.5, "z_window": 60.0}
    params.update(parameters)
    return StrategyVersion(
        strategy_id=PAIRS_STRATEGY_ID,
        version=PAIRS_VERSION,
        hypothesis="test",
        parameters=params,
    )


def returns_for_levels(levels):
    """Simple returns whose log-price levels reconstruct the given path."""
    out = [0.0]
    for prev, cur in itertools.pairwise(levels):
        out.append(math.exp(cur - prev) - 1.0)
    return out


def pairs_state(returns):
    return MarketState(
        symbol="PAIR.TEST",
        as_of="2026-01-01T00:00:00Z",
        price=100.0,
        returns=returns,
        state_fingerprint="state-pairs-test",
    )


def ar1(rng, n, phi, sigma):
    out, x = [], 0.0
    for _ in range(n):
        x = phi * x + rng.gauss(0.0, sigma)
        out.append(x)
    return out


def random_walk(rng, n, sigma):
    out, x = [], 0.0
    for _ in range(n):
        x += rng.gauss(0.0, sigma)
        out.append(x)
    return out


# --- Formation math -------------------------------------------------------


def test_hedge_ratio_recovers_true_beta():
    rng = random.Random(11)
    factor = random_walk(rng, 300, 0.01)
    log_a = [f + e for f, e in zip(factor, ar1(rng, 300, 0.9, 0.004))]
    log_b = [f + e for f, e in zip(factor, ar1(rng, 300, 0.9, 0.004))]
    beta = estimate_hedge_ratio(log_a, log_b)
    assert beta == pytest.approx(1.0, abs=0.15)


def test_hedge_ratio_is_sanity_bounded():
    log_a = [float(i) for i in range(100)]
    log_b = [1000.0 + float(i) for i in range(100)]
    assert estimate_hedge_ratio(log_a, log_b) == pytest.approx(1.0, abs=0.01)
    with pytest.raises(ValueError):
        estimate_hedge_ratio([1.0], [1.0])


def test_engle_granger_accepts_cointegrated_pair():
    rng = random.Random(21)
    factor = random_walk(rng, 300, 0.01)
    log_a = [f + e for f, e in zip(factor, ar1(rng, 300, 0.9, 0.004))]
    log_b = [f + e for f, e in zip(factor, ar1(rng, 300, 0.9, 0.004))]
    assert engle_granger_cointegrated(log_a, log_b)


def test_engle_granger_rejects_independent_random_walks():
    rng = random.Random(31)
    log_a = random_walk(rng, 300, 0.01)
    log_b = random_walk(rng, 300, 0.01)
    assert not engle_granger_cointegrated(log_a, log_b)


def test_adf_t_statistic_separates_stationary_from_unit_root():
    rng = random.Random(41)
    stationary = ar1(rng, 300, 0.85, 0.01)
    wanderer = random_walk(rng, 300, 0.01)
    assert adf_t_statistic(stationary) < -3.4
    assert adf_t_statistic(wanderer) > -3.4


def test_spread_series_is_the_ols_residual():
    log_a = [1.0, 2.0, 3.0, 4.0]
    log_b = [2.0, 4.0, 6.0, 8.0]
    resid = spread_series(log_a, log_b, 0.5)
    assert resid == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-9)


def test_half_life_matches_ar1_parameter():
    rng = random.Random(51)
    spread = ar1(rng, 2000, 0.9, 0.01)
    # -ln(2)/ln(0.9) ~= 6.58 bars.
    assert half_life(spread) == pytest.approx(6.58, rel=0.2)


def test_half_life_rejects_non_mean_reverting():
    rng = random.Random(61)
    # A random walk's AR(1) fit implies a slow half-life, outside the
    # practitioner 5..60-bar tradeable band -- the filter rejects it.
    assert half_life(random_walk(rng, 500, 0.01)) > 60


# --- Signal math ----------------------------------------------------------


def test_spread_zscore_matches_hand_computation():
    levels = [float(i) for i in range(60)]
    z = spread_zscore(levels, 60)
    mu = sum(levels) / 60
    var = sum((x - mu) ** 2 for x in levels) / 60
    assert z == pytest.approx((59.0 - mu) / math.sqrt(var))


def test_spread_zscore_needs_full_window():
    assert spread_zscore([1.0] * 30, 60) is None
    assert spread_zscore([1.0] * 60, 60) is None  # zero variance -> no signal


# --- Decision logic -------------------------------------------------------


def deep_dip_returns():
    levels = [0.0] * 59 + [-0.06]
    return returns_for_levels(levels)


def converged_returns():
    levels = [0.01 * ((i % 2) * 2 - 1) for i in range(59)] + [0.0]
    return returns_for_levels(levels)


def test_pairs_enters_long_spread_on_deep_negative_z():
    decision = make_pairs_decision(pairs_strategy(), pairs_state(deep_dip_returns()))
    assert decision.action is DecisionAction.ENTER
    assert decision.side is Side.BUY
    assert decision.strategy_id == PAIRS_STRATEGY_ID


def test_pairs_exits_when_spread_converges():
    decision = make_pairs_decision(pairs_strategy(), pairs_state(converged_returns()))
    assert decision.action is DecisionAction.EXIT
    assert decision.side is Side.SELL


def test_pairs_holds_on_positive_z_because_engine_is_long_only():
    levels = [0.0] * 59 + [0.06]
    decision = make_pairs_decision(pairs_strategy(), pairs_state(returns_for_levels(levels)))
    assert decision.action is DecisionAction.HOLD
    assert decision.side is None
    assert "long-only" in decision.rationale


def test_pairs_holds_inside_the_band():
    levels = [-0.001 * i for i in range(60)]  # gentle drift, z ~= -1.7
    decision = make_pairs_decision(pairs_strategy(), pairs_state(returns_for_levels(levels)))
    assert decision.action is DecisionAction.HOLD
    assert decision.side is None


def test_pairs_holds_while_warming_up():
    decision = make_pairs_decision(pairs_strategy(), pairs_state([0.001] * 30))
    assert decision.action is DecisionAction.HOLD
    assert "warming up" in decision.rationale


def test_pairs_rejects_wrong_strategy_id():
    bad = StrategyVersion(
        strategy_id="sniper-mean-reversion",
        version="1.0.0",
        hypothesis="test",
        parameters={},
    )
    with pytest.raises(ValueError, match="PAIRS_STRATEGY_ID_MISMATCH"):
        make_pairs_decision(bad, pairs_state(deep_dip_returns()))


def test_pairs_rejects_bad_parameters():
    state = pairs_state(deep_dip_returns())
    with pytest.raises(ValueError, match="PAIRS_INVALID_PARAMETERS"):
        make_pairs_decision(pairs_strategy(z_entry=0.0), state)
    with pytest.raises(ValueError, match="PAIRS_INVALID_PARAMETERS"):
        make_pairs_decision(pairs_strategy(z_window=1.0), state)


def test_pairs_dispatches_through_make_decision():
    decision = make_decision(pairs_strategy(), pairs_state(deep_dip_returns()))
    assert decision.strategy_id == PAIRS_STRATEGY_ID
    assert decision.action is DecisionAction.ENTER


def test_pairs_decision_is_deterministic():
    state = pairs_state(deep_dip_returns())
    first = make_pairs_decision(pairs_strategy(), state)
    second = make_pairs_decision(pairs_strategy(), state)
    assert first.decision_id == second.decision_id
