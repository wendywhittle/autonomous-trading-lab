"""Tests for the sniper mean-reversion strategy v1."""

import pytest

from atlab.models import DecisionAction, MarketState, Side, StrategyVersion
from atlab.sniper import (
    SNIPER_STRATEGY_ID,
    SNIPER_VERSION,
    make_sniper_decision,
)
from atlab.strategy import make_decision


def sniper_strategy(**parameters):
    params = {"entry_dip": 0.004, "exit_bounce": 0.003}
    params.update(parameters)
    return StrategyVersion(
        strategy_id=SNIPER_STRATEGY_ID,
        version=SNIPER_VERSION,
        hypothesis="test",
        parameters=params,
    )


def state_with_latest_return(latest_return: float) -> MarketState:
    returns = [0.001, -0.002, 0.0015, latest_return]
    return MarketState(
        symbol="TEST",
        as_of="2026-01-01T00:00:00Z",
        price=100.0,
        returns=returns,
        state_fingerprint="state-test",
    )


def test_sniper_buys_sharp_dips():
    decision = make_sniper_decision(sniper_strategy(), state_with_latest_return(-0.006))
    assert decision.action is DecisionAction.ENTER
    assert decision.side is Side.BUY
    assert decision.strategy_id == SNIPER_STRATEGY_ID


def test_sniper_sells_sharp_bounces():
    decision = make_sniper_decision(sniper_strategy(), state_with_latest_return(0.005))
    assert decision.action is DecisionAction.EXIT
    assert decision.side is Side.SELL


def test_sniper_holds_inside_the_dead_zone():
    decision = make_sniper_decision(sniper_strategy(), state_with_latest_return(0.001))
    assert decision.action is DecisionAction.HOLD
    assert decision.side is None


def test_sniper_triggers_exactly_at_the_thresholds():
    dip = make_sniper_decision(sniper_strategy(), state_with_latest_return(-0.004))
    assert dip.action is DecisionAction.ENTER
    bounce = make_sniper_decision(sniper_strategy(), state_with_latest_return(0.003))
    assert bounce.action is DecisionAction.EXIT


def test_sniper_decisions_are_deterministic():
    first = make_sniper_decision(sniper_strategy(), state_with_latest_return(-0.006))
    second = make_sniper_decision(sniper_strategy(), state_with_latest_return(-0.006))
    assert first.decision_id == second.decision_id


def test_sniper_rejects_a_mismatched_strategy_id():
    other = StrategyVersion(
        strategy_id="momentum", version="1", hypothesis="test",
        parameters={"entry_dip": 0.004, "exit_bounce": 0.003},
    )
    with pytest.raises(ValueError, match="SNIPER_STRATEGY_ID_MISMATCH"):
        make_sniper_decision(other, state_with_latest_return(-0.01))


def test_sniper_rejects_non_positive_parameters():
    with pytest.raises(ValueError, match="SNIPER_INVALID_PARAMETERS"):
        make_sniper_decision(
            sniper_strategy(entry_dip=0.0), state_with_latest_return(-0.01)
        )


def test_make_decision_dispatches_to_the_sniper_by_strategy_id():
    decision = make_decision(sniper_strategy(), state_with_latest_return(-0.008))
    assert decision.strategy_id == SNIPER_STRATEGY_ID
    assert decision.action is DecisionAction.ENTER
    assert decision.side is Side.BUY


def test_make_decision_keeps_momentum_behavior_for_other_ids():
    momentum = StrategyVersion(
        strategy_id="momentum", version="1", hypothesis="test",
        parameters={"entry_return": 0.001},
    )
    decision = make_decision(momentum, state_with_latest_return(0.005))
    assert decision.action is DecisionAction.ENTER
    assert decision.side is Side.BUY
