from typing import ClassVar

import pytest

from atlab.jev import UnconfiguredJEV
from atlab.models import DecisionAction, JEVDecision, MarketState, Side, StrategyVersion


def test_jev_boundary_fails_closed_without_real_implementation():
    strategy = StrategyVersion(
        strategy_id="momentum",
        version="1",
        hypothesis="test",
    )
    state = MarketState(
        symbol="TEST",
        as_of="2026-01-01T00:00:00Z",
        price=100,
        state_fingerprint="state-1",
    )
    proposal = JEVDecision(
        decision_id="proposal-1",
        strategy_id=strategy.strategy_id,
        strategy_version=strategy.version,
        symbol=state.symbol,
        action=DecisionAction.ENTER,
        side=Side.BUY,
        confidence=0.5,
        rationale="Strategy proposal only.",
        state_fingerprint=state.state_fingerprint,
    )

    with pytest.raises(RuntimeError, match="JEV_NOT_CONFIGURED"):
        UnconfiguredJEV().evaluate(strategy, state, proposal)


class _Answer:
    choice = "ENTER"
    confidence = 0.91
    probabilities: ClassVar = {"HOLD": 0.04, "ENTER": 0.91, "EXIT": 0.05}


class _Response:
    choices: ClassVar = {"action": _Answer()}


class _Client:
    def __init__(self):
        self.calls = []

    def system_one(self, **kwargs):
        self.calls.append(kwargs)
        return _Response()


def _inputs():
    strategy = StrategyVersion(
        strategy_id="momentum",
        version="1",
        hypothesis="test",
    )
    state = MarketState(
        symbol="TEST",
        as_of="2026-01-01T00:00:00Z",
        price=100,
        state_fingerprint="state-1",
    )
    proposal = JEVDecision(
        decision_id="proposal-1",
        strategy_id=strategy.strategy_id,
        strategy_version=strategy.version,
        symbol=state.symbol,
        action=DecisionAction.ENTER,
        side=Side.BUY,
        confidence=0.5,
        rationale="Strategy proposal only.",
        state_fingerprint=state.state_fingerprint,
    )
    return strategy, state, proposal


def test_real_typesafe_jev_adapter_maps_typed_judgment():
    from atlab.jev import TypeSafeJEV

    strategy, state, proposal = _inputs()
    client = _Client()

    decision = TypeSafeJEV(client=client).evaluate(strategy, state, proposal)

    assert decision.action is DecisionAction.ENTER
    assert decision.confidence == 0.91
    assert decision.decision_id.startswith("jev-")
    assert "TypeSafe Jev action=ENTER" in decision.rationale
    assert client.calls[0]["questions"]["action"].criteria["ENTER"] == "Open the proposed position."
    assert client.calls[0]["state"]["market_state"]["state_fingerprint"] == "state-1"
