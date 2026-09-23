import os

import pytest

from atlab.jev import TypeSafeJEV
from atlab.models import DecisionAction, JEVDecision, MarketState, Side, StrategyVersion


@pytest.mark.skipif(
    not os.getenv("TYPESAFE_API_KEY"),
    reason="TYPESAFE_API_KEY is not configured",
)
def test_live_typesafe_jev_smoke():
    strategy = StrategyVersion(
        strategy_id="m0-smoke",
        version="1",
        hypothesis="Prefer entering when the proposal is explicitly supported by the supplied state.",
    )
    state = MarketState(
        symbol="SMOKE",
        as_of="2026-01-01T00:00:00Z",
        price=100.0,
        returns=[0.01, 0.02, 0.015],
        regime="UP",
        state_fingerprint="smoke-state-v1",
    )
    proposal = JEVDecision(
        decision_id="smoke-proposal-1",
        strategy_id=strategy.strategy_id,
        strategy_version=strategy.version,
        symbol=state.symbol,
        action=DecisionAction.ENTER,
        side=Side.BUY,
        confidence=0.5,
        rationale="Deterministic strategy proposal for live Jev connectivity smoke test.",
        state_fingerprint=state.state_fingerprint,
    )

    decision = TypeSafeJEV().evaluate(strategy, state, proposal)

    assert decision.action in set(DecisionAction)
    assert 0 <= decision.confidence <= 1
    assert decision.state_fingerprint == state.state_fingerprint
