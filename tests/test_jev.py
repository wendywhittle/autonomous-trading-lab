import pytest

from atlab.jev import UnconfiguredJEV
from atlab.models import JEVDecision, DecisionAction, MarketState, Side, StrategyVersion


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
