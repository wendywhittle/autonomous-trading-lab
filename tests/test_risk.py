from datetime import UTC, datetime

from atlab.models import DecisionAction, JEVDecision, Side
from atlab.risk import DeterministicRiskEngine, RiskLimits


def decision(side=Side.BUY):
    return JEVDecision(
        decision_id="d1",
        strategy_id="s",
        strategy_version="1",
        symbol="TEST",
        action=DecisionAction.ENTER if side is Side.BUY else DecisionAction.EXIT,
        side=side,
        confidence=1,
        rationale="test",
        state_fingerprint="state",
        created_at=datetime.now(UTC),
    )


def test_buy_is_blocked_when_cash_is_insufficient():
    result = DeterministicRiskEngine().evaluate(decision(), 100, 20, available_cash=1_999)
    assert not result.approved
    assert result.reason == "INSUFFICIENT_CASH"


def test_buy_is_blocked_when_leverage_would_be_exceeded():
    result = DeterministicRiskEngine(
        RiskLimits(max_order_notional=10_000, max_position_notional=10_000, max_leverage=1.0)
    ).evaluate(
        decision(), 100, 60, current_position_notional=4_500,
        equity=5_000, available_cash=10_000,
    )
    assert not result.approved
    assert result.reason == "LEVERAGE_LIMIT"


def test_cash_and_leverage_allow_valid_buy():
    result = DeterministicRiskEngine().evaluate(
        decision(), 100, 20, equity=5_000, available_cash=5_000
    )
    assert result.approved


def test_invalid_cash_and_equity_are_blocked():
    assert not DeterministicRiskEngine().evaluate(
        decision(), 100, 1, available_cash=-1
    ).approved
    assert not DeterministicRiskEngine().evaluate(
        decision(), 100, 1, equity=-1
    ).approved
