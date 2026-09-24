from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from atlab.models import DecisionAction, JEVDecision, Side
from atlab.risk import DeterministicRiskEngine, RiskLimits
from atlab.risk_state import RiskStateSnapshot


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


def test_risk_evidence_exposes_explicit_engine_version_and_is_deterministic():
    engine = DeterministicRiskEngine()
    result = engine.evaluate(decision(), 100, 1)
    assert result.engine_version == DeterministicRiskEngine.ENGINE_VERSION
    assert result.risk_fingerprint == DeterministicRiskEngine.fingerprint_for(result)
    assert result.risk_fingerprint == DeterministicRiskEngine.fingerprint_for(result)


def test_risk_evidence_engine_version_is_immutable_and_tampering_is_rejected():
    result = DeterministicRiskEngine().evaluate(decision(), 100, 1)
    with pytest.raises(ValueError, match="RISK_ENGINE_VERSION_CONFLICT"):
        DeterministicRiskEngine.validate_evidence(
            result.model_copy(update={"engine_version": "tampered"})
        )
    with pytest.raises(ValidationError):
        result.engine_version = "tampered"


def test_risk_evidence_fields_are_immutable():
    result = DeterministicRiskEngine().evaluate(
        decision(), 100, 1, current_position_notional=10,
        equity=1_000, session_start_equity=1_000,
        high_water_mark=1_000, available_cash=900,
        risk_state=RiskStateSnapshot(
            symbol="TEST",
            current_position_notional=10,
            equity=1_000,
            session_start_equity=1_000,
            high_water_mark=1_000,
            available_cash=900,
        ),
    )
    for field, value in (
        ("risk_state_fingerprint", "tampered"),
        ("risk_limits_fingerprint", "tampered"),
        ("risk_fingerprint", "tampered"),
        ("quantity", 2),
        ("price", 101),
        ("symbol", "OTHER"),
        ("side", Side.SELL),
        ("engine_version", "tampered"),
    ):
        with __import__("pytest").raises(Exception):
            result.__setattr__(field, value)
