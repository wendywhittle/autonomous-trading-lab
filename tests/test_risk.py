from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from atlab.models import DecisionAction, JEVDecision, RiskDecision, Side
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


# --- H13: reason-asserting tests for every risk-engine deny branch. ---

def test_sell_is_blocked_when_position_is_insufficient():
    result = DeterministicRiskEngine().evaluate(
        decision(side=Side.SELL), 100, 10, current_position_notional=500,
    )
    assert not result.approved
    assert result.reason == "INSUFFICIENT_POSITION"


def test_position_notional_limit_is_enforced():
    engine = DeterministicRiskEngine(
        RiskLimits(max_order_notional=100_000, max_position_notional=1_000)
    )
    result = engine.evaluate(
        decision(), 100, 20, current_position_notional=500,
        equity=100_000, available_cash=100_000,
    )
    assert not result.approved
    assert result.reason == "POSITION_NOTIONAL_LIMIT"


def test_zero_equity_is_blocked():
    result = DeterministicRiskEngine().evaluate(
        decision(), 100, 1, equity=0, available_cash=10_000
    )
    assert not result.approved
    assert result.reason == "ZERO_EQUITY"


def test_invalid_order_size_is_blocked():
    # Non-positive price/quantity never produce an approval: pydantic's
    # gt=0 boundaries reject them before the INVALID_ORDER_SIZE denial
    # reason can be materialized. Fail-closed either way.
    for price, quantity in ((0, 1), (100, 0), (-5, 1), (100, -2)):
        with pytest.raises(ValidationError):
            DeterministicRiskEngine().evaluate(decision(), price, quantity)


def test_invalid_position_is_blocked():
    result = DeterministicRiskEngine().evaluate(
        decision(), 100, 1, current_position_notional=-1
    )
    assert not result.approved
    assert result.reason == "INVALID_POSITION"


def test_invalid_available_cash_is_blocked():
    result = DeterministicRiskEngine().evaluate(decision(), 100, 1, available_cash=-1)
    assert not result.approved
    assert result.reason == "INVALID_AVAILABLE_CASH"


def test_invalid_equity_inputs_are_blocked():
    assert DeterministicRiskEngine().evaluate(decision(), 100, 1, equity=-1).reason == "INVALID_EQUITY"
    assert DeterministicRiskEngine().evaluate(
        decision(), 100, 1, equity=100, session_start_equity=-1
    ).reason == "INVALID_EQUITY"
    assert DeterministicRiskEngine().evaluate(
        decision(), 100, 1, equity=100, high_water_mark=-1
    ).reason == "INVALID_EQUITY"


def test_daily_loss_limit_is_enforced():
    result = DeterministicRiskEngine().evaluate(
        decision(), 100, 1, equity=9_000, session_start_equity=10_000,
        available_cash=10_000,
    )
    assert not result.approved
    assert result.reason == "DAILY_LOSS_LIMIT"


def test_drawdown_limit_is_enforced():
    result = DeterministicRiskEngine().evaluate(
        decision(), 100, 1, equity=8_500, high_water_mark=10_000,
        available_cash=10_000,
    )
    assert not result.approved
    assert result.reason == "DRAWDOWN_LIMIT"


def test_hold_decision_is_not_executable():
    hold_decision = decision().model_copy(
        update={"action": DecisionAction.HOLD, "side": None}
    )
    result = DeterministicRiskEngine().evaluate(hold_decision, 100, 1)
    assert not result.approved
    assert result.reason == "HOLD_DECISION"


def test_decision_side_mismatch_is_blocked():
    wrong_side = decision().model_copy(update={"side": Side.SELL})
    result = DeterministicRiskEngine().evaluate(wrong_side, 100, 1)
    assert not result.approved
    assert result.reason == "INVALID_DECISION_SIDE"


def test_missing_decision_side_is_blocked():
    no_side = decision().model_copy(update={"side": None})
    result = DeterministicRiskEngine().evaluate(no_side, 100, 1)
    assert not result.approved
    assert result.reason == "INVALID_DECISION_SIDE"


def test_order_notional_limit_is_enforced():
    result = DeterministicRiskEngine().evaluate(decision(), 100, 100)
    assert not result.approved
    assert result.reason == "ORDER_NOTIONAL_LIMIT"


def test_risk_state_input_mismatch_is_rejected():
    engine = DeterministicRiskEngine()
    state = RiskStateSnapshot(
        symbol="TEST",
        current_position_notional=0,
        equity=10_000,
        session_start_equity=10_000,
        high_water_mark=10_000,
        available_cash=10_000,
    )
    with pytest.raises(ValueError, match="RISK_STATE_INPUT_MISMATCH"):
        engine.evaluate(decision(), 100, 1, equity=9_999, risk_state=state)


def test_risk_evidence_incomplete_is_rejected():
    signed = DeterministicRiskEngine().evaluate(decision(), 100, 1)
    tampered = signed.model_copy(update={"quantity": None})
    with pytest.raises(ValueError, match="RISK_EVIDENCE_INCOMPLETE"):
        DeterministicRiskEngine.fingerprint_for(tampered)


def test_risk_evidence_version_conflict_is_rejected():
    signed = DeterministicRiskEngine().evaluate(decision(), 100, 1)
    wrong_version = signed.model_copy(update={"engine_version": "0.0.0"})
    with pytest.raises(ValueError, match="RISK_ENGINE_VERSION_CONFLICT"):
        DeterministicRiskEngine.fingerprint_for(wrong_version)


def test_unapproved_risk_evidence_is_rejected():
    denied = DeterministicRiskEngine().evaluate(
        decision(), 100, 1, equity=0, available_cash=10_000
    )
    assert denied.reason == "ZERO_EQUITY"
    with pytest.raises(RuntimeError, match="RISK_DECISION_NOT_APPROVED"):
        DeterministicRiskEngine.validate_evidence(denied)


def test_missing_risk_fingerprint_is_rejected():
    unsigned = RiskDecision(
        approved=True, reason="APPROVED",
        engine_version=DeterministicRiskEngine.ENGINE_VERSION,
        max_notional=2500, decision_id="d1",
        action=DecisionAction.ENTER, symbol="TEST", side=Side.BUY,
        quantity=1, price=100, current_position_notional=0,
    )
    with pytest.raises(RuntimeError, match="RISK_EVIDENCE_MISSING"):
        DeterministicRiskEngine.validate_evidence(unsigned)


def test_tampered_risk_evidence_is_rejected():
    signed = DeterministicRiskEngine().evaluate(decision(), 100, 1)
    tampered = signed.model_copy(update={"price": 101})
    with pytest.raises(RuntimeError, match="RISK_EVIDENCE_TAMPERED"):
        DeterministicRiskEngine.validate_evidence(tampered)


def sell_decision():
    return JEVDecision(
        decision_id="d-sell",
        strategy_id="s",
        strategy_version="1",
        symbol="TEST",
        action=DecisionAction.EXIT,
        side=Side.SELL,
        confidence=1,
        rationale="test",
        state_fingerprint="state",
        created_at=datetime.now(UTC),
    )


def test_loss_halts_never_block_exits():
    # A SELL only reduces exposure, so the daily-loss and drawdown halts
    # must not trap the system in a bleeding position -- the stop-loss exit
    # has to get through even while new entries are halted.
    engine = DeterministicRiskEngine()
    daily = engine.evaluate(
        sell_decision(), 100, 1, current_position_notional=100,
        equity=9_000, session_start_equity=10_000, high_water_mark=10_000,
        available_cash=10_000,
    )
    assert daily.approved, daily.reason
    drawdown = engine.evaluate(
        sell_decision(), 100, 1, current_position_notional=100,
        equity=8_500, session_start_equity=10_000, high_water_mark=10_000,
        available_cash=10_000,
    )
    assert drawdown.approved, drawdown.reason


def test_loss_halts_still_block_new_entries():
    engine = DeterministicRiskEngine()
    assert engine.evaluate(
        decision(), 100, 1,
        equity=9_000, session_start_equity=10_000, available_cash=10_000,
    ).reason == "DAILY_LOSS_LIMIT"
    assert engine.evaluate(
        decision(), 100, 1,
        equity=8_500, high_water_mark=10_000, available_cash=10_000,
    ).reason == "DRAWDOWN_LIMIT"
