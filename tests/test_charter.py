"""Tests for Wendy's risk charter v1: the $10/$40/$15/$50 cage."""

from datetime import UTC, datetime

import pytest

from atlab.adapters import InMemoryMarketData
from atlab.charter import (
    CHARTER_ID,
    CHARTER_MAX_DAILY_LOSS,
    CHARTER_MAX_DRAWDOWN,
    CHARTER_MAX_LEVERAGE,
    CHARTER_MAX_ORDER_NOTIONAL,
    CHARTER_MAX_POSITION_NOTIONAL,
    RiskCharter,
    charter_limits,
    charter_quantity_policy,
)
from atlab.ledger import ImmutableLedger
from atlab.models import DecisionAction, JEVDecision, MarketObservation, Side, StrategyVersion
from atlab.portfolio import PaperPortfolio
from atlab.risk import DeterministicRiskEngine
from atlab.runtime import PaperTradingEngine, TradingMode
from atlab.sniper import SNIPER_STRATEGY_ID


def buy_decision():
    return JEVDecision(
        decision_id="charter-d1",
        strategy_id="sniper-mean-reversion",
        strategy_version="1.0.0",
        symbol="TEST",
        action=DecisionAction.ENTER,
        side=Side.BUY,
        confidence=0.8,
        rationale="test",
        state_fingerprint="state",
        created_at=datetime.now(UTC),
    )


def test_charter_numbers_match_the_agreed_cage():
    assert CHARTER_MAX_ORDER_NOTIONAL == 10.0
    assert CHARTER_MAX_POSITION_NOTIONAL == 40.0
    assert CHARTER_MAX_DAILY_LOSS == 15.0
    assert CHARTER_MAX_DRAWDOWN == 50.0
    assert CHARTER_MAX_LEVERAGE == 1.0


def test_charter_fingerprint_is_stable_and_sensitive():
    first = RiskCharter().fingerprint()
    assert first == RiskCharter().fingerprint()
    altered = RiskCharter(max_order_notional=11.0).fingerprint()
    assert altered != first


def test_charter_limits_map_to_the_risk_engine():
    limits = charter_limits()
    assert limits.max_order_notional == 10.0
    assert limits.max_position_notional == 40.0
    assert limits.max_daily_loss == 15.0
    assert limits.max_drawdown == 50.0
    assert limits.max_leverage == 1.0


def test_charter_blocks_any_trade_over_ten_dollars():
    engine = DeterministicRiskEngine(charter_limits())
    blocked = engine.evaluate(
        buy_decision(), 100.0, 0.1001,  # $10.01 notional
        equity=200.0, available_cash=200.0,
        session_start_equity=200.0, high_water_mark=200.0,
    )
    assert not blocked.approved
    assert blocked.reason == "ORDER_NOTIONAL_LIMIT"
    allowed = engine.evaluate(
        buy_decision(), 100.0, 0.10,  # exactly $10.00
        equity=200.0, available_cash=200.0,
        session_start_equity=200.0, high_water_mark=200.0,
    )
    assert allowed.approved


def test_charter_blocks_deployment_beyond_forty_dollars():
    engine = DeterministicRiskEngine(charter_limits())
    result = engine.evaluate(
        buy_decision(), 100.0, 0.10,
        current_position_notional=35.0,  # $35 + $10 would be $45
        equity=200.0, available_cash=200.0,
        session_start_equity=200.0, high_water_mark=200.0,
    )
    assert not result.approved
    assert result.reason == "POSITION_NOTIONAL_LIMIT"


def test_charter_halts_after_fifteen_dollar_daily_loss():
    engine = DeterministicRiskEngine(charter_limits())
    result = engine.evaluate(
        buy_decision(), 100.0, 0.10,
        equity=184.99,  # $15.01 below the session start
        available_cash=180.0,
        session_start_equity=200.0, high_water_mark=200.0,
    )
    assert not result.approved
    assert result.reason == "DAILY_LOSS_LIMIT"


def test_charter_halts_after_fifty_dollar_total_drawdown():
    engine = DeterministicRiskEngine(charter_limits())
    result = engine.evaluate(
        buy_decision(), 100.0, 0.10,
        equity=149.99,  # $50.01 below the high-water mark
        available_cash=140.0,
        session_start_equity=120.0, high_water_mark=200.0,
    )
    assert not result.approved
    assert result.reason == "DRAWDOWN_LIMIT"


def test_charter_quantity_policy_sizes_every_order_to_ten_dollars():
    policy = charter_quantity_policy()
    assert policy(50.0) == pytest.approx(0.2)
    assert policy(5.0) == pytest.approx(2.0)
    assert policy(10.0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="INVALID_PRICE_FOR_QUANTITY_POLICY"):
        policy(0.0)


def test_charter_id_is_versioned():
    assert CHARTER_ID == "wendy-risk-charter-v1"
    assert RiskCharter().charter_id == CHARTER_ID
    assert RiskCharter().stop_loss == 5.0


def _dip_observations():
    # Alternating dips and flat bars so the sniper fires repeatedly.
    prices = [100.0]
    for index in range(1, 30):
        prices.append(prices[-1] * (0.994 if index % 2 else 1.0))
    return [
        MarketObservation(
            symbol="TEST",
            timestamp=datetime.fromtimestamp(index, tz=UTC),
            price=price,
            source="test",
        )
        for index, price in enumerate(prices)
    ]


def test_paper_run_under_charter_sizes_every_order_to_ten_dollars(tmp_path):
    strategy = StrategyVersion(
        strategy_id=SNIPER_STRATEGY_ID,
        version="1.0.0",
        hypothesis="test",
        parameters={"entry_dip": 0.004, "exit_bounce": 0.003},
    )
    engine = PaperTradingEngine(
        InMemoryMarketData(_dip_observations()),
        strategy,
        DeterministicRiskEngine(charter_limits()),
        PaperPortfolio(200.0),
        ImmutableLedger(tmp_path / "ledger.sqlite3"),
        quantity_policy=charter_quantity_policy(),
        mode=TradingMode.PAPER,
    )
    results = engine.run("TEST")
    orders = [cycle.order for cycle in results if cycle.order is not None]
    assert orders, "expected the sniper to trade on repeated dips"
    for order in orders:
        assert order.notional == pytest.approx(10.0, rel=1e-9)
    # Position cap: never more than $40 deployed at once.
    position_value = 0.0
    for cycle in results:
        order = cycle.order
        if order is not None:
            signed = order.notional if order.side is Side.BUY else -order.notional
            position_value += signed
        assert position_value <= 40.0 + 1e-9


def _crash_observations():
    # Flat, then a hard -3%/bar slide: the stop-loss must cut the position.
    prices = [100.0] * 10 + [100.0 * (0.97 ** i) for i in range(1, 25)]
    return [
        MarketObservation(
            symbol="TEST",
            timestamp=datetime.fromtimestamp(index, tz=UTC),
            price=price,
            source="test",
        )
        for index, price in enumerate(prices)
    ]


def _sniper_strategy():
    return StrategyVersion(
        strategy_id=SNIPER_STRATEGY_ID,
        version="1.0.0",
        hypothesis="test",
        parameters={"entry_dip": 0.004, "exit_bounce": 0.003},
    )


def _charter_engine(observations, tmp_path, stop_loss):
    return PaperTradingEngine(
        InMemoryMarketData(observations),
        _sniper_strategy(),
        DeterministicRiskEngine(charter_limits()),
        PaperPortfolio(200.0),
        ImmutableLedger(tmp_path / "ledger.sqlite3"),
        quantity_policy=charter_quantity_policy(),
        stop_loss=stop_loss,
        mode=TradingMode.PAPER,
    )


def test_stop_loss_cuts_a_bleeding_position(tmp_path):
    engine = _charter_engine(_crash_observations(), tmp_path, stop_loss=5.0)
    results = engine.run("TEST")
    stop_trades = [
        cycle.order for cycle in results
        if cycle.order is not None and "Stop-loss" in cycle.decision.rationale
    ]
    assert stop_trades, "expected the stop-loss to fire in a hard slide"
    for order in stop_trades:
        assert order.side is Side.SELL
    # The position must be flat after the stop-loss exits: the latch keeps
    # selling until nothing remains, so no stub is stranded.
    assert engine.portfolio.position_quantity == pytest.approx(0.0)
    # Even in a 24-bar -3%/bar crash, the realized loss stays inside the
    # $15 daily-loss gate: the stop-loss cuts each lot and the halt stops
    # new entries.
    assert engine.portfolio.realized_pnl > -15.0


def test_no_stop_loss_without_configuration(tmp_path):
    engine = _charter_engine(_crash_observations(), tmp_path, stop_loss=None)
    results = engine.run("TEST")
    rationales = [cycle.decision.rationale for cycle in results]
    assert not any("Stop-loss" in rationale for rationale in rationales)


def test_stop_loss_decision_is_deterministic(tmp_path):
    first = _charter_engine(_crash_observations(), tmp_path / "a", stop_loss=5.0)
    second = _charter_engine(_crash_observations(), tmp_path / "b", stop_loss=5.0)
    first_ids = [
        cycle.decision.decision_id for cycle in first.run("TEST")
        if "Stop-loss" in cycle.decision.rationale
    ]
    second_ids = [
        cycle.decision.decision_id for cycle in second.run("TEST")
        if "Stop-loss" in cycle.decision.rationale
    ]
    assert first_ids and first_ids == second_ids


def test_invalid_stop_loss_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="INVALID_STOP_LOSS"):
        _charter_engine(_crash_observations(), tmp_path, stop_loss=0.0)
