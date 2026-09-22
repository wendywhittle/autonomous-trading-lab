from datetime import UTC, datetime

import pytest

from atlab.adapters import InMemoryMarketData
from atlab.ledger import ImmutableLedger
from atlab.models import MarketObservation, StrategyVersion
from atlab.portfolio import PaperPortfolio
from atlab.risk import DeterministicRiskEngine, RiskLimits
from atlab.runtime import PaperTradingEngine, TradingMode


def obs(price, second):
    return MarketObservation(
        symbol="TEST",
        timestamp=datetime.fromtimestamp(second, tz=UTC),
        price=price,
        source="test",
    )


def strategy():
    return StrategyVersion(
        strategy_id="momentum",
        version="1.0.0",
        hypothesis="test",
        parameters={"entry_return": 0.001},
    )


def test_paper_engine_runs_end_to_end(tmp_path):
    adapter = InMemoryMarketData([obs(100, 1), obs(101, 2), obs(102, 3)])
    engine = PaperTradingEngine(
        adapter,
        strategy(),
        DeterministicRiskEngine(),
        PaperPortfolio(1000),
        ImmutableLedger(tmp_path / "ledger.jsonl"),
        quantity=2,
    )

    results = engine.run("TEST")

    assert len(results) == 2
    assert all(item.order is not None for item in results)
    assert results[-1].equity == 1002
    assert results[-1].order.fill_price == 102
    assert [event.event_type for event in engine.ledger.read()] == [
        "DECISION",
        "ORDER",
        "DECISION",
        "ORDER",
    ]


def test_paper_engine_is_repeatable(tmp_path):
    observations = [obs(100, 1), obs(101, 2), obs(102, 3)]
    results = []
    for name in ("a", "b"):
        engine = PaperTradingEngine(
            InMemoryMarketData(observations),
            strategy(),
            DeterministicRiskEngine(),
            PaperPortfolio(1000),
            ImmutableLedger(tmp_path / f"{name}.jsonl"),
        )
        results.append(engine.run("TEST"))

    assert [item.decision.decision_id for item in results[0]] == [
        item.decision.decision_id for item in results[1]
    ]
    assert [item.risk_reason for item in results[0]] == [
        item.risk_reason for item in results[1]
    ]


def test_paper_engine_rejects_non_paper_modes(tmp_path):
    with pytest.raises(ValueError, match="PAPER_ENGINE_REQUIRES_PAPER_MODE"):
        PaperTradingEngine(
            InMemoryMarketData([obs(100, 1), obs(101, 2)]),
            strategy(),
            DeterministicRiskEngine(),
            PaperPortfolio(1000),
            ImmutableLedger(tmp_path / "ledger.jsonl"),
            mode=TradingMode.LIVE_DISABLED,
        )


def test_risk_sell_reduces_exposure():
    from atlab.models import DecisionAction, JEVDecision, Side

    decision = JEVDecision(
        decision_id="exit-1",
        strategy_id="momentum",
        strategy_version="1.0.0",
        symbol="TEST",
        action=DecisionAction.EXIT,
        side=Side.SELL,
        confidence=1,
        rationale="test",
        state_fingerprint="state",
    )
    result = DeterministicRiskEngine(
        RiskLimits(max_position_notional=200, max_order_notional=200)
    ).evaluate(decision, 100, 1, current_position_notional=150)

    assert result.approved


def test_risk_blocks_daily_loss_and_drawdown():
    from atlab.models import DecisionAction, JEVDecision, Side

    decision = JEVDecision(
        decision_id="risk-1",
        strategy_id="momentum",
        strategy_version="1.0.0",
        symbol="TEST",
        action=DecisionAction.ENTER,
        side=Side.BUY,
        confidence=1,
        rationale="test",
        state_fingerprint="state",
    )
    engine = DeterministicRiskEngine(
        RiskLimits(max_daily_loss=50, max_drawdown=100)
    )
    daily = engine.evaluate(
        decision, 100, 1, equity=940, session_start_equity=1000, high_water_mark=1000
    )
    assert not daily.approved
    assert daily.reason == "DAILY_LOSS_LIMIT"

    drawdown = engine.evaluate(
        decision, 100, 1, equity=890, session_start_equity=900, high_water_mark=1000
    )
    assert not drawdown.approved
    assert drawdown.reason == "DRAWDOWN_LIMIT"


def test_risk_limits_are_strict_boundaries():
    from atlab.models import DecisionAction, JEVDecision, Side

    decision = JEVDecision(
        decision_id="risk-2",
        strategy_id="momentum",
        strategy_version="1.0.0",
        symbol="TEST",
        action=DecisionAction.ENTER,
        side=Side.BUY,
        confidence=1,
        rationale="test",
        state_fingerprint="state",
    )
    engine = DeterministicRiskEngine(
        RiskLimits(max_daily_loss=50, max_drawdown=100, max_order_notional=200)
    )
    result = engine.evaluate(
        decision, 100, 1, equity=950, session_start_equity=1000, high_water_mark=1000
    )
    assert result.approved
