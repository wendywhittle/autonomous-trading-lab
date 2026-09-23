from datetime import UTC, datetime

import pytest

from atlab.adapters import InMemoryMarketData
from atlab.ledger import ImmutableLedger
from atlab.models import MarketObservation, StrategyVersion
from atlab.portfolio import PaperPortfolio
from atlab.risk import DeterministicRiskEngine, RiskLimits
from atlab.runtime import PaperTradingEngine, TradingMode
from atlab.state import build_state
from atlab.strategy import make_decision


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
        ImmutableLedger(tmp_path / "ledger.sqlite3"),
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
            ImmutableLedger(tmp_path / f"{name}.sqlite3"),
        )
        results.append(engine.run("TEST"))

    assert [item.decision.decision_id for item in results[0]] == [
        item.decision.decision_id for item in results[1]
    ]
    assert [item.risk_reason for item in results[0]] == [
        item.risk_reason for item in results[1]
    ]


class CountingJEV:
    def __init__(self):
        self.calls = 0

    def evaluate(self, strategy, state, proposal):
        self.calls += 1
        return proposal.model_copy(
            update={
                "decision_id": f"jev-eval-{self.calls}",
                "confidence": 0.9,
                "rationale": f"evaluation-{self.calls}",
            }
        )


def test_paper_engine_persists_jev_evaluation_across_restart(tmp_path):
    observations = [obs(100, 1), obs(101, 2), obs(102, 3)]
    ledger = ImmutableLedger(tmp_path / "ledger.sqlite3")
    jev = CountingJEV()
    engine = PaperTradingEngine(
        InMemoryMarketData(observations),
        strategy(),
        DeterministicRiskEngine(),
        PaperPortfolio(1000),
        ledger,
        jev=jev,
        portfolio_state_path=tmp_path / "portfolio.json",
        risk_state_path=tmp_path / "risk.json",
    )

    first = engine.run("TEST")
    assert len(first) == 2
    assert jev.calls == 2
    assert len([e for e in ledger.read() if e.event_type == "JEV_EVALUATION"]) == 2

    restarted_jev = CountingJEV()
    restarted = PaperTradingEngine(
        InMemoryMarketData(observations),
        strategy(),
        DeterministicRiskEngine(),
        PaperPortfolio(0),
        ledger,
        jev=restarted_jev,
        portfolio_state_path=tmp_path / "portfolio.json",
        risk_state_path=tmp_path / "risk.json",
    )
    assert restarted.run("TEST") == ()
    assert restarted_jev.calls == 0


def test_paper_engine_rejects_non_paper_modes(tmp_path):
    with pytest.raises(ValueError, match="PAPER_ENGINE_REQUIRES_PAPER_MODE"):
        PaperTradingEngine(
            InMemoryMarketData([obs(100, 1), obs(101, 2)]),
            strategy(),
            DeterministicRiskEngine(),
            PaperPortfolio(1000),
            ImmutableLedger(tmp_path / "ledger.sqlite3"),
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


def test_paper_engine_recovers_incomplete_decision_after_restart(tmp_path):
    observations = [obs(100, 1), obs(101, 2), obs(102, 3)]
    ledger = ImmutableLedger(tmp_path / "ledger.sqlite3")
    state = build_state(observations[:2], as_of=observations[1].timestamp)
    decision = make_decision(strategy(), state)
    ledger.append("DECISION", decision.decision_id, decision.model_dump(mode="json"))

    portfolio_path = tmp_path / "portfolio.json"
    risk_path = tmp_path / "risk.json"
    engine = PaperTradingEngine(
        InMemoryMarketData(observations),
        strategy(),
        DeterministicRiskEngine(),
        PaperPortfolio(1000),
        ledger,
        portfolio_state_path=portfolio_path,
        risk_state_path=risk_path,
    )

    results = engine.run("TEST")

    assert len(results) == 2
    assert results[0].order is not None
    assert engine.portfolio.position_quantity == 2
    assert len(ledger.read()) == 4

    restarted = PaperTradingEngine(
        InMemoryMarketData(observations),
        strategy(),
        DeterministicRiskEngine(),
        PaperPortfolio(0),
        ledger,
        portfolio_state_path=portfolio_path,
        risk_state_path=risk_path,
    )
    assert restarted.run("TEST") == ()
    assert restarted.portfolio.position_quantity == 2
    assert restarted.portfolio.cash == 797
