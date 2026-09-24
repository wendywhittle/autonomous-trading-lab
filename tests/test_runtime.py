from datetime import UTC, datetime
from typing import ClassVar

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
        # H6: the risk outcome is persisted before execution so a crash
        # later in the cycle can resume from the recorded outcome.
        "RISK_EVALUATION",
        "DECISION",
        "ORDER",
        "RISK_EVALUATION",
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
    # 1 pre-existing DECISION + (RISK_EVALUATION + ORDER) for cycle 1 +
    # (RISK_EVALUATION + DECISION + ORDER) for cycle 2.
    assert len(ledger.read()) == 6

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

    
class MalformedJEV:
    def __init__(self, **updates):
        self.updates = updates

    def evaluate(self, strategy, state, proposal):
        return proposal.model_copy(update=self.updates)


def test_paper_engine_rejects_jev_identity_tampering(tmp_path):
    observations = [obs(100, 1), obs(101, 2)]
    for updates, error in (
        ({"symbol": "EVIL"}, "JEV_DECISION_SYMBOL_MISMATCH"),
        ({"strategy_id": "other"}, "JEV_DECISION_STRATEGY_MISMATCH"),
        ({"strategy_version": "9.9.9"}, "JEV_DECISION_STRATEGY_VERSION_MISMATCH"),
        ({"state_fingerprint": "forged"}, "JEV_DECISION_STATE_MISMATCH"),
    ):
        engine = PaperTradingEngine(
            InMemoryMarketData(observations),
            strategy(),
            DeterministicRiskEngine(),
            PaperPortfolio(1000),
            ImmutableLedger(tmp_path / f"{error}.sqlite3"),
            jev=MalformedJEV(**updates),
        )
        with pytest.raises(RuntimeError, match=error):
            engine.run("TEST")
        assert engine.ledger.read() == []


def test_paper_engine_rejects_jev_side_tampering(tmp_path):
    observations = [obs(100, 1), obs(101, 2)]
    engine = PaperTradingEngine(
        InMemoryMarketData(observations),
        strategy(),
        DeterministicRiskEngine(),
        PaperPortfolio(1000),
        ImmutableLedger(tmp_path / "ledger.sqlite3"),
        jev=MalformedJEV(side="SELL"),
    )

    with pytest.raises(RuntimeError, match="JEV_DECISION_SIDE_MISMATCH"):
        engine.run("TEST")
    assert engine.ledger.read() == []


def test_paper_engine_rejects_corrupt_persisted_jev_evaluation(tmp_path):
    observations = [obs(100, 1), obs(101, 2)]
    ledger = ImmutableLedger(tmp_path / "ledger.sqlite3")
    state = build_state(observations, as_of=observations[1].timestamp)
    proposal = make_decision(strategy(), state)
    corrupted = proposal.model_copy(
        update={"decision_id": "jev-corrupt", "symbol": "EVIL"}
    )
    ledger.append(
        "JEV_EVALUATION",
        f"jev-{proposal.decision_id}",
        {
            "proposal_id": proposal.decision_id,
            "decision": corrupted.model_dump(mode="json"),
        },
    )

    engine = PaperTradingEngine(
        InMemoryMarketData(observations),
        strategy(),
        DeterministicRiskEngine(),
        PaperPortfolio(1000),
        ledger,
    )

    with pytest.raises(RuntimeError, match="JEV_DECISION_SYMBOL_MISMATCH"):
        engine.run("TEST")
    assert [event.event_type for event in ledger.read()] == ["JEV_EVALUATION"]


def test_jev_exit_side_is_deterministically_sell():
    from atlab.jev import TypeSafeJEV

    class Answer:
        choice: ClassVar = "EXIT"
        confidence: ClassVar = 0.9
        probabilities: ClassVar = {"HOLD": 0.05, "ENTER": 0.05, "EXIT": 0.9}

    class Response:
        choices: ClassVar = {"action": Answer()}

    class Client:
        def system_one(self, **kwargs):
            return Response()

    strategy_value = strategy()
    state = build_state(
        [obs(100, 1), obs(99, 2)],
        as_of=obs(99, 2).timestamp,
    )
    proposal = make_decision(strategy_value, state)
    decision = TypeSafeJEV(client=Client()).evaluate(
        strategy_value, state, proposal
    )
    assert decision.action.value == "EXIT"
    assert decision.side.value == "SELL"


def test_risk_engine_rejects_action_side_mismatch():
    from atlab.models import DecisionAction, JEVDecision, Side

    decision = JEVDecision(
        decision_id="invalid-side",
        strategy_id="momentum",
        strategy_version="1.0.0",
        symbol="TEST",
        action=DecisionAction.ENTER,
        side=Side.SELL,
        confidence=1,
        rationale="malformed",
        state_fingerprint="state",
    )
    result = DeterministicRiskEngine().evaluate(decision, 100, 1)
    assert not result.approved
    assert result.reason == "INVALID_DECISION_SIDE"


# --- H1/H7: the kill switch is global; any KILL_SWITCH event halts the loop. ---

def test_engine_halts_when_kill_switch_event_already_present(tmp_path):
    adapter = InMemoryMarketData([obs(100, 1), obs(101, 2), obs(102, 3)])
    ledger = ImmutableLedger(tmp_path / "ledger.sqlite3")
    ledger.append("KILL_SWITCH", "kill-manual-1", {"reason": "operator halt"})
    engine = PaperTradingEngine(
        adapter,
        strategy(),
        DeterministicRiskEngine(),
        PaperPortfolio(1000),
        ledger,
        quantity=2,
    )

    with pytest.raises(RuntimeError, match="KILL_SWITCH_ACTIVE"):
        engine.run("TEST")


def test_engine_halts_at_next_boundary_after_kill_event(tmp_path):
    adapter = InMemoryMarketData([obs(100, 1), obs(101, 2), obs(102, 3)])
    ledger_path = tmp_path / "ledger.sqlite3"
    engine = PaperTradingEngine(
        adapter,
        strategy(),
        DeterministicRiskEngine(),
        PaperPortfolio(1000),
        ImmutableLedger(ledger_path),
        quantity=2,
    )
    assert len(engine.run("TEST")) == 2

    # Operator kill between runs: the next run halts at the first boundary,
    # regardless of which decision the kill was scoped to (H7).
    ImmutableLedger(ledger_path).append(
        "KILL_SWITCH", "kill-manual-2", {"reason": "operator halt"}
    )
    restarted = PaperTradingEngine(
        adapter,
        strategy(),
        DeterministicRiskEngine(),
        PaperPortfolio(1000),
        ImmutableLedger(ledger_path),
        quantity=2,
    )
    with pytest.raises(RuntimeError, match="KILL_SWITCH_ACTIVE"):
        restarted.run("TEST")


def test_armed_kill_switch_halts_engine_and_persists_kill_event(tmp_path):
    # H1: kill_switch=True arms PaperExecution. The first executable decision
    # raises, the engine records the KILL_SWITCH event, and every later cycle
    # (and restart) halts globally.
    adapter = InMemoryMarketData([obs(100, 1), obs(101, 2), obs(102, 3)])
    ledger_path = tmp_path / "ledger.sqlite3"
    engine = PaperTradingEngine(
        adapter,
        strategy(),
        DeterministicRiskEngine(),
        PaperPortfolio(1000),
        ImmutableLedger(ledger_path),
        quantity=2,
        kill_switch=True,
    )

    with pytest.raises(RuntimeError, match="KILL_SWITCH_ACTIVE"):
        engine.run("TEST")

    assert engine.ledger.has_event_type("KILL_SWITCH")

    restarted = PaperTradingEngine(
        adapter,
        strategy(),
        DeterministicRiskEngine(),
        PaperPortfolio(1000),
        ImmutableLedger(ledger_path),
        quantity=2,
    )
    with pytest.raises(RuntimeError, match="KILL_SWITCH_ACTIVE"):
        restarted.run("TEST")


# --- H6: crash between portfolio persist and ORDER append backfills. ---

def test_crash_between_portfolio_persist_and_order_append_backfills(tmp_path):
    from atlab.models import OrderStatus, PaperOrder

    # H6: the fill was applied and the portfolio state persisted, but the
    # process crashed before the ORDER event was appended. On restart the
    # engine backfills the ORDER event from the deterministic order without
    # re-executing or double-applying the fill.
    observations = [obs(100, 1), obs(101, 2)]
    portfolio_path = tmp_path / "portfolio.json"

    # The pre-crash cycle: decide and fill exactly as the engine would.
    pre_crash_state = build_state(observations[:2], as_of=observations[1].timestamp)
    decision = make_decision(strategy(), pre_crash_state)
    assert decision.action.value == "ENTER"
    order = PaperOrder(
        order_id=f"paper-{decision.decision_id}",
        decision_id=decision.decision_id,
        symbol=decision.symbol,
        side=decision.side,
        quantity=2,
        fill_price=101,
        notional=202,
        status=OrderStatus.FILLED,
    )
    crashed_portfolio = PaperPortfolio(1000)
    crashed_portfolio.apply(order)
    crashed_portfolio.save_state(portfolio_path)
    assert crashed_portfolio.has_applied_order(order.order_id)

    # Restart with the persisted portfolio state and a ledger that never saw
    # the ORDER event.
    engine = PaperTradingEngine(
        InMemoryMarketData(observations),
        strategy(),
        DeterministicRiskEngine(),
        PaperPortfolio(1000),
        ImmutableLedger(tmp_path / "ledger.sqlite3"),
        quantity=2,
        portfolio_state_path=portfolio_path,
    )
    results = engine.run("TEST")

    order_events = [e for e in engine.ledger.read() if e.event_type == "ORDER"]
    assert len(order_events) == 1
    assert order_events[0].payload["order_id"] == order.order_id
    assert order_events[0].payload["fill_price"] == 101
    # No double-apply: exactly one fill of 2 @ 101 against 1000 cash.
    assert engine.portfolio.cash == pytest.approx(798)
    assert results[0].order is not None
    assert results[0].order.order_id == order.order_id


def test_engine_blocks_buy_exceeding_available_cash(tmp_path):
    # M3: available_cash now flows into risk evaluation. A BUY larger than
    # cash is blocked with INSUFFICIENT_CASH instead of driving cash
    # negative.
    adapter = InMemoryMarketData([obs(100, 1), obs(101, 2)])
    engine = PaperTradingEngine(
        adapter,
        strategy(),
        DeterministicRiskEngine(),
        PaperPortfolio(100.0),
        ImmutableLedger(tmp_path / "ledger.sqlite3"),
        quantity=2,  # notional 202 > cash 100
    )

    results = engine.run("TEST")

    assert len(results) == 1
    assert results[0].order is None
    assert results[0].risk_reason == "INSUFFICIENT_CASH"
    assert engine.portfolio.cash == 100.0
    assert [event.event_type for event in engine.ledger.read()] == [
        "RISK_EVALUATION",
        "DECISION",
        "RISK_BLOCK",
    ]
