from datetime import UTC, datetime

import pytest

from atlab.ledger import ImmutableLedger
from atlab.models import DecisionAction, MarketObservation, StrategyVersion
from atlab.paper import PaperExecution
from atlab.replay import replay
from atlab.risk import DeterministicRiskEngine, RiskLimits
from atlab.state import build_state
from atlab.strategy import make_decision


def obs(price: float, second: int) -> MarketObservation:
    return MarketObservation(
        symbol="TEST",
        timestamp=datetime.fromtimestamp(second, tz=UTC),
        price=price,
        source="test",
    )


def test_state_is_deterministic():
    assert build_state([obs(100, 1), obs(101, 2)]).state_fingerprint == build_state(
        [obs(100, 1), obs(101, 2)]
    ).state_fingerprint


def test_state_has_no_future_leakage():
    state = build_state(
        [obs(100, 1), obs(101, 2), obs(500, 3)],
        as_of=datetime.fromtimestamp(2, tz=UTC),
    )
    assert state.price == 101


def test_strategy_is_versioned_and_deterministic():
    state = build_state([obs(100, 1), obs(101, 2)])
    strategy = StrategyVersion(
        strategy_id="momentum",
        version="1.0.0",
        hypothesis="test",
        parameters={"entry_return": 0.001},
    )
    a, b = make_decision(strategy, state), make_decision(strategy, state)
    assert (
        a.strategy_version == "1.0.0"
        and a.action is DecisionAction.ENTER
        and a.decision_id == b.decision_id
    )


def test_risk_rejects_large_order():
    state = build_state([obs(100, 1), obs(101, 2)])
    strategy = StrategyVersion(
        strategy_id="momentum",
        version="1",
        hypothesis="test",
        parameters={"entry_return": 0.001},
    )
    decision = make_decision(strategy, state)
    result = DeterministicRiskEngine(RiskLimits(max_order_notional=50)).evaluate(
        decision, 101, 1
    )
    assert not result.approved


def test_kill_switch_blocks_paper_execution():
    state = build_state([obs(100, 1), obs(101, 2)])
    strategy = StrategyVersion(
        strategy_id="momentum",
        version="1",
        hypothesis="test",
        parameters={"entry_return": 0.001},
    )
    decision = make_decision(strategy, state)
    with pytest.raises(RuntimeError, match="KILL_SWITCH_ACTIVE"):
        PaperExecution(kill_switch=True).submit(decision, 1, 101)


def test_ledger_replay(tmp_path):
    ledger = ImmutableLedger(tmp_path / "ledger.jsonl")
    ledger.append("DECISION", "d1", {"action": "ENTER"})
    ledger.append("ORDER", "o1", {"status": "FILLED"})
    assert replay(ledger.read()) == ["d1", "o1"]
