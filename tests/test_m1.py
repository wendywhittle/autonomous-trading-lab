from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from atlab.adapters import InMemoryMarketData
from atlab.backtest import run_backtest
from atlab.models import MarketObservation, StrategyVersion
from atlab.paper import PaperExecution
from atlab.portfolio import PaperPortfolio
from atlab.registry import StrategyRegistry
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


def test_adapter_is_provider_independent():
    adapter = InMemoryMarketData([obs(100, 1), obs(101, 2)])
    assert len(tuple(adapter.observations("TEST"))) == 2


def test_registry_rejects_mutating_existing_version():
    registry = StrategyRegistry()
    registry.register(strategy())
    with pytest.raises(ValueError, match="STRATEGY_VERSION_IMMUTABLE"):
        registry.register(
            StrategyVersion(
                strategy_id="momentum",
                version="1.0.0",
                hypothesis="changed",
            )
        )


def test_backtest_uses_only_available_history():
    result = run_backtest([obs(100, 1), obs(101, 2), obs(500, 3)], strategy())
    assert len(result.decisions) == 2
    assert result.decisions[0].state_fingerprint != result.decisions[1].state_fingerprint


def test_portfolio_accounting():
    state = build_state([obs(100, 1), obs(101, 2)])
    decision = make_decision(strategy(), state)
    order = PaperExecution().submit(decision, 2, 101)
    portfolio = PaperPortfolio(1000)
    snap = portfolio.apply(order)
    assert snap.cash == 798
    assert snap.position_quantity == 2
    assert snap.equity == 1000


def test_portfolio_state_round_trip(tmp_path):
    portfolio = PaperPortfolio(1000)
    state = build_state([obs(100, 1), obs(101, 2)])
    order = PaperExecution().submit(make_decision(strategy(), state), 2, 101)
    portfolio.apply(order)
    path = tmp_path / "portfolio.json"
    portfolio.save_state(path)

    restored = PaperPortfolio(0)
    restored.load_state(path)
    assert restored.state() == portfolio.state()
def test_registered_strategy_cannot_be_mutated_in_place():
    registry = StrategyRegistry()
    registered = registry.register(strategy())
    with pytest.raises((ValidationError, TypeError)):
        registered.version = "2.0.0"
    assert registry.get("momentum", "1.0.0").version == "1.0.0"


def test_registered_strategy_parameters_cannot_mutate_registry_state():
    registry = StrategyRegistry()
    registered = registry.register(strategy())
    registered.parameters["entry_return"] = 99
    assert registry.get("momentum", "1.0.0").parameters["entry_return"] == 0.001


