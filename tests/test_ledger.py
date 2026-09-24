from datetime import UTC, datetime

import pytest

from atlab.adapters import InMemoryMarketData
from atlab.ledger import ImmutableLedger
from atlab.models import MarketObservation, StrategyVersion
from atlab.portfolio import PaperPortfolio
from atlab.risk import DeterministicRiskEngine
from atlab.runtime import PaperTradingEngine


def test_ledger_identity_stable_for_same_database(tmp_path):
    db = tmp_path / "ledger.sqlite3"
    first = ImmutableLedger(db).identity

    assert first
    assert ImmutableLedger(db).identity == first


def test_ledger_identity_changes_when_database_recreated(tmp_path):
    # M6: a deleted-and-recreated database is a different database, even
    # at the same path — sequences restart at 0 and companion state must
    # not be trusted against it.
    db = tmp_path / "ledger.sqlite3"
    old_identity = ImmutableLedger(db).identity

    db.unlink()
    for suffix in ("-wal", "-shm"):
        sidecar = tmp_path / f"ledger.sqlite3{suffix}"
        if sidecar.exists():
            sidecar.unlink()

    assert ImmutableLedger(db).identity != old_identity


def _obs(price: float, second: int) -> MarketObservation:
    return MarketObservation(
        symbol="TEST",
        timestamp=datetime.fromtimestamp(second, tz=UTC),
        price=price,
        source="test",
    )


def _strategy() -> StrategyVersion:
    return StrategyVersion(
        strategy_id="momentum",
        version="1.0.0",
        hypothesis="test",
        parameters={"entry_return": 0.001},
    )


def test_engine_fails_closed_when_ledger_replaced(tmp_path):
    # M6 end-to-end: the engine persists portfolio state bound to the
    # ledger's identity; if the ledger file is deleted and recreated, the
    # restart fails closed instead of silently forking the audit trail.
    workdir = tmp_path / "work"
    workdir.mkdir()
    ledger_path = workdir / "ledger.sqlite3"
    portfolio_path = workdir / "portfolio.json"

    engine = PaperTradingEngine(
        InMemoryMarketData([_obs(100, 1), _obs(101, 2)]),
        _strategy(),
        DeterministicRiskEngine(),
        PaperPortfolio(1000.0),
        ImmutableLedger(ledger_path),
        quantity=1,
        portfolio_state_path=portfolio_path,
    )
    engine.run("TEST")
    assert engine.portfolio.position_quantity == 1

    ledger_path.unlink()
    for suffix in ("-wal", "-shm"):
        sidecar = workdir / f"ledger.sqlite3{suffix}"
        if sidecar.exists():
            sidecar.unlink()

    with pytest.raises(ValueError, match="LEDGER_IDENTITY_MISMATCH"):
        PaperTradingEngine(
            InMemoryMarketData([_obs(100, 1), _obs(101, 2)]),
            _strategy(),
            DeterministicRiskEngine(),
            PaperPortfolio(1000.0),
            ImmutableLedger(ledger_path),
            quantity=1,
            portfolio_state_path=portfolio_path,
        )
