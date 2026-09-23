from datetime import UTC, datetime

import pytest

from atlab.experiment import DataSnapshot, ExperimentRecord
from atlab.models import MarketObservation, StrategyVersion


def obs(price, second, *, volume=0, metadata=None):
    return MarketObservation(
        symbol="TEST",
        timestamp=datetime.fromtimestamp(second, tz=UTC),
        price=price,
        volume=volume,
        source="test",
        metadata=metadata or {},
    )


def strategy(parameters=None):
    return StrategyVersion(
        strategy_id="momentum",
        version="1.0.0",
        hypothesis="test",
        parameters=parameters or {"entry_return": 0.001},
    )


def test_data_snapshot_is_content_addressed_and_order_independent():
    first = [obs(100, 1), obs(101, 2, volume=5)]
    snapshot_a = DataSnapshot.create(first)
    snapshot_b = DataSnapshot.create(list(reversed(first)))

    assert snapshot_a.snapshot_id == snapshot_b.snapshot_id
    assert snapshot_a.observations == snapshot_b.observations
    assert snapshot_a.start < snapshot_a.end


def test_data_snapshot_changes_when_observation_changes():
    original = DataSnapshot.create([obs(100, 1), obs(101, 2)])
    changed = DataSnapshot.create([obs(100, 1), obs(102, 2)])

    assert original.snapshot_id != changed.snapshot_id


def test_data_snapshot_rejects_empty_or_mixed_symbols():
    with pytest.raises(ValueError, match="EMPTY_DATA_SNAPSHOT"):
        DataSnapshot.create([])
    with pytest.raises(ValueError, match="MULTIPLE_SYMBOLS"):
        DataSnapshot.create(
            [
                obs(100, 1),
                MarketObservation(
                    symbol="OTHER",
                    timestamp=datetime.fromtimestamp(2, tz=UTC),
                    price=101,
                    source="test",
                ),
            ]
        )


def test_experiment_identity_is_reproducible():
    snapshot = DataSnapshot.create([obs(100, 1), obs(101, 2)])
    first = ExperimentRecord.create(strategy(), snapshot)
    second = ExperimentRecord.create(strategy(), snapshot)

    assert first == second
    assert len(first.experiment_id) == 64


def test_experiment_identity_changes_with_strategy_parameters():
    snapshot = DataSnapshot.create([obs(100, 1), obs(101, 2)])
    first = ExperimentRecord.create(strategy(), snapshot)
    changed = ExperimentRecord.create(
        strategy({"entry_return": 0.002}), snapshot
    )

    assert first.experiment_id != changed.experiment_id
