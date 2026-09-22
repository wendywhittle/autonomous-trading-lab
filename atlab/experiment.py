from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime

from .models import MarketObservation, StrategyVersion


@dataclass(frozen=True)
class DataSnapshot:
    """Content-addressed market-data snapshot for reproducible experiments."""

    snapshot_id: str
    symbol: str
    observations: tuple[MarketObservation, ...]

    @classmethod
    def create(cls, observations: list[MarketObservation] | tuple[MarketObservation, ...]) -> DataSnapshot:
        ordered = tuple(sorted(observations, key=lambda item: item.timestamp))
        if not ordered:
            raise ValueError("EMPTY_DATA_SNAPSHOT")
        symbols = {item.symbol for item in ordered}
        if len(symbols) != 1:
            raise ValueError("MULTIPLE_SYMBOLS")
        payload = [
            {
                "metadata": item.metadata,
                "price": item.price,
                "source": item.source,
                "symbol": item.symbol,
                "timestamp": item.timestamp.isoformat(),
                "volume": item.volume,
            }
            for item in ordered
        ]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        snapshot_id = hashlib.sha256(encoded).hexdigest()
        return cls(snapshot_id=snapshot_id, symbol=ordered[0].symbol, observations=ordered)

    @property
    def start(self) -> datetime:
        return self.observations[0].timestamp

    @property
    def end(self) -> datetime:
        return self.observations[-1].timestamp


@dataclass(frozen=True)
class ExperimentRecord:
    """Immutable experiment identity tying strategy and data to one run."""

    experiment_id: str
    strategy_id: str
    strategy_version: str
    data_snapshot_id: str
    parameters: tuple[tuple[str, float], ...]

    @classmethod
    def create(cls, strategy: StrategyVersion, snapshot: DataSnapshot) -> ExperimentRecord:
        parameters = tuple(sorted((key, value) for key, value in strategy.parameters.items()))
        identity = {
            "data_snapshot_id": snapshot.snapshot_id,
            "parameters": parameters,
            "strategy_id": strategy.strategy_id,
            "strategy_version": strategy.version,
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        experiment_id = hashlib.sha256(encoded).hexdigest()
        return cls(
            experiment_id=experiment_id,
            strategy_id=strategy.strategy_id,
            strategy_version=strategy.version,
            data_snapshot_id=snapshot.snapshot_id,
            parameters=parameters,
        )
