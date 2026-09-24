from __future__ import annotations

import csv
import math
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from .models import MarketObservation


class MarketDataAdapter(Protocol):
    """Provider-independent market data contract."""

    def observations(self, symbol: str) -> Iterable[MarketObservation]: ...


class InMemoryMarketData:
    def __init__(self, observations: Iterable[MarketObservation]):
        self._observations = tuple(observations)

    def observations(self, symbol: str) -> Iterable[MarketObservation]:
        return tuple(item for item in self._observations if item.symbol == symbol)


class CsvMarketData:
    """CSV OHLCV market-data adapter.

    Expected header: ``timestamp,symbol,open,high,low,close`` with an optional
    ``volume`` column. ``price`` is the close. Rows are validated eagerly at
    construction; :meth:`observations` returns them in deterministic
    chronological order (ties broken by file row order, so the output is
    stable). Consumers iterate with a historical cutoff, so no future data
    can leak into a state built at an earlier timestamp.

    Timezone handling: timestamps without an explicit offset are interpreted
    as UTC. This is a silent assumption — a source file recorded in
    exchange-local time will be shifted by hours with no warning. Convert
    such files to UTC (or add explicit offsets) before loading.
    """

    REQUIRED_COLUMNS = ("timestamp", "symbol", "open", "high", "low", "close")

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._observations = self._load()

    @property
    def path(self) -> Path:
        return self._path

    def observations(self, symbol: str) -> Iterable[MarketObservation]:
        return tuple(item for item in self._observations if item.symbol == symbol)

    def _load(self) -> tuple[MarketObservation, ...]:
        try:
            handle = self._path.open("r", newline="", encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"CSV_UNREADABLE:{self._path}") from exc
        with handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            missing = [col for col in self.REQUIRED_COLUMNS if col not in header]
            if missing:
                raise ValueError(f"CSV_MISSING_COLUMNS:{','.join(missing)}")
            rows = []
            for index, record in enumerate(reader):
                if record.get("timestamp") is None or not str(record.get("timestamp")).strip():
                    continue  # skip blank lines
                rows.append((index, self._parse_row(index, record)))
        rows.sort(key=lambda item: (item[1].timestamp, item[0]))
        return tuple(observation for _, observation in rows)

    def _parse_row(self, index: int, record: dict) -> MarketObservation:
        label = f"row {index + 2}"  # 1-based + header
        timestamp = self._parse_timestamp(record["timestamp"], label)
        symbol = str(record["symbol"]).strip()
        if not symbol:
            raise ValueError(f"CSV_EMPTY_SYMBOL:{label}")
        numeric = {}
        for column in ("open", "high", "low", "close", "volume"):
            raw = record.get(column)
            if raw is None or str(raw).strip() == "":
                if column == "volume":
                    numeric[column] = 0.0
                    continue
                raise ValueError(f"CSV_MISSING_VALUE:{column}:{label}")
            try:
                numeric[column] = float(str(raw).strip())
            except ValueError as exc:
                raise ValueError(f"CSV_BAD_NUMERIC:{column}:{label}") from exc
        for column in ("open", "high", "low", "close", "volume"):
            if not math.isfinite(numeric[column]):
                raise ValueError(f"CSV_NON_FINITE_NUMERIC:{column}:{label}")
        if numeric["close"] <= 0:
            raise ValueError(f"CSV_NON_POSITIVE_PRICE:{label}")
        if not (numeric["low"] <= min(numeric["open"], numeric["close"]) <= max(numeric["open"], numeric["close"]) <= numeric["high"]):
            raise ValueError(f"CSV_HIGH_LOW_INCONSISTENT:{label}")
        return MarketObservation(
            symbol=symbol,
            timestamp=timestamp,
            price=numeric["close"],
            volume=numeric["volume"],
            source=f"csv:{self._path.name}",
            metadata={
                "open": numeric["open"],
                "high": numeric["high"],
                "low": numeric["low"],
                "close": numeric["close"],
            },
        )

    @staticmethod
    def _parse_timestamp(raw: str, label: str) -> datetime:
        try:
            parsed = datetime.fromisoformat(str(raw).strip())
        except ValueError as exc:
            raise ValueError(f"CSV_BAD_TIMESTAMP:{label}") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed
