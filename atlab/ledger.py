from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import ClassVar

from .models import LedgerEvent


class ImmutableLedger:
    """Append-only JSONL ledger with process-local single-writer protection."""

    _locks: ClassVar[dict[str, Lock]] = {}

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        key = str(self.path.resolve())
        self._lock = self._locks.setdefault(key, Lock())

    def append(self, event_type: str, event_id: str, payload: dict) -> LedgerEvent:
        with self._lock:
            existing = self._read_unlocked()
            if existing and event_id in {event.event_id for event in existing}:
                raise ValueError("LEDGER_EVENT_ID_EXISTS")
            sequence = len(existing)
            event = LedgerEvent(
                sequence=sequence,
                event_type=event_type,
                event_id=event_id,
                timestamp=datetime.now(UTC),
                payload=payload,
            )
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(event.model_dump_json() + "\n")
                handle.flush()
            return event

    def _read_unlocked(self) -> list[LedgerEvent]:
        if not self.path.exists():
            return []
        return [
            LedgerEvent.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def contains_event_id(self, event_id: str) -> bool:
        with self._lock:
            return any(event.event_id == event_id for event in self._read_unlocked())

    def contains_event_id(self, event_id: str) -> bool:
        with self._lock:
            return any(event.event_id == event_id for event in self._read_unlocked())

    def read(self) -> list[LedgerEvent]:
        with self._lock:
            return self._read_unlocked()
