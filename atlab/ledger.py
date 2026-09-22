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
            sequence = self._next_sequence()
            event = LedgerEvent(
                sequence=sequence,
                event_type=event_type,
                event_id=event_id,
                timestamp=datetime.now(UTC),
                payload=payload,
            )
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(event.model_dump_json() + "\n")
            return event

    def _next_sequence(self) -> int:
        if not self.path.exists():
            return 0
        with self.path.open("rb") as handle:
            return sum(1 for _ in handle)

    def read(self) -> list[LedgerEvent]:
        if not self.path.exists():
            return []
        return [
            LedgerEvent.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line
        ]
