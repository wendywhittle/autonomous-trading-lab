from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from .models import LedgerEvent


class ImmutableLedger:
    """Append-only SQLite ledger with transactional multi-process sequencing."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY,
                event_type TEXT NOT NULL,
                event_id TEXT NOT NULL UNIQUE,
                timestamp TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        return connection

    def _row_to_event(self, row: tuple) -> LedgerEvent:
        return LedgerEvent(
            sequence=row[0],
            event_type=row[1],
            event_id=row[2],
            timestamp=datetime.fromisoformat(row[3]),
            payload=json.loads(row[4]),
        )

    def _initialize(self) -> None:
        connection = self._connect()
        connection.close()

    def append(self, event_type: str, event_id: str, payload: dict) -> LedgerEvent:
        event = LedgerEvent(
            sequence=0,
            event_type=event_type,
            event_id=event_id,
            timestamp=datetime.now(UTC),
            payload=payload,
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            sequence = connection.execute(
                "SELECT COALESCE(MAX(sequence) + 1, 0) FROM events"
            ).fetchone()[0]
            event = event.model_copy(update={"sequence": sequence})
            connection.execute(
                """
                INSERT INTO events(sequence, event_type, event_id, timestamp, payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event.sequence,
                    event.event_type,
                    event.event_id,
                    event.timestamp.isoformat(),
                    json.dumps(event.payload, sort_keys=True, separators=(",", ":")),
                ),
            )
            connection.execute("COMMIT")
            return event
        except sqlite3.IntegrityError as exc:
            connection.execute("ROLLBACK")
            raise ValueError("LEDGER_EVENT_ID_EXISTS") from exc
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def contains_event_id(self, event_id: str) -> bool:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT 1 FROM events WHERE event_id = ? LIMIT 1", (event_id,)
            ).fetchone()
            return row is not None
        finally:
            connection.close()

    def events_for_decision(self, decision_id: str) -> list[LedgerEvent]:
        events = self.read()
        return [
            event
            for event in events
            if event.event_id == decision_id
            or event.payload.get("decision_id") == decision_id
        ]

    def read(self) -> list[LedgerEvent]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT sequence, event_type, event_id, timestamp, payload
                FROM events ORDER BY sequence
                """
            ).fetchall()
            return [self._row_to_event(row) for row in rows]
        finally:
            connection.close()
