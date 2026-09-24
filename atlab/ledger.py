from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from .models import LedgerEvent

#: Link value for the first entry of a hash-chained ledger.
LEDGER_GENESIS_PREV_HASH = "GENESIS"


class ImmutableLedger:
    """Append-only SQLite ledger with transactional sequencing and a hash chain.

    Every entry commits ``prev_hash`` (the previous entry's hash, or
    ``LEDGER_GENESIS_PREV_HASH`` for the first entry) and ``entry_hash``
    (SHA-256 over the entry's canonical fields plus ``prev_hash``). A
    ``ledger_meta`` tip anchor records the latest (sequence, hash) in the
    same transaction as each append, so suffix truncation and full wipes are
    detected as well as in-place edits. The full chain is verified on every
    ``read()`` and whenever a ledger is opened, so post-hoc tampering
    (UPDATE/DELETE/renumber) fails closed instead of passing silently.

    Databases created before the hash chain existed are migrated on first
    connect: the columns are added and the chain is rebuilt deterministically
    over existing rows in sequence order.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @staticmethod
    def _hash_entry(
        sequence: int,
        event_type: str,
        event_id: str,
        timestamp_iso: str,
        payload_json: str,
        prev_hash: str,
    ) -> str:
        canonical = "\n".join(
            (
                str(sequence),
                event_type,
                event_id,
                timestamp_iso,
                payload_json,
                prev_hash,
            )
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

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
                payload TEXT NOT NULL,
                prev_hash TEXT NOT NULL DEFAULT '',
                entry_hash TEXT NOT NULL DEFAULT ''
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ledger_meta (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                tip_sequence INTEGER NOT NULL,
                tip_hash TEXT NOT NULL
            )
            """
        )
        self._ensure_hash_chain(connection)
        return connection

    def _ensure_hash_chain(self, connection: sqlite3.Connection) -> None:
        """Migrate pre-chain databases and (re)build the hash chain if needed."""
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(events)").fetchall()
        }
        migrated = False
        if "prev_hash" not in columns:
            connection.execute(
                "ALTER TABLE events ADD COLUMN prev_hash TEXT NOT NULL DEFAULT ''"
            )
            migrated = True
        if "entry_hash" not in columns:
            connection.execute(
                "ALTER TABLE events ADD COLUMN entry_hash TEXT NOT NULL DEFAULT ''"
            )
            migrated = True
        if migrated:
            self._rebuild_chain(connection)
            return
        (unchained,) = connection.execute(
            "SELECT COUNT(*) FROM events WHERE entry_hash = ''"
        ).fetchone()
        if unchained:
            # A previous migration was interrupted; the rebuild is
            # deterministic, so recomputing the whole chain is safe.
            self._rebuild_chain(connection)

    def _rebuild_chain(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            "SELECT sequence, event_type, event_id, timestamp, payload "
            "FROM events ORDER BY sequence"
        ).fetchall()
        prev_hash = LEDGER_GENESIS_PREV_HASH
        for sequence, event_type, event_id, timestamp, payload in rows:
            entry_hash = self._hash_entry(
                sequence, event_type, event_id, timestamp, payload, prev_hash
            )
            connection.execute(
                "UPDATE events SET prev_hash = ?, entry_hash = ? WHERE sequence = ?",
                (prev_hash, entry_hash, sequence),
            )
            prev_hash = entry_hash
        self._set_tip(connection, rows[-1][0] if rows else None, prev_hash)

    @staticmethod
    def _set_tip(
        connection: sqlite3.Connection, tip_sequence: int | None, tip_hash: str
    ) -> None:
        if tip_sequence is None:
            connection.execute("DELETE FROM ledger_meta WHERE id = 1")
            return
        connection.execute(
            """
            INSERT INTO ledger_meta (id, tip_sequence, tip_hash)
            VALUES (1, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                tip_sequence = excluded.tip_sequence,
                tip_hash = excluded.tip_hash
            """,
            (tip_sequence, tip_hash),
        )

    def _row_to_event(self, row: tuple) -> LedgerEvent:
        return LedgerEvent(
            sequence=row[0],
            event_type=row[1],
            event_id=row[2],
            timestamp=datetime.fromisoformat(row[3]),
            payload=json.loads(row[4]),
            prev_hash=row[5],
            entry_hash=row[6],
        )

    def _initialize(self) -> None:
        connection = self._connect()
        connection.close()
        # Verify at startup: a persisted ledger that was tampered with while
        # this process was away fails closed here, before any append or read.
        self.verify_chain()

    def verify_chain(self) -> None:
        """Verify sequence contiguity, the full hash chain, and the tip anchor.

        Raises ``ValueError("LEDGER_CHAIN_BROKEN")`` on sequence gaps, a
        wiped table, a deleted tip anchor, or suffix truncation; raises
        ``ValueError("LEDGER_CHAIN_TAMPERED")`` on any hash/link mismatch.
        """
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT sequence, event_type, event_id, timestamp, payload, "
                "prev_hash, entry_hash FROM events ORDER BY sequence"
            ).fetchall()
            meta = connection.execute(
                "SELECT tip_sequence, tip_hash FROM ledger_meta WHERE id = 1"
            ).fetchone()
        finally:
            connection.close()
        if not rows:
            if meta is not None:
                # Every event was deleted but the tip anchor remains.
                raise ValueError("LEDGER_CHAIN_BROKEN")
            return
        prev_hash = LEDGER_GENESIS_PREV_HASH
        for index, row in enumerate(rows):
            (
                sequence,
                event_type,
                event_id,
                timestamp,
                payload,
                row_prev_hash,
                row_entry_hash,
            ) = row
            if sequence != index:
                raise ValueError("LEDGER_CHAIN_BROKEN")
            if row_prev_hash != prev_hash:
                raise ValueError("LEDGER_CHAIN_TAMPERED")
            expected = self._hash_entry(
                sequence, event_type, event_id, timestamp, payload, row_prev_hash
            )
            if not hmac.compare_digest(expected, row_entry_hash):
                raise ValueError("LEDGER_CHAIN_TAMPERED")
            prev_hash = row_entry_hash
        if meta is None:
            # Rows exist but no tip anchor: the anchor was deleted, or the
            # chain was assembled outside this ledger. Fail closed.
            raise ValueError("LEDGER_CHAIN_BROKEN")
        tip_sequence, tip_hash = meta
        if tip_sequence != rows[-1][0] or not hmac.compare_digest(
            tip_hash, prev_hash
        ):
            # Suffix truncation (or a forged shorter chain): the anchor
            # disagrees with the table contents.
            raise ValueError("LEDGER_CHAIN_TAMPERED")

    def _append_in_connection(
        self,
        connection: sqlite3.Connection,
        event_type: str,
        event_id: str,
        payload: dict,
    ) -> LedgerEvent:
        timestamp_iso = datetime.now(UTC).isoformat()
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        sequence = connection.execute(
            "SELECT COALESCE(MAX(sequence) + 1, 0) FROM events"
        ).fetchone()[0]
        row = connection.execute(
            "SELECT entry_hash FROM events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        prev_hash = row[0] if row is not None else LEDGER_GENESIS_PREV_HASH
        entry_hash = self._hash_entry(
            sequence, event_type, event_id, timestamp_iso, payload_json, prev_hash
        )
        try:
            connection.execute(
                """
                INSERT INTO events(sequence, event_type, event_id, timestamp,
                                   payload, prev_hash, entry_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sequence,
                    event_type,
                    event_id,
                    timestamp_iso,
                    payload_json,
                    prev_hash,
                    entry_hash,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("LEDGER_EVENT_ID_EXISTS") from exc
        # Anchor the chain tip in the same transaction: a suffix deletion or
        # full wipe leaves the anchor disagreeing with the table, which
        # verify_chain reports instead of accepting a truncated chain.
        self._set_tip(connection, sequence, entry_hash)
        return LedgerEvent(
            sequence=sequence,
            event_type=event_type,
            event_id=event_id,
            timestamp=datetime.fromisoformat(timestamp_iso),
            payload=json.loads(payload_json),
            prev_hash=prev_hash,
            entry_hash=entry_hash,
        )

    def append(self, event_type: str, event_id: str, payload: dict) -> LedgerEvent:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            event = self._append_in_connection(
                connection, event_type, event_id, payload
            )
            connection.execute("COMMIT")
            return event
        except Exception:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
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

    def has_event_type(self, event_type: str) -> bool:
        """Return whether any event of the given type exists (global scope)."""
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT 1 FROM events WHERE event_type = ? LIMIT 1", (event_type,)
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

    def events_for_proposal(self, proposal_id: str) -> list[LedgerEvent]:
        events = self.read()
        return [
            event
            for event in events
            if event.payload.get("proposal_id") == proposal_id
        ]

    def read(self) -> list[LedgerEvent]:
        # The chain is verified on every read so tampering fails closed at
        # the point of use, not just at startup.
        self.verify_chain()
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT sequence, event_type, event_id, timestamp, payload,
                       prev_hash, entry_hash
                FROM events ORDER BY sequence
                """
            ).fetchall()
            return [self._row_to_event(row) for row in rows]
        finally:
            connection.close()
