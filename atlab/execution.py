from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .broker import BrokerOrderRequest, BrokerOrderResult, BrokerOrderStatus
from .models import Side


@dataclass(frozen=True)
class ExecutionIntent:
    """Durable intent and latest known external execution state."""

    idempotency_key: str
    request: BrokerOrderRequest
    status: BrokerOrderStatus = BrokerOrderStatus.UNKNOWN
    result: BrokerOrderResult | None = None

    def __post_init__(self) -> None:
        if self.status is BrokerOrderStatus.UNKNOWN and self.result is None:
            return
        if self.result is None:
            raise ValueError("EXECUTION_RESULT_REQUIRED")


class ExecutionIntentStore:
    """SQLite-backed durable execution intent state.

    Execution intent state shares the SQLite database file with the immutable
    audit ledger. Coordinator-level transactions can therefore commit or roll
    back current execution state and its audit event together.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS execution_intents (
                idempotency_key TEXT PRIMARY KEY,
                request TEXT NOT NULL,
                status TEXT NOT NULL,
                result TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS execution_halt (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                active INTEGER NOT NULL,
                reason TEXT NOT NULL
            )
            """
        )
        return connection

    def halt(self, reason: str) -> None:
        if not reason:
            raise ValueError("EXECUTION_HALT_REASON_REQUIRED")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO execution_halt(id, active, reason)
                VALUES (1, 1, ?)
                ON CONFLICT(id) DO UPDATE SET active = 1, reason = excluded.reason
                """,
                (reason,),
            )
            connection.execute("COMMIT")
        finally:
            connection.close()

    def is_halted(self) -> bool:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT active FROM execution_halt WHERE id = 1"
            ).fetchone()
            return bool(row and row[0])
        finally:
            connection.close()

    def halt_reason(self) -> str | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT reason FROM execution_halt WHERE id = 1 AND active = 1"
            ).fetchone()
            return row[0] if row else None
        finally:
            connection.close()

    def clear_halt(self) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("UPDATE execution_halt SET active = 0 WHERE id = 1")
            connection.execute("COMMIT")
        finally:
            connection.close()

    @staticmethod
    def _encode_request(request: BrokerOrderRequest) -> str:
        return json.dumps(
            {
                "idempotency_key": request.idempotency_key,
                "symbol": request.symbol,
                "side": request.side.value,
                "quantity": request.quantity,
                "price": request.price,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _decode_request(data: str) -> BrokerOrderRequest:
        encoded = json.loads(data)
        return BrokerOrderRequest(
            idempotency_key=encoded["idempotency_key"],
            symbol=encoded["symbol"],
            side=Side(encoded["side"]),
            quantity=float(encoded["quantity"]),
            price=float(encoded["price"]) if encoded["price"] is not None else None,
        )

    @staticmethod
    def _encode_result(result: BrokerOrderResult | None) -> str | None:
        if result is None:
            return None
        return json.dumps(
            {
                "accepted": result.accepted,
                "broker_order_id": result.broker_order_id,
                "status": result.status.value,
                "message": result.message,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _decode_result(data: str | None) -> BrokerOrderResult | None:
        if data is None:
            return None
        encoded = json.loads(data)
        return BrokerOrderResult(
            accepted=bool(encoded["accepted"]),
            broker_order_id=encoded["broker_order_id"],
            status=BrokerOrderStatus(encoded["status"]),
            message=str(encoded["message"]),
        )

    def _initialize(self) -> None:
        connection = self._connect()
        connection.close()

    def _row_to_intent(self, row: tuple) -> ExecutionIntent:
        return ExecutionIntent(
            idempotency_key=row[0],
            request=self._decode_request(row[1]),
            status=BrokerOrderStatus(row[2]),
            result=self._decode_result(row[3]),
        )

    def _get_in_connection(
        self,
        connection: sqlite3.Connection,
        idempotency_key: str,
    ) -> ExecutionIntent | None:
        row = connection.execute(
            "SELECT idempotency_key, request, status, result "
            "FROM execution_intents WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        return self._row_to_intent(row) if row is not None else None

    def _record_in_connection(
        self,
        connection: sqlite3.Connection,
        request: BrokerOrderRequest,
    ) -> tuple[ExecutionIntent, bool]:
        existing = self._get_in_connection(connection, request.idempotency_key)
        encoded_request = self._encode_request(request)
        if existing is not None:
            if existing.request != request:
                raise ValueError("EXECUTION_INTENT_CONFLICT")
            return existing, False

        connection.execute(
            """
            INSERT INTO execution_intents(idempotency_key, request, status, result)
            VALUES (?, ?, ?, NULL)
            """,
            (
                request.idempotency_key,
                encoded_request,
                BrokerOrderStatus.UNKNOWN.value,
            ),
        )
        created = self._get_in_connection(connection, request.idempotency_key)
        if created is None:
            raise RuntimeError("EXECUTION_INTENT_NOT_PERSISTED")
        return created, True

    def _update_result_in_connection(
        self,
        connection: sqlite3.Connection,
        idempotency_key: str,
        result: BrokerOrderResult,
    ) -> ExecutionIntent:
        existing = self._get_in_connection(connection, idempotency_key)
        if existing is None:
            raise KeyError("EXECUTION_INTENT_NOT_FOUND")

        if existing.result is not None and existing.result != result:
            terminal = {
                BrokerOrderStatus.FILLED,
                BrokerOrderStatus.CANCELED,
                BrokerOrderStatus.REJECTED,
            }
            allowed = {
                BrokerOrderStatus.UNKNOWN: {
                    BrokerOrderStatus.ACCEPTED,
                    BrokerOrderStatus.PARTIALLY_FILLED,
                    BrokerOrderStatus.FILLED,
                    BrokerOrderStatus.CANCELED,
                    BrokerOrderStatus.REJECTED,
                },
                BrokerOrderStatus.ACCEPTED: {
                    BrokerOrderStatus.ACCEPTED,
                    BrokerOrderStatus.PARTIALLY_FILLED,
                    BrokerOrderStatus.FILLED,
                    BrokerOrderStatus.CANCELED,
                    BrokerOrderStatus.REJECTED,
                },
                BrokerOrderStatus.PARTIALLY_FILLED: {
                    BrokerOrderStatus.PARTIALLY_FILLED,
                    BrokerOrderStatus.FILLED,
                    BrokerOrderStatus.CANCELED,
                },
                BrokerOrderStatus.FILLED: {BrokerOrderStatus.FILLED},
                BrokerOrderStatus.CANCELED: {BrokerOrderStatus.CANCELED},
                BrokerOrderStatus.REJECTED: {BrokerOrderStatus.REJECTED},
            }
            if result.status not in allowed[existing.status]:
                if existing.status in terminal:
                    raise ValueError("EXECUTION_TERMINAL_STATE_CONFLICT")
                raise ValueError("EXECUTION_STATE_TRANSITION_CONFLICT")

        connection.execute(
            """
            UPDATE execution_intents
            SET status = ?, result = ?
            WHERE idempotency_key = ?
            """,
            (
                result.status.value,
                self._encode_result(result),
                idempotency_key,
            ),
        )
        updated = self._get_in_connection(connection, idempotency_key)
        if updated is None:
            raise RuntimeError("EXECUTION_INTENT_NOT_PERSISTED")
        return updated

    def record(self, request: BrokerOrderRequest) -> ExecutionIntent:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            intent, _ = self._record_in_connection(connection, request)
            connection.execute("COMMIT")
            return intent
        except Exception:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            raise
        finally:
            connection.close()

    def update_result(
        self,
        idempotency_key: str,
        result: BrokerOrderResult,
    ) -> ExecutionIntent:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            updated = self._update_result_in_connection(
                connection, idempotency_key, result
            )
            connection.execute("COMMIT")
            return updated
        except Exception:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            raise
        finally:
            connection.close()

    def get(self, idempotency_key: str) -> ExecutionIntent | None:
        connection = self._connect()
        try:
            return self._get_in_connection(connection, idempotency_key)
        finally:
            connection.close()

    def pending_keys(self) -> tuple[str, ...]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT idempotency_key
                FROM execution_intents
                WHERE status IN (?, ?, ?)
                ORDER BY idempotency_key
                """,
                (
                    BrokerOrderStatus.UNKNOWN.value,
                    BrokerOrderStatus.ACCEPTED.value,
                    BrokerOrderStatus.PARTIALLY_FILLED.value,
                ),
            ).fetchall()
            return tuple(row[0] for row in rows)
        finally:
            connection.close()

    def all_keys(self) -> tuple[str, ...]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT idempotency_key FROM execution_intents ORDER BY idempotency_key"
            ).fetchall()
            return tuple(row[0] for row in rows)
        finally:
            connection.close()

    def unknown_keys(self) -> tuple[str, ...]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT idempotency_key
                FROM execution_intents
                WHERE status = ?
                ORDER BY idempotency_key
                """,
                (BrokerOrderStatus.UNKNOWN.value,),
            ).fetchall()
            return tuple(row[0] for row in rows)
        finally:
            connection.close()


@dataclass(frozen=True)
class ExecutionReconciliation:
    idempotency_key: str
    result: BrokerOrderResult
