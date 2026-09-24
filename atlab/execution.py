from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .broker import BrokerOrderRequest, BrokerOrderResult, BrokerOrderStatus
from .models import RiskDecision, Side
from .risk import RiskLimits
from .risk_state import RiskStateSnapshot


@dataclass(frozen=True)
class ExecutionIntent:
    """Durable intent and latest known external execution state."""

    idempotency_key: str
    request: BrokerOrderRequest
    status: BrokerOrderStatus = BrokerOrderStatus.UNKNOWN
    result: BrokerOrderResult | None = None
    authorization_id: str | None = None
    authorization_fingerprint: str | None = None
    authorization_issued_at: str | None = None
    authorization_expires_at: str | None = None
    risk_decision: RiskDecision | None = None
    risk_fingerprint: str | None = None
    risk_state_version: int | None = None
    risk_state_fingerprint: str | None = None
    risk_limits_version: int | None = None
    risk_limits_fingerprint: str | None = None

    def __post_init__(self) -> None:
        binding = (
            self.authorization_id,
            self.authorization_fingerprint,
            self.authorization_issued_at,
            self.authorization_expires_at,
        )
        if any(value is not None for value in binding) and not all(
            value is not None for value in binding
        ):
            raise ValueError("EXECUTION_INTENT_AUTHORIZATION_BINDING_INCOMPLETE")
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
                result TEXT,
                authorization_id TEXT,
                authorization_fingerprint TEXT,
                authorization_issued_at TEXT,
                authorization_expires_at TEXT,
                risk_decision TEXT,
                risk_fingerprint TEXT,
                risk_state_version INTEGER,
                risk_state_fingerprint TEXT,
                risk_limits_version INTEGER,
                risk_limits_fingerprint TEXT
            )
            """
        )
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(execution_intents)").fetchall()
        }
        for column in (
            "authorization_id",
            "authorization_fingerprint",
            "authorization_issued_at",
            "authorization_expires_at",
            "risk_decision",
            "risk_fingerprint",
            "risk_state_version",
            "risk_state_fingerprint",
            "risk_limits_version",
            "risk_limits_fingerprint",
        ):
            if column not in columns:
                connection.execute(
                    f"ALTER TABLE execution_intents ADD COLUMN {column} TEXT"
                )

        connection.execute("""CREATE TABLE IF NOT EXISTS live_risk_state (
                id INTEGER PRIMARY KEY CHECK (id = 1), symbol TEXT NOT NULL,
                current_position_notional REAL NOT NULL, equity REAL NOT NULL,
                session_start_equity REAL NOT NULL, high_water_mark REAL NOT NULL,
                available_cash REAL NOT NULL, version INTEGER NOT NULL, fingerprint TEXT NOT NULL)""")
        connection.execute("""CREATE TABLE IF NOT EXISTS live_risk_limits (
                id INTEGER PRIMARY KEY CHECK (id = 1), max_position_notional REAL NOT NULL,
                max_order_notional REAL NOT NULL, max_daily_loss REAL NOT NULL,
                max_drawdown REAL NOT NULL, max_leverage REAL NOT NULL,
                version INTEGER NOT NULL, fingerprint TEXT NOT NULL)""")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS execution_halt (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                active INTEGER NOT NULL,
                reason TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS execution_authorization_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                active_authorization_id TEXT
            )
            """
        )
        return connection

    def _activate_authorization_in_connection(
        self, connection: sqlite3.Connection, authorization_id: str
    ) -> None:
        if not authorization_id:
            raise ValueError("EXECUTION_AUTHORIZATION_ID_REQUIRED")
        connection.execute(
            """
            INSERT INTO execution_authorization_state(id, active_authorization_id)
            VALUES (1, ?)
            ON CONFLICT(id) DO UPDATE SET active_authorization_id = excluded.active_authorization_id
            """,
            (authorization_id,),
        )

    def activate_authorization(self, authorization_id: str) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._activate_authorization_in_connection(connection, authorization_id)
            connection.execute("COMMIT")
        except Exception:
            try:
                connection.execute("ROLLBACK")
            finally:
                connection.close()
            raise
        else:
            connection.close()

    def authorization_state_exists(self) -> bool:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT 1 FROM execution_authorization_state WHERE id = 1"
            ).fetchone()
            return row is not None
        finally:
            connection.close()

    @staticmethod
    def _has_active_live_authority_in_connection(connection: sqlite3.Connection) -> bool:
        row=connection.execute("""SELECT 1 FROM execution_intents WHERE authorization_id IS NOT NULL AND status IN (?, ?, ?) LIMIT 1""",(BrokerOrderStatus.UNKNOWN.value,BrokerOrderStatus.ACCEPTED.value,BrokerOrderStatus.PARTIALLY_FILLED.value)).fetchone()
        return row is not None

    @staticmethod
    def _snapshot_from_row(row: tuple) -> RiskStateSnapshot:
        return RiskStateSnapshot(symbol=row[0],current_position_notional=float(row[1]),equity=float(row[2]),session_start_equity=float(row[3]),high_water_mark=float(row[4]),available_cash=float(row[5]))

    def initialize_live_risk_state(self,snapshot: RiskStateSnapshot)->int:
        connection=self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT 1 FROM live_risk_state WHERE id=1").fetchone(): raise ValueError("LIVE_RISK_STATE_ALREADY_INITIALIZED")
            connection.execute("""INSERT INTO live_risk_state(id,symbol,current_position_notional,equity,session_start_equity,high_water_mark,available_cash,version,fingerprint) VALUES (1,?,?,?,?,?,?,0,?)""",(snapshot.symbol,snapshot.current_position_notional,snapshot.equity,snapshot.session_start_equity,snapshot.high_water_mark,snapshot.available_cash,snapshot.fingerprint()))
            connection.execute("COMMIT"); return 0
        except Exception:
            try: connection.execute("ROLLBACK")
            except sqlite3.OperationalError: pass
            raise
        finally: connection.close()

    def get_live_risk_state(self)->tuple[RiskStateSnapshot,int]:
        connection=self._connect()
        try:
            row=connection.execute("""SELECT symbol,current_position_notional,equity,session_start_equity,high_water_mark,available_cash,version,fingerprint FROM live_risk_state WHERE id=1""").fetchone()
            if row is None: raise RuntimeError("LIVE_RISK_STATE_NOT_INITIALIZED")
            snapshot=self._snapshot_from_row(row)
            if snapshot.fingerprint()!=row[7]: raise RuntimeError("LIVE_RISK_STATE_CORRUPT")
            return snapshot,int(row[6])
        finally: connection.close()

    def mutate_live_risk_state(self,snapshot: RiskStateSnapshot,expected_version:int)->int:
        connection=self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if self._has_active_live_authority_in_connection(connection): raise RuntimeError("LIVE_RISK_STATE_MUTATION_BLOCKED_BY_ACTIVE_AUTHORITY")
            row=connection.execute("SELECT version FROM live_risk_state WHERE id=1").fetchone()
            if row is None: raise RuntimeError("LIVE_RISK_STATE_NOT_INITIALIZED")
            actual=int(row[0])
            if actual!=expected_version: raise ValueError("RISK_STATE_VERSION_CONFLICT")
            new_version=actual+1
            connection.execute("""UPDATE live_risk_state SET symbol=?,current_position_notional=?,equity=?,session_start_equity=?,high_water_mark=?,available_cash=?,version=?,fingerprint=? WHERE id=1""",(snapshot.symbol,snapshot.current_position_notional,snapshot.equity,snapshot.session_start_equity,snapshot.high_water_mark,snapshot.available_cash,new_version,snapshot.fingerprint()))
            connection.execute("COMMIT"); return new_version
        except Exception:
            try: connection.execute("ROLLBACK")
            except sqlite3.OperationalError: pass
            raise
        finally: connection.close()

    def initialize_live_risk_limits(self,limits: RiskLimits)->int:
        connection=self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT 1 FROM live_risk_limits WHERE id=1").fetchone(): raise ValueError("LIVE_RISK_LIMITS_ALREADY_INITIALIZED")
            connection.execute("""INSERT INTO live_risk_limits(id,max_position_notional,max_order_notional,max_daily_loss,max_drawdown,max_leverage,version,fingerprint) VALUES (1,?,?,?,?,?,?,?)""",(limits.max_position_notional,limits.max_order_notional,limits.max_daily_loss,limits.max_drawdown,limits.max_leverage,0,limits.fingerprint()))
            connection.execute("COMMIT"); return 0
        except Exception:
            try: connection.execute("ROLLBACK")
            except sqlite3.OperationalError: pass
            raise
        finally: connection.close()

    def get_live_risk_limits(self)->tuple[RiskLimits,int]:
        connection=self._connect()
        try:
            row=connection.execute("""SELECT max_position_notional,max_order_notional,max_daily_loss,max_drawdown,max_leverage,version,fingerprint FROM live_risk_limits WHERE id=1""").fetchone()
            if row is None: raise RuntimeError("LIVE_RISK_LIMITS_NOT_INITIALIZED")
            limits=RiskLimits(float(row[0]),float(row[1]),float(row[2]),float(row[3]),float(row[4]))
            if limits.fingerprint()!=row[6]: raise RuntimeError("LIVE_RISK_LIMITS_CORRUPT")
            return limits,int(row[5])
        finally: connection.close()

    def mutate_live_risk_limits(self,limits: RiskLimits,expected_version:int)->int:
        connection=self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if self._has_active_live_authority_in_connection(connection): raise RuntimeError("LIVE_RISK_LIMITS_MUTATION_BLOCKED_BY_ACTIVE_AUTHORITY")
            row=connection.execute("SELECT version FROM live_risk_limits WHERE id=1").fetchone()
            if row is None: raise RuntimeError("LIVE_RISK_LIMITS_NOT_INITIALIZED")
            actual=int(row[0])
            if actual!=expected_version: raise ValueError("RISK_LIMITS_VERSION_CONFLICT")
            new_version=actual+1
            connection.execute("""UPDATE live_risk_limits SET max_position_notional=?,max_order_notional=?,max_daily_loss=?,max_drawdown=?,max_leverage=?,version=?,fingerprint=? WHERE id=1""",(limits.max_position_notional,limits.max_order_notional,limits.max_daily_loss,limits.max_drawdown,limits.max_leverage,new_version,limits.fingerprint()))
            connection.execute("COMMIT"); return new_version
        except Exception:
            try: connection.execute("ROLLBACK")
            except sqlite3.OperationalError: pass
            raise
        finally: connection.close()

    def is_halted_in_connection(self,connection)->bool:
        row=connection.execute("SELECT active FROM execution_halt WHERE id=1").fetchone()
        return bool(row and row[0])

    def _acquire_live_execution_authority_in_connection(self,connection,request,authorization_binding,risk_decision):
        state_row=connection.execute("""SELECT symbol,current_position_notional,equity,session_start_equity,high_water_mark,available_cash,version,fingerprint FROM live_risk_state WHERE id=1""").fetchone()
        if state_row is None: raise RuntimeError("LIVE_RISK_STATE_NOT_INITIALIZED")
        state=self._snapshot_from_row(state_row)
        if state.fingerprint()!=state_row[7]: raise RuntimeError("LIVE_RISK_STATE_CORRUPT")
        limits_row=connection.execute("""SELECT max_position_notional,max_order_notional,max_daily_loss,max_drawdown,max_leverage,version,fingerprint FROM live_risk_limits WHERE id=1""").fetchone()
        if limits_row is None: raise RuntimeError("LIVE_RISK_LIMITS_NOT_INITIALIZED")
        limits=RiskLimits(float(limits_row[0]),float(limits_row[1]),float(limits_row[2]),float(limits_row[3]),float(limits_row[4]))
        if limits.fingerprint()!=limits_row[6]: raise RuntimeError("LIVE_RISK_LIMITS_CORRUPT")
        if risk_decision.risk_state_fingerprint!=state.fingerprint(): raise RuntimeError("RISK_STATE_CHANGED_AFTER_APPROVAL")
        if risk_decision.risk_limits_fingerprint!=limits.fingerprint(): raise RuntimeError("RISK_LIMITS_FINGERPRINT_CONFLICT")
        if risk_decision.decision_id!=request.decision_id: raise RuntimeError("RISK_DECISION_ID_CONFLICT")
        auth_row=connection.execute("SELECT active_authorization_id FROM execution_authorization_state WHERE id=1").fetchone()
        if not auth_row or auth_row[0] != authorization_binding[0]: raise RuntimeError("EXECUTION_AUTHORIZATION_NOT_ACTIVATED")
        if self.is_halted_in_connection(connection): raise RuntimeError("EXECUTION_HALTED")
        return self._record_in_connection(connection,request,authorization_binding,risk_decision,int(state_row[6]),state.fingerprint(),int(limits_row[5]),limits.fingerprint())

    def authorization_is_active(self, authorization_id: str) -> bool:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT active_authorization_id FROM execution_authorization_state WHERE id = 1"
            ).fetchone()
            return bool(row and row[0] == authorization_id)
        finally:
            connection.close()

    def _revoke_authorization_in_connection(
        self, connection: sqlite3.Connection, authorization_id: str
    ) -> bool:
        if not authorization_id:
            raise ValueError("EXECUTION_AUTHORIZATION_ID_REQUIRED")
        row = connection.execute(
            "SELECT active_authorization_id FROM execution_authorization_state WHERE id = 1"
        ).fetchone()
        active = bool(row and row[0] == authorization_id)
        if active:
            if connection.execute("""SELECT 1 FROM execution_intents WHERE authorization_id = ? AND status IN (?, ?, ?) LIMIT 1""",(authorization_id,BrokerOrderStatus.UNKNOWN.value,BrokerOrderStatus.ACCEPTED.value,BrokerOrderStatus.PARTIALLY_FILLED.value)).fetchone():
                raise RuntimeError("EXECUTION_AUTHORIZATION_REVOCATION_BLOCKED_BY_ACTIVE_AUTHORITY")
            connection.execute("UPDATE execution_authorization_state SET active_authorization_id = NULL WHERE id = 1")
        return active

    def revoke_authorization(self, authorization_id: str) -> bool:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            active = self._revoke_authorization_in_connection(connection, authorization_id)
            connection.execute("COMMIT")
            return active
        except Exception:
            try:
                connection.execute("ROLLBACK")
            finally:
                connection.close()
            raise
        else:
            connection.close()

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
                "reference_price": request.reference_price,
                "decision_id": request.decision_id,
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
            reference_price=(float(encoded["reference_price"]) if encoded.get("reference_price") is not None else None),
            decision_id=encoded.get("decision_id"),
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
                "filled_quantity": result.filled_quantity,
                "remaining_quantity": result.remaining_quantity,
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
            filled_quantity=(
                float(encoded["filled_quantity"])
                if encoded.get("filled_quantity") is not None
                else None
            ),
            remaining_quantity=(
                float(encoded["remaining_quantity"])
                if encoded.get("remaining_quantity") is not None
                else None
            ),
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
            authorization_id=row[4],
            authorization_fingerprint=row[5],
            authorization_issued_at=row[6],
            authorization_expires_at=row[7],
            risk_decision=RiskDecision.model_validate(json.loads(row[8])) if row[8] else None,
            risk_fingerprint=row[9], risk_state_version=row[10], risk_state_fingerprint=row[11],
            risk_limits_version=row[12], risk_limits_fingerprint=row[13],
        )

    def _get_in_connection(
        self,
        connection: sqlite3.Connection,
        idempotency_key: str,
    ) -> ExecutionIntent | None:
        row = connection.execute(
            "SELECT idempotency_key, request, status, result, "
            "authorization_id, authorization_fingerprint, authorization_issued_at, "
            "authorization_expires_at, risk_decision, risk_fingerprint, "
            "risk_state_version, risk_state_fingerprint, risk_limits_version, risk_limits_fingerprint "
            "FROM execution_intents WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        return self._row_to_intent(row) if row is not None else None

    def _record_in_connection(
        self,
        connection: sqlite3.Connection,
        request: BrokerOrderRequest,
        authorization_binding: tuple[str, str, str, str] | None = None,
        risk_decision: RiskDecision | None = None,
        risk_state_version: int | None = None,
        risk_state_fingerprint: str | None = None,
        risk_limits_version: int | None = None,
        risk_limits_fingerprint: str | None = None,
    ) -> tuple[ExecutionIntent, bool]:
        existing = self._get_in_connection(connection, request.idempotency_key)
        encoded_request = self._encode_request(request)
        expected_binding = authorization_binding or (None, None, None, None)
        if existing is not None:
            if existing.request != request:
                raise ValueError("EXECUTION_INTENT_CONFLICT")
            actual_binding = (
                existing.authorization_id,
                existing.authorization_fingerprint,
                existing.authorization_issued_at,
                existing.authorization_expires_at,
            )
            if actual_binding != expected_binding:
                raise ValueError("EXECUTION_INTENT_AUTHORIZATION_CONFLICT")
            if existing.risk_fingerprint != (risk_decision.risk_fingerprint if risk_decision else None): raise ValueError("EXECUTION_INTENT_RISK_CONFLICT")
            if (existing.risk_state_version,existing.risk_state_fingerprint,existing.risk_limits_version,existing.risk_limits_fingerprint)!=(risk_state_version,risk_state_fingerprint,risk_limits_version,risk_limits_fingerprint): raise ValueError("EXECUTION_INTENT_RISK_AUTHORITY_CONFLICT")
            return existing, False

        connection.execute(
            """
            INSERT INTO execution_intents(
                idempotency_key, request, status, result,
                authorization_id, authorization_fingerprint,
                authorization_issued_at, authorization_expires_at,
                risk_decision, risk_fingerprint, risk_state_version, risk_state_fingerprint,
                risk_limits_version, risk_limits_fingerprint
            )
            VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request.idempotency_key,
                encoded_request,
                BrokerOrderStatus.UNKNOWN.value,
                *expected_binding,
                risk_decision.model_dump_json() if risk_decision else None,
                risk_decision.risk_fingerprint if risk_decision else None,
                risk_state_version, risk_state_fingerprint, risk_limits_version, risk_limits_fingerprint,
            ),
        )
        created = self._get_in_connection(connection, request.idempotency_key)
        if created is None:
            raise RuntimeError("EXECUTION_INTENT_NOT_PERSISTED")
        return created, True

    @staticmethod
    def _validate_result_for_request(
        request: BrokerOrderRequest,
        existing: ExecutionIntent,
        result: BrokerOrderResult,
    ) -> None:
        if result.filled_quantity is None and result.remaining_quantity is None:
            if result.status is BrokerOrderStatus.PARTIALLY_FILLED:
                raise ValueError("PARTIAL_FILL_QUANTITY_REQUIRED")
            return

        filled = result.filled_quantity
        remaining = result.remaining_quantity
        if filled is None or remaining is None:
            raise ValueError("EXECUTION_FILL_QUANTITY_PAIR_REQUIRED")
        if abs((filled + remaining) - request.quantity) > 1e-9:
            raise ValueError("EXECUTION_FILL_QUANTITY_TOTAL_MISMATCH")

        previous_filled = (
            existing.result.filled_quantity
            if existing.result is not None and existing.result.filled_quantity is not None
            else 0.0
        )
        if filled < previous_filled - 1e-9:
            raise ValueError("EXECUTION_FILL_QUANTITY_REGRESSION")

        if result.status is BrokerOrderStatus.PARTIALLY_FILLED:
            if not (0 < filled < request.quantity):
                raise ValueError("INVALID_PARTIAL_FILL_QUANTITY")
        elif result.status is BrokerOrderStatus.FILLED:
            if abs(filled - request.quantity) > 1e-9 or remaining != 0:
                raise ValueError("INVALID_FILLED_QUANTITY")
        elif (
            result.status
            in {
                BrokerOrderStatus.CANCELED,
                BrokerOrderStatus.REJECTED,
            }
            and (filled > request.quantity or remaining < 0)
        ):
            raise ValueError("INVALID_TERMINAL_FILL_QUANTITY")

    def _update_result_in_connection(
        self,
        connection: sqlite3.Connection,
        idempotency_key: str,
        result: BrokerOrderResult,
    ) -> ExecutionIntent:
        existing = self._get_in_connection(connection, idempotency_key)
        if existing is None:
            raise KeyError("EXECUTION_INTENT_NOT_FOUND")

        self._validate_result_for_request(existing.request, existing, result)

        if (
            existing.result is not None
            and existing.result.broker_order_id is not None
            and result.broker_order_id != existing.result.broker_order_id
        ):
            raise ValueError("EXECUTION_BROKER_ORDER_ID_CONFLICT")

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
