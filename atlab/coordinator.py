from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from enum import Enum

from .broker import (
    BrokerAdapter,
    BrokerOrderRequest,
    BrokerOrderResult,
    BrokerOrderStatus,
)
from .execution import ExecutionIntentStore, ExecutionReconciliation
from .execution_reconciliation import (
    ExecutionConsistency,
    inspect_execution_consistency,
    reconciliation_halt_reason,
)
from .ledger import ImmutableLedger


class IntentStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    UNKNOWN = "UNKNOWN"
    TERMINAL = "TERMINAL"


@dataclass(frozen=True)
class ExecutionAttempt:
    idempotency_key: str
    status: IntentStatus
    result: BrokerOrderResult


class ExecutionCoordinator:
    """Coordinates durable intent state, provider calls, and atomic audit events."""

    def __init__(
        self,
        store: ExecutionIntentStore,
        broker: BrokerAdapter,
        ledger: ImmutableLedger,
    ):
        if store.path.resolve() != ledger.path.resolve():
            raise ValueError("EXECUTION_AND_LEDGER_MUST_SHARE_DATABASE")
        self.store = store
        self.broker = broker
        self.ledger = ledger

    def _transaction(self) -> sqlite3.Connection:
        connection = self.store._connect()
        connection.execute("BEGIN IMMEDIATE")
        return connection

    @staticmethod
    def _event_id(event_type: str, idempotency_key: str) -> str:
        return f"{event_type.lower()}-{idempotency_key}"

    def _commit_intent_creation(self, request: BrokerOrderRequest) -> None:
        connection = self._transaction()
        try:
            _, created = self.store._record_in_connection(connection, request)
            if created:
                self.ledger._append_in_connection(
                    connection,
                    "EXECUTION_INTENT_CREATED",
                    self._event_id(
                        "EXECUTION_INTENT_CREATED", request.idempotency_key
                    ),
                    {
                        "idempotency_key": request.idempotency_key,
                        "symbol": request.symbol,
                        "side": request.side.value,
                        "quantity": request.quantity,
                        "price": request.price,
                    },
                )
            connection.execute("COMMIT")
        except Exception:
            try:
                connection.execute("ROLLBACK")
            finally:
                connection.close()
            raise
        else:
            connection.close()

    @staticmethod
    def _result_event_id(
        event_type: str, idempotency_key: str, result: BrokerOrderResult
    ) -> str:
        payload = {
            "idempotency_key": idempotency_key,
            "broker_order_id": result.broker_order_id,
            "status": result.status.value,
            "accepted": result.accepted,
            "message": result.message,
            "filled_quantity": result.filled_quantity,
            "remaining_quantity": result.remaining_quantity,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        return f"{event_type.lower()}-{idempotency_key}-{digest}"

    def _commit_result(
        self,
        idempotency_key: str,
        result: BrokerOrderResult,
        event_type: str,
    ) -> None:
        connection = self._transaction()
        try:
            self.store._update_result_in_connection(connection, idempotency_key, result)
            self.ledger._append_in_connection(
                connection,
                event_type,
                self._result_event_id(event_type, idempotency_key, result),
                {
                    "idempotency_key": idempotency_key,
                    "broker_order_id": result.broker_order_id,
                    "status": result.status.value,
                    "accepted": result.accepted,
                    "message": result.message,
                    "filled_quantity": result.filled_quantity,
                    "remaining_quantity": result.remaining_quantity,
                },
            )
            connection.execute("COMMIT")
        except ValueError as exc:
            if str(exc) == "LEDGER_EVENT_ID_EXISTS":
                expected = {
                    "idempotency_key": idempotency_key,
                    "broker_order_id": result.broker_order_id,
                    "status": result.status.value,
                    "accepted": result.accepted,
                    "message": result.message,
                    "filled_quantity": result.filled_quantity,
                    "remaining_quantity": result.remaining_quantity,
                }
                existing_events = [
                    event for event in self.ledger.read()
                    if event.event_id
                    == self._result_event_id(event_type, idempotency_key, result)
                ]
                connection.execute("ROLLBACK")
                connection.close()
                if existing_events and existing_events[0].payload == expected:
                    return
                raise ValueError("EXECUTION_AUDIT_EVENT_CONFLICT") from exc
            try:
                connection.execute("ROLLBACK")
            finally:
                connection.close()
            raise
        except Exception:
            try:
                connection.execute("ROLLBACK")
            finally:
                connection.close()
            raise
        else:
            connection.close()

    @staticmethod
    def _attempt_from_intent(
        idempotency_key: str,
        status: BrokerOrderStatus,
        result: BrokerOrderResult,
    ) -> ExecutionAttempt:
        intent_status = (
            IntentStatus.TERMINAL
            if status
            in {
                BrokerOrderStatus.FILLED,
                BrokerOrderStatus.CANCELED,
                BrokerOrderStatus.REJECTED,
            }
            else (
                IntentStatus.UNKNOWN
                if status is BrokerOrderStatus.UNKNOWN
                else IntentStatus.SUBMITTED
            )
        )
        return ExecutionAttempt(idempotency_key, intent_status, result)

    def enforce_reconciliation_safety(self) -> bool:
        """Inspect execution state and durably halt on any discrepancy."""
        consistency = inspect_execution_consistency(self.store, self.ledger)
        if consistency.healthy:
            return False
        reason = reconciliation_halt_reason(consistency)
        connection = self._transaction()
        try:
            connection.execute(
                """
                INSERT INTO execution_halt(id, active, reason)
                VALUES (1, 1, ?)
                ON CONFLICT(id) DO UPDATE SET active = 1, reason = excluded.reason
                """,
                (reason,),
            )
            try:
                self.ledger._append_in_connection(
                    connection,
                    "EXECUTION_HALT_ASSERTED",
                    self._event_id("EXECUTION_HALT_ASSERTED", reason),
                    {"reason": reason, "errors": list(consistency.errors)},
                )
            except ValueError as exc:
                if str(exc) != "LEDGER_EVENT_ID_EXISTS":
                    raise
            connection.execute("COMMIT")
        except Exception:
            try:
                connection.execute("ROLLBACK")
            finally:
                connection.close()
            raise
        else:
            connection.close()
        return True

    def clear_halt(self, operator_reference: str) -> None:
        """Explicitly clear a durable halt only after clean reconciliation."""
        if not operator_reference:
            raise ValueError("EXECUTION_HALT_CLEAR_REFERENCE_REQUIRED")
        consistency = inspect_execution_consistency(self.store, self.ledger)
        if not consistency.healthy:
            raise RuntimeError(
                "EXECUTION_HALT_CLEAR_BLOCKED:" + "|".join(consistency.errors)
            )
        current_reason = self.store.halt_reason()
        if current_reason is None:
            return
        event_id = self._event_id(
            "EXECUTION_HALT_CLEARED",
            f"{current_reason}:{operator_reference}",
        )
        connection = self._transaction()
        try:
            connection.execute("UPDATE execution_halt SET active = 0 WHERE id = 1")
            try:
                self.ledger._append_in_connection(
                    connection,
                    "EXECUTION_HALT_CLEARED",
                    event_id,
                    {
                        "previous_reason": current_reason,
                        "operator_reference": operator_reference,
                    },
                )
            except ValueError as exc:
                if str(exc) != "LEDGER_EVENT_ID_EXISTS":
                    raise
            connection.execute("COMMIT")
        except Exception:
            try:
                connection.execute("ROLLBACK")
            finally:
                connection.close()
            raise
        else:
            connection.close()

    def prepare(self, request: BrokerOrderRequest) -> None:
        self._commit_intent_creation(request)

    def submit(self, request: BrokerOrderRequest) -> ExecutionAttempt:
        self.enforce_reconciliation_safety()
        if self.store.is_halted():
            reason = self.store.halt_reason() or "EXECUTION_HALTED"
            raise RuntimeError(f"EXECUTION_HALTED:{reason}")
        self.prepare(request)
        existing = self.store.get(request.idempotency_key)
        if existing is None:
            raise RuntimeError("EXECUTION_INTENT_NOT_PERSISTED")

        if existing.result is not None:
            return self._attempt_from_intent(
                request.idempotency_key,
                existing.status,
                existing.result,
            )

        try:
            result = self.broker.submit(request)
        except (ConnectionError, OSError, RuntimeError, TimeoutError):
            result = BrokerOrderResult(
                accepted=False,
                broker_order_id=None,
                status=BrokerOrderStatus.UNKNOWN,
                message="BROKER_SUBMISSION_UNKNOWN",
            )
            self._commit_result(
                request.idempotency_key, result, "EXECUTION_UNKNOWN"
            )
            return ExecutionAttempt(
                request.idempotency_key,
                IntentStatus.UNKNOWN,
                result,
            )

        self._commit_result(request.idempotency_key, result, "EXECUTION_RESULT")
        status = (
            IntentStatus.TERMINAL
            if result.status
            in {
                BrokerOrderStatus.FILLED,
                BrokerOrderStatus.CANCELED,
                BrokerOrderStatus.REJECTED,
            }
            else IntentStatus.SUBMITTED
        )
        return ExecutionAttempt(request.idempotency_key, status, result)

    def recover_unknown(self) -> tuple[ExecutionReconciliation, ...]:
        """Recover provider-accepted orders without ever resubmitting them.

        The provider must return an explicit client-key binding. A provider
        response for an unrequested key is treated as a reconciliation conflict.
        If the provider finds nothing, the durable local intent remains UNKNOWN.
        """

        keys = self.store.unknown_keys()
        if not keys:
            return ()

        recovered = self.broker.reconcile_by_idempotency_keys(list(keys))
        requested = set(keys)
        seen: set[str] = set()
        reconciliations: list[ExecutionReconciliation] = []

        for item in recovered:
            key = item.idempotency_key
            if key not in requested:
                raise ValueError("EXECUTION_RECOVERY_KEY_CONFLICT")
            if key in seen:
                raise ValueError("EXECUTION_RECOVERY_DUPLICATE_KEY")
            seen.add(key)

            intent = self.store.get(key)
            if intent is None or intent.status is not BrokerOrderStatus.UNKNOWN:
                raise ValueError("EXECUTION_RECOVERY_STATE_CONFLICT")

            self._commit_result(key, item.result, "EXECUTION_RECOVERED")
            reconciliations.append(ExecutionReconciliation(key, item.result))

        return tuple(reconciliations)

    def reconcile_discovered_orders(self) -> tuple[ExecutionReconciliation, ...]:
        """Reconcile provider orders discovered without relying on local state.

        A provider order with no matching durable client intent is a hard
        execution-boundary discrepancy: it is audited and the durable halt is
        asserted before any further autonomous submission can proceed.
        """
        discovered = self.broker.discover_open_orders()
        local_keys = set(self.store.all_keys())

        orphans = [
            item for item in discovered
            if item.idempotency_key is None
            or item.idempotency_key not in local_keys
        ]
        if orphans:
            errors = tuple(
                "EXECUTION_PROVIDER_ORDER_ORPHAN:"
                + (item.result.broker_order_id or "UNKNOWN")
                for item in orphans
            )
            self.enforce_reconciliation_safety_with_errors(
                ExecutionConsistency(False, errors)
            )
            raise RuntimeError("EXECUTION_PROVIDER_ORPHAN_DETECTED")

        reconciliations = []
        seen_keys: set[str] = set()
        for item in discovered:
            key = item.idempotency_key
            if key is None:
                continue
            if key in seen_keys:
                consistency = ExecutionConsistency(
                    False, (f"EXECUTION_PROVIDER_DUPLICATE_KEY:{key}",)
                )
                self.enforce_reconciliation_safety_with_errors(consistency)
                raise RuntimeError("EXECUTION_PROVIDER_DUPLICATE_KEY")
            seen_keys.add(key)
            intent = self.store.get(key)
            if intent is None:
                continue
            if intent.result is None:
                self._commit_result(key, item.result, "EXECUTION_RECONCILED")
                reconciliations.append(ExecutionReconciliation(key, item.result))
            elif intent.result != item.result:
                try:
                    self._commit_result(key, item.result, "EXECUTION_RECONCILED")
                except ValueError as exc:
                    if str(exc) not in {
                        "EXECUTION_TERMINAL_STATE_CONFLICT",
                        "EXECUTION_STATE_TRANSITION_CONFLICT",
                    }:
                        raise
                    consistency = ExecutionConsistency(
                        False,
                        (f"EXECUTION_PROVIDER_STATE_CONFLICT:{key}",),
                    )
                    self.enforce_reconciliation_safety_with_errors(consistency)
                    raise RuntimeError("EXECUTION_PROVIDER_STATE_CONFLICT") from exc
                reconciliations.append(ExecutionReconciliation(key, item.result))
        return tuple(reconciliations)

    def enforce_reconciliation_safety_with_errors(
        self, consistency: ExecutionConsistency
    ) -> bool:
        """Durably halt using an externally constructed reconciliation result."""
        from .execution_reconciliation import reconciliation_halt_reason
        reason = reconciliation_halt_reason(consistency)
        connection = self._transaction()
        try:
            connection.execute(
                """
                INSERT INTO execution_halt(id, active, reason)
                VALUES (1, 1, ?)
                ON CONFLICT(id) DO UPDATE SET active = 1, reason = excluded.reason
                """,
                (reason,),
            )
            self.ledger._append_in_connection(
                connection,
                "EXECUTION_HALT_ASSERTED",
                self._event_id("EXECUTION_HALT_ASSERTED", reason),
                {"reason": reason, "errors": list(consistency.errors)},
            )
            connection.execute("COMMIT")
        except ValueError as exc:
            if str(exc) != "LEDGER_EVENT_ID_EXISTS":
                try:
                    connection.execute("ROLLBACK")
                finally:
                    connection.close()
                raise
            connection.execute("COMMIT")
        except Exception:
            try:
                connection.execute("ROLLBACK")
            finally:
                connection.close()
            raise
        else:
            connection.close()
        return True

    def reconcile(
        self,
        broker_order_ids: list[str],
    ) -> tuple[ExecutionReconciliation, ...]:
        results = self.broker.reconcile(broker_order_ids)
        reconciliations: list[ExecutionReconciliation] = []

        for result in results:
            if result.broker_order_id is None:
                continue
            for key in self.store.pending_keys():
                intent = self.store.get(key)
                if intent is None or intent.result is None:
                    continue
                if intent.result.broker_order_id != result.broker_order_id:
                    continue
                self._commit_result(
                    key, result, "EXECUTION_RECONCILED"
                )
                reconciliations.append(
                    ExecutionReconciliation(key, result)
                )
                break

        return tuple(reconciliations)
