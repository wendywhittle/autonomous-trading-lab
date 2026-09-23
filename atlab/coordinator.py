from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .broker import (
    BrokerAdapter,
    BrokerOrderRequest,
    BrokerOrderResult,
    BrokerOrderStatus,
)
from .execution import ExecutionIntentStore, ExecutionReconciliation
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
    """Coordinates durable intent state, provider calls, and audit events."""

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

    @staticmethod
    def _event_id(event_type: str, idempotency_key: str) -> str:
        return f"{event_type.lower()}-{idempotency_key}"

    def _audit(
        self,
        event_type: str,
        idempotency_key: str,
        payload: dict,
    ) -> None:
        event_id = self._event_id(event_type, idempotency_key)
        try:
            self.ledger.append(event_type, event_id, payload)
        except ValueError as exc:
            if str(exc) != "LEDGER_EVENT_ID_EXISTS":
                raise

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

    def prepare(self, request: BrokerOrderRequest) -> None:
        existing = self.store.get(request.idempotency_key)
        self.store.record(request)
        if existing is None:
            self._audit(
                "EXECUTION_INTENT_CREATED",
                request.idempotency_key,
                {
                    "idempotency_key": request.idempotency_key,
                    "symbol": request.symbol,
                    "side": request.side.value,
                    "quantity": request.quantity,
                    "price": request.price,
                },
            )

    def submit(self, request: BrokerOrderRequest) -> ExecutionAttempt:
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
            self.store.update_result(request.idempotency_key, result)
            self._audit(
                "EXECUTION_UNKNOWN",
                request.idempotency_key,
                {
                    "idempotency_key": request.idempotency_key,
                    "status": result.status.value,
                    "message": result.message,
                },
            )
            return ExecutionAttempt(
                request.idempotency_key,
                IntentStatus.UNKNOWN,
                result,
            )

        self.store.update_result(request.idempotency_key, result)
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
        self._audit(
            "EXECUTION_RESULT",
            request.idempotency_key,
            {
                "idempotency_key": request.idempotency_key,
                "broker_order_id": result.broker_order_id,
                "status": result.status.value,
                "accepted": result.accepted,
                "message": result.message,
            },
        )
        return ExecutionAttempt(request.idempotency_key, status, result)

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
                self.store.update_result(key, result)
                self._audit(
                    "EXECUTION_RECONCILED",
                    key,
                    {
                        "idempotency_key": key,
                        "broker_order_id": result.broker_order_id,
                        "status": result.status.value,
                        "accepted": result.accepted,
                        "message": result.message,
                    },
                )
                reconciliations.append(
                    ExecutionReconciliation(key, result)
                )
                break

        return tuple(reconciliations)
