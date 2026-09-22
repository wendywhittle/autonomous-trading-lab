from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .broker import (
    BrokerAdapter,
    BrokerOrderRequest,
    BrokerOrderResult,
    BrokerOrderStatus,
)
from .execution import ExecutionIntentStore


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
    """Coordinates durable intent creation with provider submission.

    A provider timeout or process failure must leave the intent recoverable;
    it must never be interpreted locally as a fill.
    """

    def __init__(self, store: ExecutionIntentStore, broker: BrokerAdapter):
        self.store = store
        self.broker = broker

    def prepare(self, request: BrokerOrderRequest) -> None:
        self.store.record(request)

    def submit(self, request: BrokerOrderRequest) -> ExecutionAttempt:
        self.store.record(request)
        try:
            result = self.broker.submit(request)
        except Exception:
            return ExecutionAttempt(
                request.idempotency_key,
                IntentStatus.UNKNOWN,
                BrokerOrderResult(
                    accepted=False,
                    broker_order_id=None,
                    status=BrokerOrderStatus.UNKNOWN,
                    message="BROKER_SUBMISSION_UNKNOWN",
                ),
            )

        if result.status.value in {"FILLED", "CANCELED", "REJECTED"}:
            status = IntentStatus.TERMINAL
        else:
            status = IntentStatus.SUBMITTED
        return ExecutionAttempt(request.idempotency_key, status, result)

    def reconcile(self, broker_order_ids: list[str]) -> tuple[BrokerOrderResult, ...]:
        return self.broker.reconcile(broker_order_ids)
