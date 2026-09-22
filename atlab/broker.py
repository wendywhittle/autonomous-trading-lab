from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Protocol

from .models import PaperOrder, Side


class BrokerMode(str, Enum):
    DISABLED = "DISABLED"
    PAPER = "PAPER"
    LIVE = "LIVE"


class BrokerOrderStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    CANCELED = "CANCELED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class BrokerOrderRequest:
    idempotency_key: str
    symbol: str
    side: Side
    quantity: float
    price: float | None = None

    def __post_init__(self) -> None:
        if not self.idempotency_key:
            raise ValueError("INVALID_IDEMPOTENCY_KEY")
        if not self.symbol:
            raise ValueError("INVALID_SYMBOL")
        if self.quantity <= 0:
            raise ValueError("INVALID_QUANTITY")
        if self.price is not None and self.price <= 0:
            raise ValueError("INVALID_PRICE")


@dataclass(frozen=True)
class BrokerOrderResult:
    accepted: bool
    broker_order_id: str | None
    status: BrokerOrderStatus
    message: str


class BrokerAdapter(Protocol):
    """Provider-independent execution contract."""

    mode: BrokerMode

    def submit(self, request: BrokerOrderRequest) -> BrokerOrderResult: ...

    def cancel(self, broker_order_id: str) -> bool: ...

    def get_order(self, broker_order_id: str) -> BrokerOrderResult: ...

    def reconcile(self, broker_order_ids: list[str]) -> tuple[BrokerOrderResult, ...]: ...


class IdempotentBroker:
    """Process-local idempotency guard for a provider adapter.

    A durable implementation must persist the key/request/result mapping and
    reconcile externally after a crash between provider submission and local
    persistence. This guard prevents duplicate submission while one process is
    alive; it does not claim to solve the external crash window.
    """

    def __init__(self, adapter: BrokerAdapter):
        self.adapter = adapter
        self.mode = adapter.mode
        self._lock = Lock()
        self._requests: dict[str, BrokerOrderRequest] = {}
        self._results: dict[str, BrokerOrderResult] = {}

    def submit(self, request: BrokerOrderRequest) -> BrokerOrderResult:
        with self._lock:
            previous = self._requests.get(request.idempotency_key)
            if previous is not None:
                if previous != request:
                    return BrokerOrderResult(
                        accepted=False,
                        broker_order_id=None,
                        status=BrokerOrderStatus.REJECTED,
                        message="IDEMPOTENCY_KEY_CONFLICT",
                    )
                return self._results[request.idempotency_key]

            result = self.adapter.submit(request)
            self._requests[request.idempotency_key] = request
            self._results[request.idempotency_key] = result
            return result

    def cancel(self, broker_order_id: str) -> bool:
        return self.adapter.cancel(broker_order_id)

    def get_order(self, broker_order_id: str) -> BrokerOrderResult:
        return self.adapter.get_order(broker_order_id)

    def reconcile(self, broker_order_ids: list[str]) -> tuple[BrokerOrderResult, ...]:
        return self.adapter.reconcile(broker_order_ids)


class DisabledBroker:
    """Explicitly non-executing adapter used until live execution is authorized."""

    mode = BrokerMode.DISABLED

    def submit(self, request: BrokerOrderRequest) -> BrokerOrderResult:
        return BrokerOrderResult(
            accepted=False,
            broker_order_id=None,
            status=BrokerOrderStatus.REJECTED,
            message="LIVE_BROKER_DISABLED",
        )

    def cancel(self, broker_order_id: str) -> bool:
        return False

    def get_order(self, broker_order_id: str) -> BrokerOrderResult:
        return BrokerOrderResult(
            accepted=False,
            broker_order_id=broker_order_id,
            status=BrokerOrderStatus.UNKNOWN,
            message="LIVE_BROKER_DISABLED",
        )

    def reconcile(self, broker_order_ids: list[str]) -> tuple[BrokerOrderResult, ...]:
        return tuple(self.get_order(order_id) for order_id in broker_order_ids)


def paper_request(order: PaperOrder) -> BrokerOrderRequest:
    return BrokerOrderRequest(
        idempotency_key=order.order_id,
        symbol=order.symbol,
        side=order.side,
        quantity=order.quantity,
        price=order.fill_price,
    )
