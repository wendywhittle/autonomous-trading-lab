from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from .models import PaperOrder, Side


class BrokerMode(str, Enum):
    DISABLED = "DISABLED"
    PAPER = "PAPER"
    LIVE = "LIVE"


@dataclass(frozen=True)
class BrokerOrderRequest:
    idempotency_key: str
    symbol: str
    side: Side
    quantity: float
    price: float | None = None


@dataclass(frozen=True)
class BrokerOrderResult:
    accepted: bool
    broker_order_id: str | None
    status: str
    message: str


class BrokerAdapter(Protocol):
    """Provider-independent execution contract."""

    mode: BrokerMode

    def submit(self, request: BrokerOrderRequest) -> BrokerOrderResult: ...

    def cancel(self, broker_order_id: str) -> bool: ...

    def get_order(self, broker_order_id: str) -> BrokerOrderResult: ...

    def reconcile(self, broker_order_ids: list[str]) -> tuple[BrokerOrderResult, ...]: ...


class DisabledBroker:
    """Explicitly non-executing adapter used until live execution is authorized."""

    mode = BrokerMode.DISABLED

    def submit(self, request: BrokerOrderRequest) -> BrokerOrderResult:
        return BrokerOrderResult(
            accepted=False,
            broker_order_id=None,
            status="DISABLED",
            message="LIVE_BROKER_DISABLED",
        )

    def cancel(self, broker_order_id: str) -> bool:
        return False

    def get_order(self, broker_order_id: str) -> BrokerOrderResult:
        return BrokerOrderResult(
            accepted=False,
            broker_order_id=broker_order_id,
            status="DISABLED",
            message="LIVE_BROKER_DISABLED",
        )

    def reconcile(self, broker_order_ids: list[str]) -> tuple[BrokerOrderResult, ...]:
        return tuple(
            self.get_order(order_id) for order_id in broker_order_ids
        )


def paper_request(order: PaperOrder) -> BrokerOrderRequest:
    return BrokerOrderRequest(
        idempotency_key=order.order_id,
        symbol=order.symbol,
        side=order.side,
        quantity=order.quantity,
        price=order.fill_price,
    )
