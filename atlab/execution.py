from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .broker import BrokerOrderRequest, BrokerOrderResult, BrokerOrderStatus


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
    """Atomic JSON state for durable execution intent lifecycle."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("INVALID_EXECUTION_INTENT_STATE")
        return data

    @staticmethod
    def _encode_request(request: BrokerOrderRequest) -> dict[str, Any]:
        return {
            "idempotency_key": request.idempotency_key,
            "symbol": request.symbol,
            "side": request.side.value,
            "quantity": request.quantity,
            "price": request.price,
        }

    @staticmethod
    def _encode_result(result: BrokerOrderResult | None) -> dict[str, Any] | None:
        if result is None:
            return None
        return {
            "accepted": result.accepted,
            "broker_order_id": result.broker_order_id,
            "status": result.status.value,
            "message": result.message,
        }

    @staticmethod
    def _decode_result(data: dict[str, Any] | None) -> BrokerOrderResult | None:
        if data is None:
            return None
        return BrokerOrderResult(
            accepted=bool(data["accepted"]),
            broker_order_id=data["broker_order_id"],
            status=BrokerOrderStatus(data["status"]),
            message=str(data["message"]),
        )

    def _write(self, data: dict[str, Any]) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(data, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def record(self, request: BrokerOrderRequest) -> ExecutionIntent:
        data = self._read()
        key = request.idempotency_key
        existing = data.get(key)

        if existing is not None:
            if existing["request"] != self._encode_request(request):
                raise ValueError("EXECUTION_INTENT_CONFLICT")
            return self.get(key)

        data[key] = {
            "request": self._encode_request(request),
            "status": BrokerOrderStatus.UNKNOWN.value,
            "result": None,
        }
        self._write(data)
        return self.get(key)

    def update_result(
        self,
        idempotency_key: str,
        result: BrokerOrderResult,
    ) -> ExecutionIntent:
        data = self._read()
        existing = data.get(idempotency_key)
        if existing is None:
            raise KeyError("EXECUTION_INTENT_NOT_FOUND")

        existing["status"] = result.status.value
        existing["result"] = self._encode_result(result)
        data[idempotency_key] = existing
        self._write(data)
        return self.get(idempotency_key)

    def get(self, idempotency_key: str) -> ExecutionIntent | None:
        data = self._read()
        encoded = data.get(idempotency_key)
        if encoded is None:
            return None

        from .models import Side

        request_data = encoded["request"]
        request = BrokerOrderRequest(
            idempotency_key=request_data["idempotency_key"],
            symbol=request_data["symbol"],
            side=Side(request_data["side"]),
            quantity=float(request_data["quantity"]),
            price=(
                float(request_data["price"])
                if request_data["price"] is not None
                else None
            ),
        )
        result = self._decode_result(encoded.get("result"))
        return ExecutionIntent(
            idempotency_key=idempotency_key,
            request=request,
            status=BrokerOrderStatus(encoded["status"]),
            result=result,
        )

    def pending_keys(self) -> tuple[str, ...]:
        data = self._read()
        return tuple(
            sorted(
                key
                for key, value in data.items()
                if value["status"] in {
                    BrokerOrderStatus.UNKNOWN.value,
                    BrokerOrderStatus.ACCEPTED.value,
                    BrokerOrderStatus.PARTIALLY_FILLED.value,
                }
            )
        )


@dataclass(frozen=True)
class ExecutionReconciliation:
    idempotency_key: str
    result: BrokerOrderResult
