from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .broker import BrokerOrderRequest, BrokerOrderResult


@dataclass(frozen=True)
class ExecutionIntent:
    """Durable intent to submit one broker order.

    An intent records what the system meant to submit before any external
    provider call. It is not proof that an order was accepted or filled.
    """

    idempotency_key: str
    request: BrokerOrderRequest


class ExecutionIntentStore:
    """Append-free, atomic JSON state for durable execution intents."""

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

    def record(self, request: BrokerOrderRequest) -> ExecutionIntent:
        data = self._read()
        key = request.idempotency_key
        existing = data.get(key)
        encoded_request = request.__dict__.copy()
        encoded_request["side"] = request.side.value

        if existing is not None:
            if existing != encoded_request:
                raise ValueError("EXECUTION_INTENT_CONFLICT")
            return ExecutionIntent(key, request)

        data[key] = encoded_request
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(data, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(self.path)
        return ExecutionIntent(key, request)

    def get(self, idempotency_key: str) -> ExecutionIntent | None:
        data = self._read()
        encoded = data.get(idempotency_key)
        if encoded is None:
            return None
        from .models import Side

        return ExecutionIntent(
            idempotency_key=idempotency_key,
            request=BrokerOrderRequest(
                idempotency_key=idempotency_key,
                symbol=encoded["symbol"],
                side=Side(encoded["side"]),
                quantity=float(encoded["quantity"]),
                price=float(encoded["price"]) if encoded["price"] is not None else None,
            ),
        )

    def pending_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._read()))


@dataclass(frozen=True)
class ExecutionReconciliation:
    idempotency_key: str
    result: BrokerOrderResult
