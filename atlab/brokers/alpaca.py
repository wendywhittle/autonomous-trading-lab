"""Alpaca broker adapter (paper trading).

Targets the Alpaca **paper** API only. The live base URL is unreachable
unless the caller explicitly opts in with ``allow_live=True``; there is no
ambient path to live-money execution.

Idempotency-key contract (M14)
------------------------------
The live-path idempotency key for a trading intent is::

    live-{decision_id}

It is unique per trading intent and stable across retries and restarts. It
is sent to Alpaca as ``client_order_id`` so a retried submission after a
crash reconciles to the same provider order instead of creating a
duplicate. Build it with :func:`alpaca_idempotency_key` — never hand-roll
the prefix. Callers are responsible for stamping the key onto the
:class:`~atlab.broker.BrokerOrderRequest` before submitting; the adapter
forwards ``request.idempotency_key`` verbatim as ``client_order_id``.

Credentials
-----------
Read from the ``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY`` environment
variables at construction time (or passed explicitly). They are never
logged, never written to files, and never leave the HTTPS request headers.
Construction fails closed with ``ALPACA_CREDENTIALS_MISSING`` when absent.

Rate limits
-----------
An HTTP 429 raises ``ALPACA_RATE_LIMITED`` immediately. The adapter never
retries in a loop; the caller (and the coordinator's UNKNOWN-result path)
decides what happens next.
"""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..broker import (
    BrokerDiscoveredOrder,
    BrokerMode,
    BrokerOrderRequest,
    BrokerOrderResult,
    BrokerOrderStatus,
    BrokerRecoveryResult,
)
from ..models import Side

ALPACA_PAPER_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_BASE_URL = "https://api.alpaca.markets"

_API_KEY_ENV = "APCA_API_KEY_ID"
_API_SECRET_ENV = "APCA_API_SECRET_KEY"

# Alpaca order states -> lab status. Anything unlisted (pending_cancel,
# stopped, suspended, calculated, held, done_for_day, ...) maps to UNKNOWN:
# fail closed rather than guessing at a transitional provider state.
_STATUS_MAP = {
    "new": BrokerOrderStatus.ACCEPTED,
    "accepted": BrokerOrderStatus.ACCEPTED,
    "pending_new": BrokerOrderStatus.ACCEPTED,
    "filled": BrokerOrderStatus.FILLED,
    "partially_filled": BrokerOrderStatus.PARTIALLY_FILLED,
    "rejected": BrokerOrderStatus.REJECTED,
    "expired": BrokerOrderStatus.REJECTED,
    "canceled": BrokerOrderStatus.CANCELED,
}


def alpaca_idempotency_key(decision_id: str) -> str:
    """Build the live-path idempotency key for a trading intent (M14).

    Contract: ``live-{decision_id}`` — unique per intent, stable across
    retries and restarts, sent to Alpaca as ``client_order_id``.
    """
    if not isinstance(decision_id, str) or not decision_id:
        raise ValueError("INVALID_DECISION_ID")
    return f"live-{decision_id}"


def _parse_json(body: bytes) -> object:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("ALPACA_BAD_RESPONSE") from exc
    if not text.strip():
        return {}
    try:
        return json.loads(text)
    except ValueError as exc:
        raise RuntimeError("ALPACA_BAD_RESPONSE") from exc


class AlpacaBroker:
    """Alpaca adapter. Paper by default; LIVE only via explicit opt-in."""

    mode: BrokerMode

    def __init__(
        self,
        mode: BrokerMode = BrokerMode.PAPER,
        *,
        base_url: str | None = None,
        api_key_id: str | None = None,
        api_secret_key: str | None = None,
        allow_live: bool = False,
        timeout_seconds: float = 10.0,
    ) -> None:
        if mode is BrokerMode.LIVE and not allow_live:
            raise ValueError("ALPACA_LIVE_NOT_ALLOWED")
        if mode is BrokerMode.DISABLED:
            raise ValueError("ALPACA_MODE_DISABLED_UNSUPPORTED")
        resolved = (base_url or "").rstrip("/") or (
            ALPACA_LIVE_BASE_URL if mode is BrokerMode.LIVE else ALPACA_PAPER_BASE_URL
        )
        if mode is BrokerMode.PAPER and resolved == ALPACA_LIVE_BASE_URL:
            raise ValueError("ALPACA_LIVE_URL_IN_PAPER_MODE")
        key_id = api_key_id if api_key_id is not None else os.environ.get(_API_KEY_ENV)
        secret = (
            api_secret_key if api_secret_key is not None else os.environ.get(_API_SECRET_ENV)
        )
        if not key_id or not secret:
            raise ValueError("ALPACA_CREDENTIALS_MISSING")
        if timeout_seconds <= 0:
            raise ValueError("ALPACA_INVALID_TIMEOUT")
        self.mode = mode
        self._base_url = resolved
        # Credentials live only on the instance, only ever sent as HTTPS
        # headers. They are never logged, never repr'd, never persisted.
        self._api_key_id = key_id
        self._api_secret_key = secret
        self._timeout_seconds = timeout_seconds

    def __repr__(self) -> str:  # pragma: no cover - defensive
        return f"AlpacaBroker(mode={self.mode.value}, base_url={self._base_url})"

    # -- transport -----------------------------------------------------
    def _request(self, method: str, path: str, payload: dict | None = None) -> tuple[int, object]:
        """Return (http_status, parsed_json). Raises coded errors only."""
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        request = Request(
            self._base_url + path,
            data=data,
            method=method,
            headers={
                "APCA-API-KEY-ID": self._api_key_id,
                "APCA-API-SECRET-KEY": self._api_secret_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                return response.status, _parse_json(response.read())
        except HTTPError as exc:
            if exc.code == 429:
                # Hard stop: never retry a rate limit in a loop.
                raise RuntimeError("ALPACA_RATE_LIMITED") from exc
            try:
                body = exc.read()
            except (OSError, ValueError):
                body = b""
            return exc.code, _parse_json(body)
        except (URLError, OSError) as exc:
            raise RuntimeError("ALPACA_REQUEST_FAILED") from exc

    # -- order mapping --------------------------------------------------
    @staticmethod
    def _result_from_order(order: dict) -> BrokerOrderResult:
        # Defensive against malformed provider payloads: fail closed with a
        # coded error rather than raising AttributeError deep in mapping.
        order_id = order.get("id") if isinstance(order, dict) else None
        if not order_id or not isinstance(order_id, str):
            raise RuntimeError("ALPACA_BAD_RESPONSE")
        status = _STATUS_MAP.get(str(order.get("status", "")).lower(), BrokerOrderStatus.UNKNOWN)

        def _num(value: object) -> float | None:
            try:
                number = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None
            return number

        if status is BrokerOrderStatus.FILLED:
            filled = _num(order.get("filled_qty")) or _num(order.get("qty")) or 0.0
            if filled <= 0:
                status = BrokerOrderStatus.UNKNOWN
            else:
                return BrokerOrderResult(
                    accepted=True,
                    broker_order_id=order_id,
                    status=status,
                    message="ALPACA_FILLED",
                    filled_quantity=filled,
                    remaining_quantity=0.0,
                )
        if status is BrokerOrderStatus.PARTIALLY_FILLED:
            filled = _num(order.get("filled_qty")) or 0.0
            total = _num(order.get("qty")) or 0.0
            remaining = total - filled
            if filled > 0 and remaining > 0:
                return BrokerOrderResult(
                    accepted=True,
                    broker_order_id=order_id,
                    status=status,
                    message="ALPACA_PARTIALLY_FILLED",
                    filled_quantity=filled,
                    remaining_quantity=remaining,
                )
            status = BrokerOrderStatus.UNKNOWN
        if status in (
            BrokerOrderStatus.ACCEPTED,
            BrokerOrderStatus.PARTIALLY_FILLED,
            BrokerOrderStatus.CANCELED,
        ):
            return BrokerOrderResult(
                accepted=True,
                broker_order_id=order_id,
                status=status,
                message=f"ALPACA_{status.value}",
            )
        # REJECTED and UNKNOWN: accepted=False. The provider order id is
        # kept when present for forensics; acceptance stays false.
        return BrokerOrderResult(
            accepted=False,
            broker_order_id=order_id,
            status=status,
            message=f"ALPACA_{status.value}",
        )

    # -- BrokerAdapter protocol -----------------------------------------
    def submit(self, request: BrokerOrderRequest) -> BrokerOrderResult:
        # Charter: limit orders only. A missing limit price is refused
        # locally rather than sent as a market order.
        if request.price is None:
            raise ValueError("ALPACA_LIMIT_PRICE_REQUIRED")
        if request.side not in (Side.BUY, Side.SELL):
            raise ValueError("ALPACA_INVALID_SIDE")
        payload = {
            "symbol": request.symbol,
            "qty": repr(request.quantity),  # fractional qty as decimal string
            "side": request.side.value.lower(),
            "type": "limit",
            "time_in_force": "day",
            "limit_price": repr(request.price),
            "client_order_id": request.idempotency_key,
        }
        status, body = self._request("POST", "/v2/orders", payload)
        if 200 <= status < 300:
            if not isinstance(body, dict):
                raise RuntimeError("ALPACA_BAD_RESPONSE")
            return self._result_from_order(body)
        if 400 <= status < 500:
            return BrokerOrderResult(
                accepted=False,
                broker_order_id=None,
                status=BrokerOrderStatus.REJECTED,
                message=f"ALPACA_ORDER_REJECTED:{status}",
            )
        raise RuntimeError(f"ALPACA_REQUEST_FAILED:{status}")

    def cancel(self, broker_order_id: str) -> bool:
        if not broker_order_id:
            raise ValueError("ALPACA_ORDER_ID_REQUIRED")
        status, _ = self._request("DELETE", f"/v2/orders/{broker_order_id}")
        if 200 <= status < 300:
            return True
        if status in (404, 422):
            # Not found or no longer cancelable: definitively not canceled
            # by this call.
            return False
        raise RuntimeError(f"ALPACA_REQUEST_FAILED:{status}")

    def get_order(self, broker_order_id: str) -> BrokerOrderResult:
        if not broker_order_id:
            raise ValueError("ALPACA_ORDER_ID_REQUIRED")
        status, body = self._request("GET", f"/v2/orders/{broker_order_id}")
        if status == 404:
            return BrokerOrderResult(
                accepted=False,
                broker_order_id=broker_order_id,
                status=BrokerOrderStatus.UNKNOWN,
                message="ALPACA_ORDER_NOT_FOUND",
            )
        if 200 <= status < 300:
            if not isinstance(body, dict):
                raise RuntimeError("ALPACA_BAD_RESPONSE")
            return self._result_from_order(body)
        raise RuntimeError(f"ALPACA_REQUEST_FAILED:{status}")

    def reconcile(self, broker_order_ids: list[str]) -> tuple[BrokerOrderResult, ...]:
        return tuple(self.get_order(order_id) for order_id in broker_order_ids)

    def discover_open_orders(self) -> tuple[BrokerDiscoveredOrder, ...]:
        status, body = self._request(
            "GET", "/v2/orders?status=open&limit=500&direction=desc"
        )
        if not 200 <= status < 300 or not isinstance(body, list):
            raise RuntimeError(f"ALPACA_REQUEST_FAILED:{status}")
        discovered = []
        for order in body:
            client_key = order.get("client_order_id") if isinstance(order, dict) else None
            discovered.append(
                BrokerDiscoveredOrder(
                    idempotency_key=(
                        client_key if isinstance(client_key, str) and client_key else None
                    ),
                    result=self._result_from_order(order),
                )
            )
        return tuple(discovered)

    def reconcile_by_idempotency_keys(
        self, idempotency_keys: list[str]
    ) -> tuple[BrokerRecoveryResult, ...]:
        # Alpaca has no server-side client_order_id filter; fetch recent
        # orders and match locally. Only requested keys are ever returned.
        status, body = self._request(
            "GET", "/v2/orders?status=all&limit=500&direction=desc"
        )
        if not 200 <= status < 300 or not isinstance(body, list):
            raise RuntimeError(f"ALPACA_REQUEST_FAILED:{status}")
        wanted = {key for key in idempotency_keys if key}
        found: dict[str, BrokerOrderResult] = {}
        for order in body:
            if not isinstance(order, dict):
                continue
            client_key = order.get("client_order_id")
            if client_key in wanted and client_key not in found:
                found[client_key] = self._result_from_order(order)
        return tuple(
            BrokerRecoveryResult(idempotency_key=key, result=found[key])
            for key in idempotency_keys
            if key in found
        )
