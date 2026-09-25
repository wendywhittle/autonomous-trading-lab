"""Mock-only tests for the Alpaca paper broker adapter.

No test in this file touches the network or uses real credentials. The
HTTP transport is stubbed at the ``urlopen`` boundary; everything else
exercises the real adapter code.
"""

import json
from urllib.error import HTTPError

import pytest

from atlab.broker import (
    BrokerMode,
    BrokerOrderRequest,
    BrokerOrderStatus,
)
from atlab.brokers import (
    ALPACA_LIVE_BASE_URL,
    ALPACA_PAPER_BASE_URL,
    AlpacaBroker,
    alpaca_idempotency_key,
)
from atlab.models import Side


def request(key="live-decision-1", quantity=0.02, price=650.0):
    return BrokerOrderRequest(
        idempotency_key=key,
        symbol="SPY",
        side=Side.BUY,
        quantity=quantity,
        price=price,
        decision_id="decision-1",
    )


def make_broker(**kwargs):
    kwargs.setdefault("api_key_id", "test-key-id")
    kwargs.setdefault("api_secret_key", "test-secret")
    return AlpacaBroker(**kwargs)


class FakeResponse:
    def __init__(self, status, body):
        self.status = status
        self._body = body

    def read(self):
        return json.dumps(self._body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class ScriptedTransport:
    """Stubs urlopen: records requests, replays scripted responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    def __call__(self, urlopen_request, timeout=None):
        self.requests.append(urlopen_request)
        if not self._responses:
            raise AssertionError("unexpected HTTP request")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        status, body = item
        return FakeResponse(status, body)


def patch_transport(monkeypatch, responses):
    transport = ScriptedTransport(responses)
    monkeypatch.setattr("atlab.brokers.alpaca.urlopen", transport)
    return transport


def headers_of(urlopen_request):
    return {key.lower(): value for key, value in urlopen_request.header_items()}


# --- construction ----------------------------------------------------


def test_defaults_to_paper_base_url():
    broker = make_broker()
    assert broker.mode is BrokerMode.PAPER
    assert broker._base_url == ALPACA_PAPER_BASE_URL == "https://paper-api.alpaca.markets"


def test_missing_credentials_fail_closed(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(ValueError, match="ALPACA_CREDENTIALS_MISSING"):
        AlpacaBroker()


def test_credentials_read_from_environment(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "env-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "env-secret")
    broker = AlpacaBroker()
    assert broker._api_key_id == "env-key"
    assert broker._api_secret_key == "env-secret"


def test_live_mode_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="ALPACA_LIVE_NOT_ALLOWED"):
        make_broker(mode=BrokerMode.LIVE)
    broker = make_broker(mode=BrokerMode.LIVE, allow_live=True)
    assert broker.mode is BrokerMode.LIVE
    assert broker._base_url == ALPACA_LIVE_BASE_URL == "https://api.alpaca.markets"


def test_paper_mode_rejects_live_url():
    with pytest.raises(ValueError, match="ALPACA_LIVE_URL_IN_PAPER_MODE"):
        make_broker(base_url=ALPACA_LIVE_BASE_URL)


def test_credentials_never_logged():
    broker = make_broker()
    assert "test-secret" not in repr(broker)


# --- idempotency-key contract (M14) ------------------------------------


def test_idempotency_key_contract():
    assert alpaca_idempotency_key("decision-1") == "live-decision-1"
    with pytest.raises(ValueError, match="INVALID_DECISION_ID"):
        alpaca_idempotency_key("")


# --- submit ------------------------------------------------------------


def test_submit_sends_limit_order_with_client_order_id(monkeypatch):
    transport = patch_transport(
        monkeypatch, [(200, {"id": "order-1", "status": "accepted"})]
    )
    broker = make_broker()

    result = broker.submit(request())

    assert result.accepted is True
    assert result.broker_order_id == "order-1"
    assert result.status is BrokerOrderStatus.ACCEPTED
    assert len(transport.requests) == 1

    sent = transport.requests[0]
    assert sent.full_url == ALPACA_PAPER_BASE_URL + "/v2/orders"
    assert sent.get_method() == "POST"
    body = json.loads(sent.data.decode("utf-8"))
    assert body == {
        "symbol": "SPY",
        "qty": "0.02",
        "side": "buy",
        "type": "limit",
        "time_in_force": "day",
        "limit_price": "650.0",
        "client_order_id": "live-decision-1",
    }
    headers = headers_of(sent)
    assert headers["apca-api-key-id"] == "test-key-id"
    assert headers["apca-api-secret-key"] == "test-secret"
    # Credentials travel in headers only, never in the body.
    assert "test-secret" not in sent.data.decode("utf-8")


def test_submit_refuses_missing_limit_price(monkeypatch):
    transport = patch_transport(monkeypatch, [])
    broker = make_broker()
    req = request()
    object.__setattr__(req, "price", None)
    with pytest.raises(ValueError, match="ALPACA_LIMIT_PRICE_REQUIRED"):
        broker.submit(req)
    assert transport.requests == []


@pytest.mark.parametrize(
    "provider_status,qty_fields,expected,accepted",
    [
        ("new", {}, BrokerOrderStatus.ACCEPTED, True),
        ("accepted", {}, BrokerOrderStatus.ACCEPTED, True),
        ("pending_new", {}, BrokerOrderStatus.ACCEPTED, True),
        (
            "filled",
            {"filled_qty": "0.02", "qty": "0.02"},
            BrokerOrderStatus.FILLED,
            True,
        ),
        (
            "partially_filled",
            {"filled_qty": "0.01", "qty": "0.02"},
            BrokerOrderStatus.PARTIALLY_FILLED,
            True,
        ),
        ("rejected", {}, BrokerOrderStatus.REJECTED, False),
        ("expired", {}, BrokerOrderStatus.REJECTED, False),
        ("canceled", {}, BrokerOrderStatus.CANCELED, True),
        ("pending_cancel", {}, BrokerOrderStatus.UNKNOWN, False),
        ("stopped", {}, BrokerOrderStatus.UNKNOWN, False),
        ("nonsense", {}, BrokerOrderStatus.UNKNOWN, False),
    ],
)
def test_submit_maps_provider_states(
    monkeypatch, provider_status, qty_fields, expected, accepted
):
    body = {"id": "order-9", "status": provider_status}
    body.update(qty_fields)
    patch_transport(monkeypatch, [(200, body)])
    result = make_broker().submit(request())
    assert result.status is expected
    assert result.accepted is accepted
    assert result.broker_order_id == "order-9"


def test_submit_filled_carries_fill_quantities(monkeypatch):
    patch_transport(
        monkeypatch,
        [(200, {"id": "o", "status": "filled", "filled_qty": "0.02", "qty": "0.02"})],
    )
    result = make_broker().submit(request())
    assert result.status is BrokerOrderStatus.FILLED
    assert result.filled_quantity == pytest.approx(0.02)
    assert result.remaining_quantity == pytest.approx(0.0)


def test_submit_partially_filled_carries_fill_quantities(monkeypatch):
    patch_transport(
        monkeypatch,
        [
            (
                200,
                {
                    "id": "o",
                    "status": "partially_filled",
                    "filled_qty": "0.01",
                    "qty": "0.02",
                },
            )
        ],
    )
    result = make_broker().submit(request())
    assert result.status is BrokerOrderStatus.PARTIALLY_FILLED
    assert result.filled_quantity == pytest.approx(0.01)
    assert result.remaining_quantity == pytest.approx(0.01)


def test_submit_4xx_is_a_rejection_not_a_retry(monkeypatch):
    transport = patch_transport(
        monkeypatch, [(422, {"code": 42210000, "message": "insufficient funds"})]
    )
    result = make_broker().submit(request())
    assert result.accepted is False
    assert result.status is BrokerOrderStatus.REJECTED
    assert result.broker_order_id is None
    assert len(transport.requests) == 1


def test_submit_5xx_raises_without_retry(monkeypatch):
    transport = patch_transport(monkeypatch, [(503, {"message": "unavailable"})])
    with pytest.raises(RuntimeError, match="ALPACA_REQUEST_FAILED"):
        make_broker().submit(request())
    assert len(transport.requests) == 1


def test_rate_limit_429_raises_coded_error_without_retry(monkeypatch):
    transport = patch_transport(
        monkeypatch,
        [HTTPError(ALPACA_PAPER_BASE_URL + "/v2/orders", 429, "Too Many Requests", {}, None)],
    )
    with pytest.raises(RuntimeError, match="ALPACA_RATE_LIMITED"):
        make_broker().submit(request())
    assert len(transport.requests) == 1


# --- cancel / get_order / reconcile -------------------------------------


def test_cancel_success_and_definitive_failure(monkeypatch):
    transport = patch_transport(monkeypatch, [(200, {}), (404, {})])
    broker = make_broker()
    assert broker.cancel("order-1") is True
    assert broker.cancel("order-missing") is False
    assert [r.get_method() for r in transport.requests] == ["DELETE", "DELETE"]
    assert transport.requests[0].full_url.endswith("/v2/orders/order-1")


def test_get_order_returns_mapped_result(monkeypatch):
    patch_transport(
        monkeypatch, [(200, {"id": "order-1", "status": "filled", "filled_qty": "1", "qty": "1"})]
    )
    result = make_broker().get_order("order-1")
    assert result.status is BrokerOrderStatus.FILLED
    assert result.accepted is True


def test_get_order_404_is_unknown(monkeypatch):
    patch_transport(monkeypatch, [(404, {})])
    result = make_broker().get_order("order-missing")
    assert result.accepted is False
    assert result.status is BrokerOrderStatus.UNKNOWN
    assert result.broker_order_id == "order-missing"


def test_reconcile_fetches_each_order_id(monkeypatch):
    transport = patch_transport(
        monkeypatch,
        [
            (200, {"id": "a", "status": "accepted"}),
            (200, {"id": "b", "status": "canceled"}),
        ],
    )
    results = make_broker().reconcile(["a", "b"])
    assert [r.broker_order_id for r in results] == ["a", "b"]
    assert [r.status for r in results] == [
        BrokerOrderStatus.ACCEPTED,
        BrokerOrderStatus.CANCELED,
    ]
    assert len(transport.requests) == 2


# --- discovery / recovery ------------------------------------------------


def test_discover_open_orders_maps_client_order_ids(monkeypatch):
    patch_transport(
        monkeypatch,
        [
            (
                200,
                [
                    {"id": "o1", "status": "accepted", "client_order_id": "live-a"},
                    {"id": "o2", "status": "new", "client_order_id": None},
                ],
            )
        ],
    )
    discovered = make_broker().discover_open_orders()
    assert [d.idempotency_key for d in discovered] == ["live-a", None]
    assert [d.result.status for d in discovered] == [
        BrokerOrderStatus.ACCEPTED,
        BrokerOrderStatus.ACCEPTED,
    ]
    assert discovered[0].result.broker_order_id == "o1"


def test_recover_by_idempotency_keys_matches_locally(monkeypatch):
    patch_transport(
        monkeypatch,
        [
            (
                200,
                [
                    {"id": "o1", "status": "accepted", "client_order_id": "live-a"},
                    {"id": "o2", "status": "filled", "filled_qty": "1", "qty": "1",
                     "client_order_id": "live-b"},
                    {"id": "o3", "status": "accepted", "client_order_id": "live-unwanted"},
                ],
            )
        ],
    )
    recovered = make_broker().reconcile_by_idempotency_keys(["live-a", "live-b", "live-missing"])
    assert [r.idempotency_key for r in recovered] == ["live-a", "live-b"]
    assert recovered[1].result.status is BrokerOrderStatus.FILLED
    # Unrequested keys are never returned, even when the provider knows them.
    assert all(r.idempotency_key != "live-unwanted" for r in recovered)
