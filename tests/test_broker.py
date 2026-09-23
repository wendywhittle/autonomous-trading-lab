import pytest

from atlab.broker import (
    BrokerMode,
    BrokerOrderRequest,
    BrokerOrderResult,
    BrokerOrderStatus,
    DisabledBroker,
    IdempotentBroker,
    paper_request,
)
from atlab.models import OrderStatus, PaperOrder, Side


class CountingBroker:
    mode = BrokerMode.LIVE

    def __init__(self):
        self.submissions = 0

    def submit(self, request):
        self.submissions += 1
        return type(
            "Result",
            (),
            {
                "accepted": True,
                "broker_order_id": f"broker-{self.submissions}",
                "status": BrokerOrderStatus.ACCEPTED,
                "message": "ACCEPTED",
            },
        )()

    def cancel(self, broker_order_id):
        return True

    def get_order(self, broker_order_id):
        raise AssertionError("not used")

    def reconcile(self, broker_order_ids):
        return ()


def test_disabled_broker_never_accepts_orders():
    broker = DisabledBroker()
    request = BrokerOrderRequest(
        idempotency_key="order-1",
        symbol="TEST",
        side=Side.BUY,
        quantity=1,
        price=100,
    )

    result = broker.submit(request)

    assert broker.mode is BrokerMode.DISABLED
    assert not result.accepted
    assert result.status is BrokerOrderStatus.REJECTED
    assert result.message == "LIVE_BROKER_DISABLED"
    assert result.broker_order_id is None


def test_disabled_broker_cannot_cancel_or_reconcile_live_orders():
    broker = DisabledBroker()

    assert not broker.cancel("broker-order-1")
    results = broker.reconcile(["broker-order-1", "broker-order-2"])

    assert len(results) == 2
    assert all(result.status is BrokerOrderStatus.UNKNOWN for result in results)
    assert all(result.message == "LIVE_BROKER_DISABLED" for result in results)


def test_paper_order_maps_to_provider_independent_request():
    order = PaperOrder(
        order_id="paper-1",
        decision_id="decision-1",
        symbol="TEST",
        side=Side.BUY,
        quantity=2,
        fill_price=101.5,
        notional=203,
        status=OrderStatus.FILLED,
    )

    request = paper_request(order)

    assert request.idempotency_key == "paper-1"
    assert request.symbol == "TEST"
    assert request.side is Side.BUY
    assert request.quantity == 2
    assert request.price == 101.5


def test_idempotent_broker_submits_same_key_only_once():
    adapter = CountingBroker()
    broker = IdempotentBroker(adapter)
    request = BrokerOrderRequest(
        idempotency_key="intent-1",
        symbol="TEST",
        side=Side.BUY,
        quantity=1,
        price=100,
    )

    first = broker.submit(request)
    second = broker.submit(request)

    assert first == second
    assert adapter.submissions == 1


def test_idempotent_broker_rejects_conflicting_reuse_of_key():
    adapter = CountingBroker()
    broker = IdempotentBroker(adapter)
    first_request = BrokerOrderRequest(
        idempotency_key="intent-1",
        symbol="TEST",
        side=Side.BUY,
        quantity=1,
        price=100,
    )
    conflicting_request = BrokerOrderRequest(
        idempotency_key="intent-1",
        symbol="TEST",
        side=Side.SELL,
        quantity=1,
        price=100,
    )

    broker.submit(first_request)
    result = broker.submit(conflicting_request)

    assert not result.accepted
    assert result.status is BrokerOrderStatus.REJECTED
    assert result.message == "IDEMPOTENCY_KEY_CONFLICT"
    assert adapter.submissions == 1


def test_broker_result_rejects_accepted_status_conflict():
    with pytest.raises(ValueError, match="BROKER_RESULT_ACCEPTED_STATUS_CONFLICT"):
        BrokerOrderResult(
            accepted=False,
            broker_order_id="broker-1",
            status=BrokerOrderStatus.FILLED,
            message="forged",
        )


def test_broker_result_rejects_missing_order_id_for_accepted_status():
    with pytest.raises(ValueError, match="BROKER_RESULT_ORDER_ID_REQUIRED"):
        BrokerOrderResult(
            accepted=True,
            broker_order_id=None,
            status=BrokerOrderStatus.ACCEPTED,
            message="forged",
        )


def test_broker_result_rejects_rejected_status_marked_accepted():
    with pytest.raises(ValueError, match="BROKER_RESULT_ACCEPTANCE_STATUS_CONFLICT"):
        BrokerOrderResult(
            accepted=True,
            broker_order_id="broker-1",
            status=BrokerOrderStatus.REJECTED,
            message="forged",
        )
