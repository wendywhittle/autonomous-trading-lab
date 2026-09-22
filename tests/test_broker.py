from atlab.broker import (
    BrokerMode,
    BrokerOrderRequest,
    DisabledBroker,
    paper_request,
)
from atlab.models import OrderStatus, PaperOrder, Side


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
    assert result.status == "DISABLED"
    assert result.message == "LIVE_BROKER_DISABLED"
    assert result.broker_order_id is None


def test_disabled_broker_cannot_cancel_or_reconcile_live_orders():
    broker = DisabledBroker()

    assert not broker.cancel("broker-order-1")
    results = broker.reconcile(["broker-order-1", "broker-order-2"])

    assert len(results) == 2
    assert all(result.status == "DISABLED" for result in results)
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
