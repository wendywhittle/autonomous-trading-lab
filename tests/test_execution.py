from atlab.broker import BrokerOrderRequest, BrokerOrderResult, BrokerOrderStatus
from atlab.execution import ExecutionIntentStore
from atlab.models import Side


def request(key="intent-1", side=Side.BUY, quantity=1, price=100):
    return BrokerOrderRequest(
        idempotency_key=key,
        symbol="TEST",
        side=side,
        quantity=quantity,
        price=price,
    )


def test_execution_intent_survives_restart(tmp_path):
    path = tmp_path / "intents.json"
    store = ExecutionIntentStore(path)
    original = request()

    recorded = store.record(original)
    restarted = ExecutionIntentStore(path)

    assert recorded.request == original
    assert restarted.get("intent-1").request == original
    assert restarted.pending_keys() == ("intent-1",)


def test_execution_intent_is_idempotent(tmp_path):
    store = ExecutionIntentStore(tmp_path / "intents.json")
    original = request()

    assert store.record(original).request == original
    assert store.record(original).request == original


def test_execution_intent_rejects_conflicting_reuse(tmp_path):
    store = ExecutionIntentStore(tmp_path / "intents.json")
    store.record(request())

    try:
        store.record(request(side=Side.SELL))
    except ValueError as exc:
        assert str(exc) == "EXECUTION_INTENT_CONFLICT"
    else:
        raise AssertionError("expected execution intent conflict")


def test_execution_intent_does_not_claim_external_acceptance(tmp_path):
    store = ExecutionIntentStore(tmp_path / "intents.json")
    intent = store.record(request())

    assert intent.idempotency_key == "intent-1"
    assert store.get("intent-1").request == request()


def test_execution_rejects_broker_order_id_rebinding(tmp_path):
    store = ExecutionIntentStore(tmp_path / "intents.sqlite3")
    store.record(request())
    store.update_result(
        "intent-1",
        BrokerOrderResult(
            accepted=True,
            broker_order_id="broker-1",
            status=BrokerOrderStatus.ACCEPTED,
            message="ACCEPTED",
        ),
    )

    with pytest.raises(ValueError, match="EXECUTION_BROKER_ORDER_ID_CONFLICT"):
        store.update_result(
            "intent-1",
            BrokerOrderResult(
                accepted=True,
                broker_order_id="broker-evil",
                status=BrokerOrderStatus.ACCEPTED,
                message="ACCEPTED",
            ),
        )

    assert store.get("intent-1").result.broker_order_id == "broker-1"
