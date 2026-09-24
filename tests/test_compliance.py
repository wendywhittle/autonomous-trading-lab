import pytest

from atlab.broker import BrokerMode, BrokerOrderRequest
from atlab.compliance import ComplianceAction, ComplianceEngine, CompliancePolicy
from atlab.models import Side


def request(symbol="TEST", quantity=1, price=100):
    return BrokerOrderRequest(
        idempotency_key="compliance-1",
        symbol=symbol,
        side=Side.BUY,
        quantity=quantity,
        price=price,
    )


def test_policy_fingerprint_changes_when_policy_changes():
    first = ComplianceEngine(
        CompliancePolicy(policy_id="p", version="1", max_order_notional=1000)
    )
    second = ComplianceEngine(
        CompliancePolicy(policy_id="p", version="2", max_order_notional=1000)
    )
    a = first.evaluate(request(), BrokerMode.PAPER)
    b = second.evaluate(request(), BrokerMode.PAPER)
    assert a.action is ComplianceAction.ALLOW
    assert a.policy_fingerprint != b.policy_fingerprint
    assert b.policy_version == "2"


def test_review_required_is_not_execution_allow():
    engine = ComplianceEngine(
        CompliancePolicy(
            policy_id="p",
            version="1",
            max_order_notional=1000,
            review_order_notional=100,
        )
    )
    decision = engine.evaluate(request(), BrokerMode.PAPER)
    assert decision.action is ComplianceAction.REVIEW_REQUIRED


def test_inactive_policy_fails_closed():
    engine = ComplianceEngine(
        CompliancePolicy(policy_id="p", version="1", active=False)
    )
    decision = engine.evaluate(request(), BrokerMode.PAPER)
    assert decision.action is ComplianceAction.BLOCK
    assert decision.reason == "COMPLIANCE_POLICY_INACTIVE"


def test_compliance_engine_rejects_runtime_policy_mutation():
    engine = ComplianceEngine(CompliancePolicy(policy_id="p", version="1"))

    with pytest.raises(AttributeError, match="COMPLIANCE_ENGINE_IMMUTABLE"):
        engine.policy = CompliancePolicy(policy_id="p", version="2")

    assert engine.policy.version == "1"


# --- H14: reason-asserting tests for every compliance deny branch. ---

def test_restricted_symbol_is_blocked():
    engine = ComplianceEngine(
        CompliancePolicy(
            policy_id="p", version="1", restricted_symbols=frozenset({"TEST"})
        )
    )
    decision = engine.evaluate(request(), BrokerMode.PAPER)
    assert decision.action is ComplianceAction.BLOCK
    assert decision.reason == "COMPLIANCE_SYMBOL_RESTRICTED"


def test_symbol_outside_allow_list_is_blocked():
    engine = ComplianceEngine(
        CompliancePolicy(
            policy_id="p", version="1", allowed_symbols=frozenset({"AAA"})
        )
    )
    decision = engine.evaluate(request(), BrokerMode.PAPER)
    assert decision.action is ComplianceAction.BLOCK
    assert decision.reason == "COMPLIANCE_SYMBOL_NOT_ALLOWED"


def test_disallowed_side_is_blocked():
    engine = ComplianceEngine(
        CompliancePolicy(
            policy_id="p", version="1", allowed_sides=frozenset({Side.SELL})
        )
    )
    decision = engine.evaluate(request(), BrokerMode.PAPER)
    assert decision.action is ComplianceAction.BLOCK
    assert decision.reason == "COMPLIANCE_SIDE_NOT_ALLOWED"


def test_non_finite_quantity_is_blocked():
    engine = ComplianceEngine(CompliancePolicy(policy_id="p", version="1"))
    decision = engine.evaluate(
        request(quantity=float("inf")), BrokerMode.PAPER
    )
    assert decision.action is ComplianceAction.BLOCK
    assert decision.reason == "COMPLIANCE_INVALID_QUANTITY"


def test_non_finite_price_is_blocked():
    engine = ComplianceEngine(CompliancePolicy(policy_id="p", version="1"))
    decision = engine.evaluate(request(price=float("inf")), BrokerMode.PAPER)
    assert decision.action is ComplianceAction.BLOCK
    assert decision.reason == "COMPLIANCE_INVALID_PRICE"


def test_missing_reference_price_is_blocked():
    engine = ComplianceEngine(CompliancePolicy(policy_id="p", version="1"))
    decision = engine.evaluate(request(price=None), BrokerMode.PAPER)
    assert decision.action is ComplianceAction.BLOCK
    assert decision.reason == "COMPLIANCE_REFERENCE_PRICE_REQUIRED"


def test_reference_price_satisfies_price_requirement():
    engine = ComplianceEngine(CompliancePolicy(policy_id="p", version="1"))
    priced = BrokerOrderRequest(
        idempotency_key="compliance-ref",
        symbol="TEST",
        side=Side.BUY,
        quantity=1,
        price=None,
        reference_price=100,
    )
    decision = engine.evaluate(priced, BrokerMode.PAPER)
    assert decision.action is ComplianceAction.ALLOW
    assert decision.reason == "COMPLIANCE_ALLOWED"


def test_order_notional_limit_is_blocked():
    engine = ComplianceEngine(
        CompliancePolicy(policy_id="p", version="1", max_order_notional=1000)
    )
    decision = engine.evaluate(
        request(quantity=20, price=100), BrokerMode.PAPER
    )
    assert decision.action is ComplianceAction.BLOCK
    assert decision.reason == "COMPLIANCE_ORDER_NOTIONAL_LIMIT"


def test_allow_list_permits_matching_symbol_and_side():
    engine = ComplianceEngine(
        CompliancePolicy(
            policy_id="p",
            version="1",
            allowed_symbols=frozenset({"TEST"}),
            allowed_sides=frozenset({Side.BUY}),
        )
    )
    decision = engine.evaluate(request(), BrokerMode.PAPER)
    assert decision.action is ComplianceAction.ALLOW
    assert decision.reason == "COMPLIANCE_ALLOWED"
