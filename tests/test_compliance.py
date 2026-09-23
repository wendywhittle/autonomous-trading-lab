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
