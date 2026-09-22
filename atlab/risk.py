from dataclasses import dataclass

from .models import DecisionAction, JEVDecision, RiskDecision


@dataclass(frozen=True)
class RiskLimits:
    max_position_notional: float = 10_000
    max_order_notional: float = 2_500


class DeterministicRiskEngine:
    """Pure risk gate. Identical inputs produce identical decisions."""

    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()

    def evaluate(self, decision: JEVDecision, price: float, quantity: float, current_position_notional: float = 0) -> RiskDecision:
        if price <= 0 or quantity <= 0:
            return RiskDecision(False, "INVALID_ORDER_SIZE", 0)
        if decision.action is DecisionAction.HOLD:
            return RiskDecision(False, "HOLD_DECISION", 0)
        notional = price * quantity
        if notional > self.limits.max_order_notional:
            return RiskDecision(False, "ORDER_NOTIONAL_LIMIT", self.limits.max_order_notional)
        if current_position_notional + notional > self.limits.max_position_notional:
            return RiskDecision(False, "POSITION_NOTIONAL_LIMIT", self.limits.max_position_notional)
        return RiskDecision(True, "APPROVED", min(self.limits.max_order_notional, self.limits.max_position_notional))
