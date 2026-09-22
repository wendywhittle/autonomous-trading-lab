from dataclasses import dataclass

from .models import DecisionAction, JEVDecision, RiskDecision, Side


@dataclass(frozen=True)
class RiskLimits:
    max_position_notional: float = 10_000
    max_order_notional: float = 2_500
    max_daily_loss: float = 500
    max_drawdown: float = 1_000


class DeterministicRiskEngine:
    """Pure risk gate. Identical inputs produce identical decisions."""

    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()

    def evaluate(
        self,
        decision: JEVDecision,
        price: float,
        quantity: float,
        current_position_notional: float = 0,
        *,
        equity: float | None = None,
        session_start_equity: float | None = None,
        high_water_mark: float | None = None,
    ) -> RiskDecision:
        if price <= 0 or quantity <= 0:
            return RiskDecision(approved=False, reason="INVALID_ORDER_SIZE", max_notional=0)
        if current_position_notional < 0:
            return RiskDecision(approved=False, reason="INVALID_POSITION", max_notional=0)

        if equity is not None and session_start_equity is not None:
            daily_loss = session_start_equity - equity
            if daily_loss > self.limits.max_daily_loss:
                return RiskDecision(
                    approved=False,
                    reason="DAILY_LOSS_LIMIT",
                    max_notional=0,
                )

        if equity is not None and high_water_mark is not None:
            drawdown = high_water_mark - equity
            if drawdown > self.limits.max_drawdown:
                return RiskDecision(
                    approved=False,
                    reason="DRAWDOWN_LIMIT",
                    max_notional=0,
                )

        if decision.action is DecisionAction.HOLD:
            return RiskDecision(approved=False, reason="HOLD_DECISION", max_notional=0)

        notional = price * quantity
        if notional > self.limits.max_order_notional:
            return RiskDecision(
                approved=False,
                reason="ORDER_NOTIONAL_LIMIT",
                max_notional=self.limits.max_order_notional,
            )

        if decision.side is Side.SELL:
            if notional > current_position_notional:
                return RiskDecision(
                    approved=False,
                    reason="INSUFFICIENT_POSITION",
                    max_notional=current_position_notional,
                )
            projected_position = current_position_notional - notional
        else:
            projected_position = current_position_notional + notional

        if projected_position > self.limits.max_position_notional:
            return RiskDecision(
                approved=False,
                reason="POSITION_NOTIONAL_LIMIT",
                max_notional=self.limits.max_position_notional,
            )

        return RiskDecision(
            approved=True,
            reason="APPROVED",
            max_notional=self.limits.max_order_notional,
        )
