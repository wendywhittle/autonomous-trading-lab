from dataclasses import dataclass

from .models import DecisionAction, JEVDecision, RiskDecision, Side


@dataclass(frozen=True)
class RiskLimits:
    max_position_notional: float = 10_000
    max_order_notional: float = 2_500
    max_daily_loss: float = 500
    max_drawdown: float = 1_000
    max_leverage: float = 1.0


class DeterministicRiskEngine:
    """Pure risk gate. Identical inputs produce identical decisions."""

    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()
        if self.limits.max_leverage < 0:
            raise ValueError("INVALID_MAX_LEVERAGE")

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
        available_cash: float | None = None,
    ) -> RiskDecision:
        if price <= 0 or quantity <= 0:
            return RiskDecision(approved=False, reason="INVALID_ORDER_SIZE", max_notional=0)
        if current_position_notional < 0:
            return RiskDecision(approved=False, reason="INVALID_POSITION", max_notional=0)
        if available_cash is not None and available_cash < 0:
            return RiskDecision(approved=False, reason="INVALID_AVAILABLE_CASH", max_notional=0)
        if equity is not None and equity < 0:
            return RiskDecision(approved=False, reason="INVALID_EQUITY", max_notional=0)
        if equity is not None and session_start_equity is not None:
            if session_start_equity < 0:
                return RiskDecision(approved=False, reason="INVALID_EQUITY", max_notional=0)
            if session_start_equity - equity > self.limits.max_daily_loss:
                return RiskDecision(approved=False, reason="DAILY_LOSS_LIMIT", max_notional=0)
        if equity is not None and high_water_mark is not None:
            if high_water_mark < 0:
                return RiskDecision(approved=False, reason="INVALID_EQUITY", max_notional=0)
            if high_water_mark - equity > self.limits.max_drawdown:
                return RiskDecision(approved=False, reason="DRAWDOWN_LIMIT", max_notional=0)
        expected_side = {
            DecisionAction.HOLD: None,
            DecisionAction.ENTER: Side.BUY,
            DecisionAction.EXIT: Side.SELL,
        }[decision.action]
        if decision.side is not None and decision.side is not expected_side:
            return RiskDecision(approved=False, reason="INVALID_DECISION_SIDE", max_notional=0)
        if decision.action is not DecisionAction.HOLD and decision.side is None:
            return RiskDecision(approved=False, reason="INVALID_DECISION_SIDE", max_notional=0)
        if decision.action is DecisionAction.HOLD:
            return RiskDecision(approved=False, reason="HOLD_DECISION", max_notional=0)

        notional = price * quantity
        if notional > self.limits.max_order_notional:
            return RiskDecision(
                approved=False, reason="ORDER_NOTIONAL_LIMIT",
                max_notional=self.limits.max_order_notional,
            )

        if decision.side is Side.SELL:
            if notional > current_position_notional:
                return RiskDecision(
                    approved=False, reason="INSUFFICIENT_POSITION",
                    max_notional=current_position_notional,
                )
            projected_position = current_position_notional - notional
        else:
            if available_cash is not None and notional > available_cash:
                return RiskDecision(
                    approved=False, reason="INSUFFICIENT_CASH",
                    max_notional=max(0.0, available_cash),
                )
            projected_position = current_position_notional + notional

        if equity is not None:
            if equity == 0:
                return RiskDecision(approved=False, reason="ZERO_EQUITY", max_notional=0)
            if projected_position > equity * self.limits.max_leverage:
                return RiskDecision(
                    approved=False, reason="LEVERAGE_LIMIT",
                    max_notional=max(0.0, equity * self.limits.max_leverage),
                )

        if projected_position > self.limits.max_position_notional:
            return RiskDecision(
                approved=False, reason="POSITION_NOTIONAL_LIMIT",
                max_notional=self.limits.max_position_notional,
            )

        return RiskDecision(
            approved=True, reason="APPROVED", max_notional=self.limits.max_order_notional
        )
