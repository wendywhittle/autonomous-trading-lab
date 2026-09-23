from .models import JEVDecision, OrderStatus, PaperOrder


class PaperExecution:
    """Deterministic simulated execution. This class never submits a live order."""

    def __init__(self, kill_switch: bool = False):
        self.kill_switch = kill_switch

    def submit(self, decision: JEVDecision, quantity: float, price: float) -> PaperOrder:
        if self.kill_switch:
            raise RuntimeError("KILL_SWITCH_ACTIVE")
        if decision.side is None:
            raise ValueError("DECISION_HAS_NO_EXECUTABLE_SIDE")
        return PaperOrder(order_id=f"paper-{decision.decision_id}", decision_id=decision.decision_id, symbol=decision.symbol, side=decision.side, quantity=quantity, fill_price=price, notional=quantity * price, status=OrderStatus.FILLED)
