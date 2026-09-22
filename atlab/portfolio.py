from __future__ import annotations

from dataclasses import dataclass

from .models import PaperOrder, Side


@dataclass(frozen=True)
class PortfolioSnapshot:
    cash: float
    position_quantity: float
    mark_price: float
    equity: float
    realized_pnl: float


class PaperPortfolio:
    """Minimal deterministic accounting for paper fills."""
    def __init__(self, starting_cash: float):
        if starting_cash < 0:
            raise ValueError("INVALID_STARTING_CASH")
        self.cash = starting_cash
        self.position_quantity = 0.0
        self.average_cost = 0.0
        self.realized_pnl = 0.0

    def apply(self, order: PaperOrder) -> PortfolioSnapshot:
        if order.side is Side.BUY:
            new_qty = self.position_quantity + order.quantity
            self.average_cost = ((self.position_quantity * self.average_cost) + order.notional) / new_qty
            self.position_quantity = new_qty
            self.cash -= order.notional
        else:
            if order.quantity > self.position_quantity:
                raise ValueError("INSUFFICIENT_POSITION")
            self.realized_pnl += (order.fill_price - self.average_cost) * order.quantity
            self.position_quantity -= order.quantity
            self.cash += order.notional
            if self.position_quantity == 0:
                self.average_cost = 0.0
        return self.snapshot(order.fill_price)

    def snapshot(self, mark_price: float) -> PortfolioSnapshot:
        equity = self.cash + self.position_quantity * mark_price
        return PortfolioSnapshot(self.cash, self.position_quantity, mark_price, equity, self.realized_pnl)
