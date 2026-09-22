from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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

    def state(self) -> dict[str, float]:
        return {
            "cash": self.cash,
            "position_quantity": self.position_quantity,
            "average_cost": self.average_cost,
            "realized_pnl": self.realized_pnl,
        }

    def save_state(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.state(), sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(target)

    def load_state(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        required = {"cash", "position_quantity", "average_cost", "realized_pnl"}
        if set(data) != required:
            raise ValueError("INVALID_PORTFOLIO_STATE")
        values = {key: float(data[key]) for key in required}
        if values["cash"] < 0 or values["position_quantity"] < 0 or values["average_cost"] < 0:
            raise ValueError("INVALID_PORTFOLIO_STATE")
        self.cash = values["cash"]
        self.position_quantity = values["position_quantity"]
        self.average_cost = values["average_cost"]
        self.realized_pnl = values["realized_pnl"]

    def snapshot(self, mark_price: float) -> PortfolioSnapshot:
        equity = self.cash + self.position_quantity * mark_price
        return PortfolioSnapshot(self.cash, self.position_quantity, mark_price, equity, self.realized_pnl)
