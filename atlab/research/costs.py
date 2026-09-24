"""Explicit all-in cost models.

Every research result in this lab is reported net of execution costs. A
:class:`CostModel` captures the four costs the Sniper v1 spec requires —
fees, spread, and slippage — as a flat fee per order plus proportional
basis-point charges applied to the traded notional. The model is deliberately
simple and legible: if the cost assumptions are wrong, they are wrong in
exactly one place.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

_BPS = 10_000.0


@dataclass(frozen=True)
class CostModel:
    """All-in execution cost assumptions, in account currency and basis points."""

    name: str
    fee_flat: float = 0.0  #: flat fee charged per filled order
    fee_bps: float = 0.0  #: proportional broker/exchange fee, in basis points
    slippage_bps: float = 0.0  #: adverse price move vs the signal price, in bps
    spread_bps: float = 0.0  #: spread paid when crossing, in bps

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("COST_MODEL_NAME_REQUIRED")
        for field in ("fee_flat", "fee_bps", "slippage_bps", "spread_bps"):
            if getattr(self, field) < 0:
                raise ValueError(f"COST_MODEL_NEGATIVE_{field.upper()}")

    def per_trade_cost(self, notional: float) -> float:
        """All-in cost of one filled order of the given notional."""
        if notional < 0:
            raise ValueError("COST_MODEL_NEGATIVE_NOTIONAL")
        proportional = notional * (self.fee_bps + self.slippage_bps + self.spread_bps) / _BPS
        return self.fee_flat + proportional

    def total_cost(self, num_trades: int, avg_notional: float) -> float:
        """All-in cost of ``num_trades`` orders averaging ``avg_notional``."""
        if num_trades < 0:
            raise ValueError("COST_MODEL_NEGATIVE_TRADES")
        return num_trades * self.per_trade_cost(avg_notional)

    def fingerprint(self) -> str:
        """Content hash identifying these exact cost assumptions."""
        payload = {
            "fee_bps": self.fee_bps,
            "fee_flat": self.fee_flat,
            "name": self.name,
            "slippage_bps": self.slippage_bps,
            "spread_bps": self.spread_bps,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
