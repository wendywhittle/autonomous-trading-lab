"""Wendy's live-trading risk charter, encoded as immutable code.

The charter is the cage: hard numeric limits the deterministic risk engine
enforces on every order. It is paper-first — these limits govern paper runs
today and the broker step later. Changing any number changes the charter
fingerprint, so the exact cage a run traded under is always auditable from
the ledger.

Worst-case loss under charter v1: $50.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .risk import RiskLimits

CHARTER_ID = "wendy-risk-charter-v1"
CHARTER_VERSION = "1.1.0"

# Hard limits (USD).
CHARTER_MAX_ORDER_NOTIONAL = 10.0  # max $ per single trade
CHARTER_MAX_POSITION_NOTIONAL = 40.0  # max $ deployed at once (per symbol run)
CHARTER_MAX_DAILY_LOSS = 15.0  # daily loss beyond this -> halt for the day
CHARTER_MAX_DRAWDOWN = 50.0  # total drawdown beyond this -> halt until manual re-enable
CHARTER_MAX_LEVERAGE = 1.0  # no leverage, ever
# v1.1: per-position stop-loss. The daily-loss and drawdown halts stop NEW
# entries; the stop-loss exits an open loser so a bleeding position cannot
# sit past the daily number unrealized.
CHARTER_STOP_LOSS = 5.0  # force-exit a position down $5 unrealized

# Charter terms enforced at the operator/broker layer (documented here as
# part of the charter; the paper engine trades the CSV universe it is given).
CHARTER_UNIVERSE = "liquid US stocks/ETFs only; no penny stocks; no options"
CHARTER_ORDER_TYPES = "limit orders only"
CHARTER_SESSION = "regular market hours only"


@dataclass(frozen=True)
class RiskCharter:
    """Immutable record of the agreed trading cage."""

    charter_id: str = CHARTER_ID
    version: str = CHARTER_VERSION
    max_order_notional: float = CHARTER_MAX_ORDER_NOTIONAL
    max_position_notional: float = CHARTER_MAX_POSITION_NOTIONAL
    max_daily_loss: float = CHARTER_MAX_DAILY_LOSS
    max_drawdown: float = CHARTER_MAX_DRAWDOWN
    max_leverage: float = CHARTER_MAX_LEVERAGE
    stop_loss: float = CHARTER_STOP_LOSS
    universe: str = CHARTER_UNIVERSE
    order_types: str = CHARTER_ORDER_TYPES
    session: str = CHARTER_SESSION

    def fingerprint(self) -> str:
        payload = {
            "charter_id": self.charter_id,
            "version": self.version,
            "max_order_notional": self.max_order_notional,
            "max_position_notional": self.max_position_notional,
            "max_daily_loss": self.max_daily_loss,
            "max_drawdown": self.max_drawdown,
            "max_leverage": self.max_leverage,
            "stop_loss": self.stop_loss,
            "universe": self.universe,
            "order_types": self.order_types,
            "session": self.session,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def charter_limits() -> RiskLimits:
    """Translate the charter into the deterministic risk engine's limits."""
    return RiskLimits(
        max_order_notional=CHARTER_MAX_ORDER_NOTIONAL,
        max_position_notional=CHARTER_MAX_POSITION_NOTIONAL,
        max_daily_loss=CHARTER_MAX_DAILY_LOSS,
        max_drawdown=CHARTER_MAX_DRAWDOWN,
        max_leverage=CHARTER_MAX_LEVERAGE,
    )


def charter_quantity_policy(
    target_notional: float = CHARTER_MAX_ORDER_NOTIONAL,
):
    """Size each order to approximately ``target_notional`` dollars.

    Returns a price -> quantity policy (fractional shares allowed, as the
    paper engine and Alpaca both support them). With a $10 target, every
    order notional is ~$10 regardless of the share price, so the
    max_order_notional gate binds uniformly.
    """

    def policy(price: float) -> float:
        if price <= 0:
            raise ValueError("INVALID_PRICE_FOR_QUANTITY_POLICY")
        return target_notional / price

    return policy
