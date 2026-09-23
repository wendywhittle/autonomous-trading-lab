from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum

from .broker import BrokerMode, BrokerOrderRequest
from .models import Side


class ComplianceAction(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


@dataclass(frozen=True)
class CompliancePolicy:
    """Deterministic execution-policy controls, not a legal compliance claim."""

    policy_id: str
    version: str
    active: bool = True
    allowed_symbols: frozenset[str] = frozenset()
    restricted_symbols: frozenset[str] = frozenset()
    allowed_sides: frozenset[Side] = frozenset({Side.BUY, Side.SELL})
    max_order_notional: float = 2500.0
    review_order_notional: float | None = None


@dataclass(frozen=True)
class ComplianceDecision:
    action: ComplianceAction
    reason: str
    policy_id: str
    policy_version: str
    policy_fingerprint: str


class ComplianceEngine:
    """Fail-closed deterministic pre-trade compliance boundary.

    The engine is immutable after construction. A policy change therefore
    requires constructing a new engine explicitly; an in-flight decision
    cannot be silently retargeted by mutating the existing control plane.
    """

    def __init__(self, policy: CompliancePolicy):
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("COMPLIANCE_ENGINE_IMMUTABLE")
        object.__setattr__(self, name, value)

    def _fingerprint(self) -> str:
        p = self.policy
        payload = {
            "policy_id": p.policy_id,
            "version": p.version,
            "active": p.active,
            "allowed_symbols": sorted(p.allowed_symbols),
            "restricted_symbols": sorted(p.restricted_symbols),
            "allowed_sides": sorted(side.value for side in p.allowed_sides),
            "max_order_notional": p.max_order_notional,
            "review_order_notional": p.review_order_notional,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def evaluate(self, request: BrokerOrderRequest, mode: BrokerMode) -> ComplianceDecision:
        p = self.policy
        fingerprint = self._fingerprint()

        def decision(action: ComplianceAction, reason: str) -> ComplianceDecision:
            return ComplianceDecision(action, reason, p.policy_id, p.version, fingerprint)

        if not p.active:
            return decision(ComplianceAction.BLOCK, "COMPLIANCE_POLICY_INACTIVE")
        if not request.symbol or request.symbol in p.restricted_symbols:
            return decision(ComplianceAction.BLOCK, "COMPLIANCE_SYMBOL_RESTRICTED")
        if p.allowed_symbols and request.symbol not in p.allowed_symbols:
            return decision(ComplianceAction.BLOCK, "COMPLIANCE_SYMBOL_NOT_ALLOWED")
        if request.side not in p.allowed_sides:
            return decision(ComplianceAction.BLOCK, "COMPLIANCE_SIDE_NOT_ALLOWED")
        if not math.isfinite(request.quantity) or request.quantity <= 0:
            return decision(ComplianceAction.BLOCK, "COMPLIANCE_INVALID_QUANTITY")
        if request.price is not None and (
            not math.isfinite(request.price) or request.price <= 0
        ):
            return decision(ComplianceAction.BLOCK, "COMPLIANCE_INVALID_PRICE")

        effective_price = request.price if request.price is not None else request.reference_price
        if effective_price is None:
            return decision(ComplianceAction.BLOCK, "COMPLIANCE_REFERENCE_PRICE_REQUIRED")

        if not math.isfinite(effective_price) or effective_price <= 0:
            return decision(ComplianceAction.BLOCK, "COMPLIANCE_INVALID_REFERENCE_PRICE")

        if effective_price is not None:
            notional = request.quantity * effective_price
            if not math.isfinite(notional) or notional <= 0:
                return decision(ComplianceAction.BLOCK, "COMPLIANCE_INVALID_NOTIONAL")
            if notional > p.max_order_notional:
                return decision(ComplianceAction.BLOCK, "COMPLIANCE_ORDER_NOTIONAL_LIMIT")
            if p.review_order_notional is not None and notional >= p.review_order_notional:
                return decision(ComplianceAction.REVIEW_REQUIRED, "COMPLIANCE_REVIEW_THRESHOLD")

        return decision(ComplianceAction.ALLOW, "COMPLIANCE_ALLOWED")
