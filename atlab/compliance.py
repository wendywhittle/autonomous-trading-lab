from __future__ import annotations

import math
from dataclasses import dataclass
import hashlib
import json
from enum import Enum

from .broker import BrokerMode, BrokerOrderRequest
from .models import Side


class ComplianceAction(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


@dataclass(frozen=True)
class CompliancePolicy:
    """Deterministic execution-policy controls.

    This is a control-plane boundary, not a claim of legal or regulatory
    compliance. Jurisdiction-specific rules can be added as explicit policy
    versions without delegating the decision to an LLM.
    """

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
    """Fail-closed deterministic pre-trade compliance boundary."""

    def __init__(self, policy: CompliancePolicy):
        self.policy = policy

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
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def evaluate(
        self, request: BrokerOrderRequest, mode: BrokerMode
    ) -> ComplianceDecision:
        p = self.policy
        fingerprint = self._fingerprint()
        if not p.active:
            return ComplianceDecision(
                ComplianceAction.BLOCK, "COMPLIANCE_POLICY_INACTIVE", p.policy_id, p.version, fingerprint
            )
        if not request.symbol or request.symbol in p.restricted_symbols:
            return ComplianceDecision(
                ComplianceAction.BLOCK, "COMPLIANCE_SYMBOL_RESTRICTED", p.policy_id, p.version
            )
        if p.allowed_symbols and request.symbol not in p.allowed_symbols:
            return ComplianceDecision(
                ComplianceAction.BLOCK, "COMPLIANCE_SYMBOL_NOT_ALLOWED", p.policy_id, p.version
            )
        if request.side not in p.allowed_sides:
            return ComplianceDecision(
                ComplianceAction.BLOCK, "COMPLIANCE_SIDE_NOT_ALLOWED", p.policy_id, p.version
            )
        if not math.isfinite(request.quantity) or request.quantity <= 0:
            return ComplianceDecision(
                ComplianceAction.BLOCK, "COMPLIANCE_INVALID_QUANTITY", p.policy_id, p.version
            )
        if request.price is not None and (
            not math.isfinite(request.price) or request.price <= 0
        ):
            return ComplianceDecision(
                ComplianceAction.BLOCK, "COMPLIANCE_INVALID_PRICE", p.policy_id, p.version
            )

        if request.price is not None:
            notional = request.quantity * request.price
            if not math.isfinite(notional) or notional <= 0:
                return ComplianceDecision(
                    ComplianceAction.BLOCK,
                    "COMPLIANCE_INVALID_NOTIONAL",
                    p.policy_id,
                    p.version,
                    fingerprint,
                )
            if notional > p.max_order_notional:
                return ComplianceDecision(
                    ComplianceAction.BLOCK,
                    "COMPLIANCE_ORDER_NOTIONAL_LIMIT",
                    p.policy_id,
                    p.version,
                )
            if p.review_order_notional is not None and notional >= p.review_order_notional:
                return ComplianceDecision(
                    ComplianceAction.REVIEW_REQUIRED,
                    "COMPLIANCE_REVIEW_THRESHOLD",
                    p.policy_id,
                    p.version,
                )

        return ComplianceDecision(
            ComplianceAction.ALLOW, "COMPLIANCE_ALLOWED", p.policy_id, p.version
        )
