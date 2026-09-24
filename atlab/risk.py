from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .models import DecisionAction, RiskDecision, Side
from .risk_state import RiskStateSnapshot


@dataclass(frozen=True)
class RiskLimits:
    max_position_notional: float = 10_000
    max_order_notional: float = 2_500
    max_daily_loss: float = 500
    max_drawdown: float = 1_000
    max_leverage: float = 1.0

    def __post_init__(self) -> None:
        # Canonicalize numerics to float so fingerprints are stable across
        # SQLite round-trips (which return floats for stored integers),
        # mirroring RiskStateSnapshot.
        for field in (
            "max_position_notional",
            "max_order_notional",
            "max_daily_loss",
            "max_drawdown",
            "max_leverage",
        ):
            object.__setattr__(self, field, float(getattr(self, field)))

    def fingerprint(self) -> str:
        payload = {
            "max_position_notional": self.max_position_notional,
            "max_order_notional": self.max_order_notional,
            "max_daily_loss": self.max_daily_loss,
            "max_drawdown": self.max_drawdown,
            "max_leverage": self.max_leverage,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class DeterministicRiskEngine:
    """Pure risk gate. Identical inputs produce identical decisions."""

    ENGINE_VERSION = "deterministic-risk-v2"

    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()
        if self.limits.max_leverage < 0:
            raise ValueError("INVALID_MAX_LEVERAGE")

    def _result(self, decision, price, quantity, current_position_notional,
                approved, reason, max_notional, *, equity, session_start_equity,
                high_water_mark, available_cash, risk_state_fingerprint=None):
        limits_fingerprint = self.limits.fingerprint()
        evidence = RiskDecision(
            approved=approved, reason=reason, engine_version=self.ENGINE_VERSION,
            max_notional=max_notional,
            decision_id=decision.decision_id, action=decision.action,
            symbol=decision.symbol, side=decision.side, quantity=quantity,
            price=price, current_position_notional=current_position_notional,
            equity=equity, session_start_equity=session_start_equity,
            high_water_mark=high_water_mark, available_cash=available_cash,
            risk_limits_fingerprint=limits_fingerprint,
            risk_state_fingerprint=risk_state_fingerprint,
        )
        return evidence.model_copy(
            update={"risk_fingerprint": self.fingerprint_for(evidence)}
        )

    @classmethod
    def fingerprint_for(cls, decision: RiskDecision) -> str:
        required = (
            decision.decision_id, decision.action, decision.symbol,
            decision.quantity, decision.price, decision.current_position_notional,
            decision.risk_limits_fingerprint,
        )
        if decision.engine_version != cls.ENGINE_VERSION:
            raise ValueError("RISK_ENGINE_VERSION_CONFLICT")
        if any(value is None for value in required):
            raise ValueError("RISK_EVIDENCE_INCOMPLETE")
        payload = {
            "engine_version": decision.engine_version,
            "approved": decision.approved, "reason": decision.reason,
            "max_notional": decision.max_notional,
            "decision_id": decision.decision_id,
            "action": decision.action.value,
            "symbol": decision.symbol,
            "side": decision.side.value if decision.side else None,
            "quantity": decision.quantity, "price": decision.price,
            "current_position_notional": decision.current_position_notional,
            "equity": decision.equity,
            "session_start_equity": decision.session_start_equity,
            "high_water_mark": decision.high_water_mark,
            "available_cash": decision.available_cash,
            "risk_limits_fingerprint": decision.risk_limits_fingerprint,
            "risk_state_fingerprint": decision.risk_state_fingerprint,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @classmethod
    def validate_evidence(cls, decision: RiskDecision) -> None:
        if not decision.approved:
            raise RuntimeError("RISK_DECISION_NOT_APPROVED")
        if not decision.risk_fingerprint:
            raise RuntimeError("RISK_EVIDENCE_MISSING")
        if cls.fingerprint_for(decision) != decision.risk_fingerprint:
            raise RuntimeError("RISK_EVIDENCE_TAMPERED")

    def evaluate(self, decision, price, quantity, current_position_notional=0, *,
                 equity=None, session_start_equity=None, high_water_mark=None,
                 available_cash=None, risk_state: RiskStateSnapshot | None = None):
        if risk_state is not None and (
            risk_state.symbol != decision.symbol
            or risk_state.current_position_notional != current_position_notional
            or risk_state.equity != equity
            or risk_state.session_start_equity != session_start_equity
            or risk_state.high_water_mark != high_water_mark
            or risk_state.available_cash != available_cash
        ):
            raise ValueError("RISK_STATE_INPUT_MISMATCH")
        risk_state_fingerprint = risk_state.fingerprint() if risk_state else None

        def result(approved, reason, max_notional):
            return self._result(
                decision, price, quantity, current_position_notional,
                approved, reason, max_notional,
                equity=equity, session_start_equity=session_start_equity,
                high_water_mark=high_water_mark, available_cash=available_cash,
                risk_state_fingerprint=risk_state_fingerprint,
            )

        if price <= 0 or quantity <= 0:
            return result(False, "INVALID_ORDER_SIZE", 0)
        if current_position_notional < 0:
            return result(False, "INVALID_POSITION", 0)
        if available_cash is not None and available_cash < 0:
            return result(False, "INVALID_AVAILABLE_CASH", 0)
        if equity is not None and equity < 0:
            return result(False, "INVALID_EQUITY", 0)
        if equity is not None and session_start_equity is not None:
            if session_start_equity < 0:
                return result(False, "INVALID_EQUITY", 0)
            if session_start_equity - equity > self.limits.max_daily_loss:
                return result(False, "DAILY_LOSS_LIMIT", 0)
        if equity is not None and high_water_mark is not None:
            if high_water_mark < 0:
                return result(False, "INVALID_EQUITY", 0)
            if high_water_mark - equity > self.limits.max_drawdown:
                return result(False, "DRAWDOWN_LIMIT", 0)

        expected_side = {
            DecisionAction.HOLD: None, DecisionAction.ENTER: Side.BUY,
            DecisionAction.EXIT: Side.SELL,
        }[decision.action]
        if decision.side is not None and decision.side is not expected_side:
            return result(False, "INVALID_DECISION_SIDE", 0)
        if decision.action is not DecisionAction.HOLD and decision.side is None:
            return result(False, "INVALID_DECISION_SIDE", 0)
        if decision.action is DecisionAction.HOLD:
            return result(False, "HOLD_DECISION", 0)

        notional = price * quantity
        if notional > self.limits.max_order_notional:
            return result(False, "ORDER_NOTIONAL_LIMIT", self.limits.max_order_notional)

        if decision.side is Side.SELL:
            if notional > current_position_notional:
                return result(False, "INSUFFICIENT_POSITION", current_position_notional)
            projected_position = current_position_notional - notional
        else:
            if available_cash is not None and notional > available_cash:
                return result(False, "INSUFFICIENT_CASH", max(0.0, available_cash))
            projected_position = current_position_notional + notional

        if equity is not None:
            if equity == 0:
                return result(False, "ZERO_EQUITY", 0)
            if projected_position > equity * self.limits.max_leverage:
                return result(False, "LEVERAGE_LIMIT",
                              max(0.0, equity * self.limits.max_leverage))
        if projected_position > self.limits.max_position_notional:
            return result(False, "POSITION_NOTIONAL_LIMIT",
                          self.limits.max_position_notional)
        return result(True, "APPROVED", self.limits.max_order_notional)
