from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from .compliance import ComplianceAction, ComplianceEngine, CompliancePolicy
from .broker import (
    BrokerAdapter,
    BrokerMode,
    BrokerOrderRequest,
    BrokerOrderResult,
    BrokerOrderStatus,
)
from .execution import ExecutionIntentStore, ExecutionReconciliation
from .execution_reconciliation import (
    ExecutionConsistency,
    inspect_execution_consistency,
    reconciliation_halt_reason,
)
from .ledger import ImmutableLedger
from .promotion import ExecutionAuthorization, PromotionGate, PromotionMode
from .risk import DeterministicRiskEngine
from .risk_state import RiskStateSnapshot
from .models import DecisionAction, RiskDecision


class IntentStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    UNKNOWN = "UNKNOWN"
    TERMINAL = "TERMINAL"


@dataclass(frozen=True)
class ExecutionAttempt:
    idempotency_key: str
    status: IntentStatus
    result: BrokerOrderResult


class ExecutionCoordinator:
    """Coordinates durable intent state, provider calls, and atomic audit events."""

    def __init__(
        self,
        store: ExecutionIntentStore,
        broker: BrokerAdapter,
        ledger: ImmutableLedger,
        authorization: ExecutionAuthorization | None = None,
        compliance: ComplianceEngine | None = None,
        risk: DeterministicRiskEngine | None = None,
        risk_state_provider: Callable[[BrokerOrderRequest], RiskStateSnapshot] | None = None,
    ):
        if store.path.resolve() != ledger.path.resolve():
            raise ValueError("EXECUTION_AND_LEDGER_MUST_SHARE_DATABASE")
        self.store = store
        self._broker = broker
        self.ledger = ledger
        self._authorization = authorization
        self._compliance = compliance or ComplianceEngine(
            CompliancePolicy(policy_id="default", version="1")
        )
        self._risk = risk or DeterministicRiskEngine()
        self._risk_state_provider = risk_state_provider

    @property
    def broker(self) -> BrokerAdapter:
        return self._broker

    @property
    def authorization(self) -> ExecutionAuthorization | None:
        return self._authorization

    @property
    def compliance(self) -> ComplianceEngine:
        return self._compliance

    def _require_execution_authorization(self) -> ExecutionAuthorization | None:
        if getattr(self.broker, "mode", None) is not BrokerMode.LIVE:
            return None
        authorization = self._authorization
        if authorization is None or authorization.target is not PromotionMode.LIVE:
            raise RuntimeError("EXECUTION_AUTHORIZATION_REQUIRED")
        if not PromotionGate.validate_authorization(authorization):
            raise RuntimeError("EXECUTION_AUTHORIZATION_INVALID")
        if not self.store.authorization_is_active(authorization.authorization_id):
            raise RuntimeError("EXECUTION_AUTHORIZATION_NOT_ACTIVATED")
        return authorization

    def activate_execution_authorization(self, operator_reference: str) -> None:
        if not operator_reference:
            raise ValueError("EXECUTION_AUTHORIZATION_ACTIVATE_REFERENCE_REQUIRED")
        authorization = self._authorization
        if authorization is None or authorization.target is not PromotionMode.LIVE:
            raise RuntimeError("EXECUTION_AUTHORIZATION_REQUIRED")
        if not PromotionGate.validate_authorization(authorization):
            raise RuntimeError("EXECUTION_AUTHORIZATION_INVALID")
        connection = self._transaction()
        try:
            self.store._activate_authorization_in_connection(
                connection, authorization.authorization_id
            )
            self.ledger._append_in_connection(
                connection,
                "EXECUTION_AUTHORIZATION_ACTIVATED",
                self._event_id("EXECUTION_AUTHORIZATION_ACTIVATED", authorization.authorization_id),
                {
                    "authorization_id": authorization.authorization_id,
                    "operator_reference": operator_reference,
                },
            )
            connection.execute("COMMIT")
        except Exception:
            try:
                connection.execute("ROLLBACK")
            finally:
                connection.close()
            raise
        else:
            connection.close()

    def revoke_execution_authorization(self, operator_reference: str) -> None:
        if not operator_reference:
            raise ValueError("EXECUTION_AUTHORIZATION_REVOKE_REFERENCE_REQUIRED")
        authorization = self._authorization
        if authorization is None:
            return
        connection = self._transaction()
        try:
            active = self.store._revoke_authorization_in_connection(
                connection, authorization.authorization_id
            )
            if not active:
                connection.execute("COMMIT")
                connection.close()
                return
            self.ledger._append_in_connection(
                connection,
                "EXECUTION_AUTHORIZATION_REVOKED",
                self._event_id("EXECUTION_AUTHORIZATION_REVOKED", authorization.authorization_id),
                {
                    "authorization_id": authorization.authorization_id,
                    "operator_reference": operator_reference,
                },
            )
            connection.execute("COMMIT")
        except Exception:
            try:
                connection.execute("ROLLBACK")
            finally:
                connection.close()
            raise
        else:
            connection.close()

    def _transaction(self) -> sqlite3.Connection:
        connection = self.store._connect()
        connection.execute("BEGIN IMMEDIATE")
        return connection

    @staticmethod
    def _event_id(event_type: str, idempotency_key: str) -> str:
        return f"{event_type.lower()}-{idempotency_key}"

    @staticmethod
    def _compliance_event_id(
        request: BrokerOrderRequest, decision
    ) -> str:
        payload = {
            "idempotency_key": request.idempotency_key,
            "symbol": request.symbol,
            "side": request.side.value,
            "quantity": request.quantity,
            "price": request.price,
            "action": decision.action.value,
            "reason": decision.reason,
            "policy_id": decision.policy_id,
            "policy_version": decision.policy_version,
            "policy_fingerprint": decision.policy_fingerprint,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        return f"compliance-decision-{request.idempotency_key}-{digest}"

    @staticmethod
    def _compliance_payload(request: BrokerOrderRequest, decision) -> dict:
        return {
            "idempotency_key": request.idempotency_key,
            "symbol": request.symbol,
            "side": request.side.value,
            "quantity": request.quantity,
            "price": request.price,
            "action": decision.action.value,
            "reason": decision.reason,
            "policy_id": decision.policy_id,
            "policy_version": decision.policy_version,
            "policy_fingerprint": decision.policy_fingerprint,
        }

    def _append_compliance_in_connection(
        self, connection: sqlite3.Connection, request: BrokerOrderRequest, decision
    ) -> None:
        payload = self._compliance_payload(request, decision)
        event_id = self._compliance_event_id(request, decision)
        try:
            self.ledger._append_in_connection(
                connection,
                "COMPLIANCE_DECISION",
                event_id,
                payload,
            )
        except ValueError as exc:
            if str(exc) != "LEDGER_EVENT_ID_EXISTS":
                raise
            row = connection.execute(
                "SELECT event_type, payload FROM events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if row is None or row[0] != "COMPLIANCE_DECISION":
                raise ValueError("COMPLIANCE_AUDIT_EVENT_CONFLICT") from exc
            if json.loads(row[1]) != payload:
                raise ValueError("COMPLIANCE_AUDIT_EVENT_CONFLICT") from exc

    def _commit_compliance_and_intent(
        self,
        request: BrokerOrderRequest,
        decision,
        authorization: ExecutionAuthorization | None = None,
        risk_decision: RiskDecision | None = None,
    ) -> None:
        connection = self._transaction()
        try:
            self._append_compliance_in_connection(connection, request, decision)
            if decision.action is ComplianceAction.ALLOW:
                authorization_binding = None
                if getattr(self.broker, "mode", None) is BrokerMode.LIVE:
                    if authorization is None:
                        raise RuntimeError("EXECUTION_AUTHORIZATION_REQUIRED")
                    authorization_binding = (
                        authorization.authorization_id,
                        PromotionGate.authorization_fingerprint(authorization),
                        authorization.issued_at,
                        authorization.expires_at,
                    )
                _, created = self.store._record_in_connection(
                    connection, request, authorization_binding, risk_decision
                )
                if created:
                    self.ledger._append_in_connection(
                        connection,
                        "EXECUTION_INTENT_CREATED",
                        self._event_id(
                            "EXECUTION_INTENT_CREATED", request.idempotency_key
                        ),
                        {
                            "idempotency_key": request.idempotency_key,
                            "symbol": request.symbol,
                            "side": request.side.value,
                            "quantity": request.quantity,
                            "price": request.price,
                            "authorization_id": (
                                authorization.authorization_id
                                if authorization is not None
                                else None
                            ),
                            "authorization_fingerprint": (
                                PromotionGate.authorization_fingerprint(authorization)
                                if authorization is not None
                                else None
                            ),
                            "authorization_issued_at": (
                                authorization.issued_at
                                if authorization is not None
                                else None
                            ),
                            "authorization_expires_at": (
                                authorization.expires_at
                                if authorization is not None
                                else None
                            ),
                        },
                    )
            connection.execute("COMMIT")
        except Exception:
            try:
                connection.execute("ROLLBACK")
            finally:
                connection.close()
            raise
        else:
            connection.close()

    @staticmethod
    def _result_event_id(
        event_type: str, idempotency_key: str, result: BrokerOrderResult
    ) -> str:
        payload = {
            "idempotency_key": idempotency_key,
            "broker_order_id": result.broker_order_id,
            "status": result.status.value,
            "accepted": result.accepted,
            "message": result.message,
            "filled_quantity": result.filled_quantity,
            "remaining_quantity": result.remaining_quantity,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        return f"{event_type.lower()}-{idempotency_key}-{digest}"

    def _commit_result(
        self,
        idempotency_key: str,
        result: BrokerOrderResult,
        event_type: str,
    ) -> None:
        connection = self._transaction()
        try:
            self.store._update_result_in_connection(connection, idempotency_key, result)
            self.ledger._append_in_connection(
                connection,
                event_type,
                self._result_event_id(event_type, idempotency_key, result),
                {
                    "idempotency_key": idempotency_key,
                    "broker_order_id": result.broker_order_id,
                    "status": result.status.value,
                    "accepted": result.accepted,
                    "message": result.message,
                    "filled_quantity": result.filled_quantity,
                    "remaining_quantity": result.remaining_quantity,
                },
            )
            connection.execute("COMMIT")
        except ValueError as exc:
            if str(exc) == "LEDGER_EVENT_ID_EXISTS":
                expected = {
                    "idempotency_key": idempotency_key,
                    "broker_order_id": result.broker_order_id,
                    "status": result.status.value,
                    "accepted": result.accepted,
                    "message": result.message,
                    "filled_quantity": result.filled_quantity,
                    "remaining_quantity": result.remaining_quantity,
                }
                existing_events = [
                    event for event in self.ledger.read()
                    if event.event_id
                    == self._result_event_id(event_type, idempotency_key, result)
                ]
                connection.execute("ROLLBACK")
                connection.close()
                if existing_events and existing_events[0].payload == expected:
                    return
                raise ValueError("EXECUTION_AUDIT_EVENT_CONFLICT") from exc
            try:
                connection.execute("ROLLBACK")
            finally:
                connection.close()
            raise
        except Exception:
            try:
                connection.execute("ROLLBACK")
            finally:
                connection.close()
            raise
        else:
            connection.close()

    @staticmethod
    def _attempt_from_intent(
        idempotency_key: str,
        status: BrokerOrderStatus,
        result: BrokerOrderResult,
    ) -> ExecutionAttempt:
        intent_status = (
            IntentStatus.TERMINAL
            if status
            in {
                BrokerOrderStatus.FILLED,
                BrokerOrderStatus.CANCELED,
                BrokerOrderStatus.REJECTED,
            }
            else (
                IntentStatus.UNKNOWN
                if status is BrokerOrderStatus.UNKNOWN
                else IntentStatus.SUBMITTED
            )
        )
        return ExecutionAttempt(idempotency_key, intent_status, result)

    def enforce_reconciliation_safety(self) -> bool:
        """Inspect execution state and durably halt on any discrepancy."""
        consistency = inspect_execution_consistency(self.store, self.ledger)
        if consistency.healthy:
            return False
        reason = reconciliation_halt_reason(consistency)
        connection = self._transaction()
        try:
            connection.execute(
                """
                INSERT INTO execution_halt(id, active, reason)
                VALUES (1, 1, ?)
                ON CONFLICT(id) DO UPDATE SET active = 1, reason = excluded.reason
                """,
                (reason,),
            )
            try:
                self.ledger._append_in_connection(
                    connection,
                    "EXECUTION_HALT_ASSERTED",
                    self._event_id("EXECUTION_HALT_ASSERTED", reason),
                    {"reason": reason, "errors": list(consistency.errors)},
                )
            except ValueError as exc:
                if str(exc) != "LEDGER_EVENT_ID_EXISTS":
                    raise
            connection.execute("COMMIT")
        except Exception:
            try:
                connection.execute("ROLLBACK")
            finally:
                connection.close()
            raise
        else:
            connection.close()
        return True

    def clear_halt(self, operator_reference: str) -> None:
        """Explicitly clear a durable halt only after clean reconciliation."""
        if not operator_reference:
            raise ValueError("EXECUTION_HALT_CLEAR_REFERENCE_REQUIRED")
        consistency = inspect_execution_consistency(self.store, self.ledger)
        if not consistency.healthy:
            raise RuntimeError(
                "EXECUTION_HALT_CLEAR_BLOCKED:" + "|".join(consistency.errors)
            )
        current_reason = self.store.halt_reason()
        if current_reason is None:
            return
        event_id = self._event_id(
            "EXECUTION_HALT_CLEARED",
            f"{current_reason}:{operator_reference}",
        )
        connection = self._transaction()
        try:
            connection.execute("UPDATE execution_halt SET active = 0 WHERE id = 1")
            try:
                self.ledger._append_in_connection(
                    connection,
                    "EXECUTION_HALT_CLEARED",
                    event_id,
                    {
                        "previous_reason": current_reason,
                        "operator_reference": operator_reference,
                    },
                )
            except ValueError as exc:
                if str(exc) != "LEDGER_EVENT_ID_EXISTS":
                    raise
            connection.execute("COMMIT")
        except Exception:
            try:
                connection.execute("ROLLBACK")
            finally:
                connection.close()
            raise
        else:
            connection.close()

    def _validate_live_risk(self, request: BrokerOrderRequest, risk_decision: RiskDecision | None) -> None:
        if risk_decision is None:
            raise RuntimeError("LIVE_RISK_DECISION_REQUIRED")
        if request.decision_id == request.idempotency_key:
            raise RuntimeError("LIVE_DECISION_ID_REQUIRED")
        DeterministicRiskEngine.validate_evidence(risk_decision)
        if not risk_decision.risk_state_fingerprint:
            raise RuntimeError("RISK_STATE_EVIDENCE_REQUIRED")
        if risk_decision.risk_limits_fingerprint != self._risk.limits.fingerprint():
            raise RuntimeError("RISK_LIMITS_FINGERPRINT_CONFLICT")
        if request.decision_id != risk_decision.decision_id:
            raise RuntimeError("RISK_DECISION_ID_CONFLICT")
        if request.symbol != risk_decision.symbol:
            raise RuntimeError("RISK_SYMBOL_CONFLICT")
        if request.side is not risk_decision.side:
            raise RuntimeError("RISK_SIDE_CONFLICT")
        if request.quantity != risk_decision.quantity:
            raise RuntimeError("RISK_QUANTITY_CONFLICT")
        effective_price = request.price if request.price is not None else request.reference_price
        if effective_price is None:
            raise RuntimeError("LIVE_MARKET_ORDER_REFERENCE_PRICE_REQUIRED")
        if effective_price != risk_decision.price:
            raise RuntimeError("RISK_PRICE_CONFLICT")
        if risk_decision.action is DecisionAction.HOLD:
            raise RuntimeError("RISK_HOLD_NOT_EXECUTABLE")
        if self._risk_state_provider is None:
            raise RuntimeError("LIVE_RISK_STATE_PROVIDER_REQUIRED")
        current = self._risk_state_provider(request)
        if current.fingerprint() != risk_decision.risk_state_fingerprint:
            raise RuntimeError("RISK_STATE_CHANGED_AFTER_APPROVAL")
        if current.symbol != risk_decision.symbol:
            raise RuntimeError("RISK_STATE_SYMBOL_CONFLICT")
        if current.current_position_notional != risk_decision.current_position_notional:
            raise RuntimeError("RISK_STATE_POSITION_CONFLICT")
        if current.equity != risk_decision.equity:
            raise RuntimeError("RISK_STATE_EQUITY_CONFLICT")
        if current.session_start_equity != risk_decision.session_start_equity:
            raise RuntimeError("RISK_STATE_SESSION_EQUITY_CONFLICT")
        if current.high_water_mark != risk_decision.high_water_mark:
            raise RuntimeError("RISK_STATE_HIGH_WATER_MARK_CONFLICT")
        if current.available_cash != risk_decision.available_cash:
            raise RuntimeError("RISK_STATE_CASH_CONFLICT")

    def prepare(self, request: BrokerOrderRequest, risk_decision: RiskDecision | None = None) -> None:
        live = getattr(self.broker, "mode", BrokerMode.DISABLED) is BrokerMode.LIVE
        authorization = self._require_execution_authorization()
        if live:
            self._validate_live_risk(request, risk_decision)
        decision = self.compliance.evaluate(
            request, getattr(self.broker, "mode", BrokerMode.DISABLED)
        )
        self._commit_compliance_and_intent(request, decision, authorization, risk_decision)
        if decision.action is not ComplianceAction.ALLOW:
            raise RuntimeError(
                f"{decision.reason}:{decision.policy_id}:{decision.policy_version}"
            )

    def submit(self, request: BrokerOrderRequest, risk_decision: RiskDecision | None = None) -> ExecutionAttempt:
        self._require_execution_authorization()
        self.enforce_reconciliation_safety()
        if self.store.is_halted():
            reason = self.store.halt_reason() or "EXECUTION_HALTED"
            raise RuntimeError(f"EXECUTION_HALTED:{reason}")
        self.prepare(request, risk_decision)
        existing = self.store.get(request.idempotency_key)
        if existing is None:
            raise RuntimeError("EXECUTION_INTENT_NOT_PERSISTED")

        if existing.result is not None:
            return self._attempt_from_intent(
                request.idempotency_key,
                existing.status,
                existing.result,
            )

        # Re-check authoritative risk state immediately before the external side effect.
        if getattr(self.broker, "mode", None) is BrokerMode.LIVE:
            self._validate_live_risk(request, risk_decision)

        # Re-check LIVE authorization after all deterministic preparation and
        # immediately before the external side effect.
        authorization = self._require_execution_authorization()
        if getattr(self.broker, "mode", None) is BrokerMode.LIVE:
            if authorization is None:
                raise RuntimeError("EXECUTION_AUTHORIZATION_REQUIRED")
            if existing.risk_fingerprint != risk_decision.risk_fingerprint:
                raise RuntimeError("EXECUTION_INTENT_RISK_CONFLICT")
            expected_binding = (
                authorization.authorization_id,
                PromotionGate.authorization_fingerprint(authorization),
                authorization.issued_at,
                authorization.expires_at,
            )
            actual_binding = (
                existing.authorization_id,
                existing.authorization_fingerprint,
                existing.authorization_issued_at,
                existing.authorization_expires_at,
            )
            if actual_binding != expected_binding:
                raise RuntimeError("EXECUTION_INTENT_AUTHORIZATION_CONFLICT")

        try:
            result = self.broker.submit(request)
        except (ConnectionError, OSError, RuntimeError, TimeoutError):
            result = BrokerOrderResult(
                accepted=False,
                broker_order_id=None,
                status=BrokerOrderStatus.UNKNOWN,
                message="BROKER_SUBMISSION_UNKNOWN",
            )
            self._commit_result(
                request.idempotency_key, result, "EXECUTION_UNKNOWN"
            )
            return ExecutionAttempt(
                request.idempotency_key,
                IntentStatus.UNKNOWN,
                result,
            )

        self._commit_result(request.idempotency_key, result, "EXECUTION_RESULT")
        status = (
            IntentStatus.TERMINAL
            if result.status
            in {
                BrokerOrderStatus.FILLED,
                BrokerOrderStatus.CANCELED,
                BrokerOrderStatus.REJECTED,
            }
            else IntentStatus.SUBMITTED
        )
        return ExecutionAttempt(request.idempotency_key, status, result)

    def recover_unknown(self) -> tuple[ExecutionReconciliation, ...]:
        """Recover provider-accepted orders without ever resubmitting them.

        The provider must return an explicit client-key binding. A provider
        response for an unrequested key is treated as a reconciliation conflict.
        If the provider finds nothing, the durable local intent remains UNKNOWN.
        """

        keys = self.store.unknown_keys()
        if not keys:
            return ()

        recovered = self.broker.reconcile_by_idempotency_keys(list(keys))
        requested = set(keys)
        seen: set[str] = set()
        reconciliations: list[ExecutionReconciliation] = []

        for item in recovered:
            key = item.idempotency_key
            if key not in requested:
                raise ValueError("EXECUTION_RECOVERY_KEY_CONFLICT")
            if key in seen:
                raise ValueError("EXECUTION_RECOVERY_DUPLICATE_KEY")
            seen.add(key)

            intent = self.store.get(key)
            if intent is None or intent.status is not BrokerOrderStatus.UNKNOWN:
                raise ValueError("EXECUTION_RECOVERY_STATE_CONFLICT")

            self._commit_result(key, item.result, "EXECUTION_RECOVERED")
            reconciliations.append(ExecutionReconciliation(key, item.result))

        return tuple(reconciliations)

    def reconcile_discovered_orders(self) -> tuple[ExecutionReconciliation, ...]:
        """Reconcile provider orders discovered without relying on local state.

        A provider order with no matching durable client intent is a hard
        execution-boundary discrepancy: it is audited and the durable halt is
        asserted before any further autonomous submission can proceed.
        """
        discovered = self.broker.discover_open_orders()
        local_keys = set(self.store.all_keys())

        orphans = [
            item for item in discovered
            if item.idempotency_key is None
            or item.idempotency_key not in local_keys
        ]
        if orphans:
            errors = tuple(
                "EXECUTION_PROVIDER_ORDER_ORPHAN:"
                + (item.result.broker_order_id or "UNKNOWN")
                for item in orphans
            )
            self.enforce_reconciliation_safety_with_errors(
                ExecutionConsistency(False, errors)
            )
            raise RuntimeError("EXECUTION_PROVIDER_ORPHAN_DETECTED")

        reconciliations = []
        seen_keys: set[str] = set()
        for item in discovered:
            key = item.idempotency_key
            if key is None:
                continue
            if key in seen_keys:
                consistency = ExecutionConsistency(
                    False, (f"EXECUTION_PROVIDER_DUPLICATE_KEY:{key}",)
                )
                self.enforce_reconciliation_safety_with_errors(consistency)
                raise RuntimeError("EXECUTION_PROVIDER_DUPLICATE_KEY")
            seen_keys.add(key)
            intent = self.store.get(key)
            if intent is None:
                continue
            if intent.result is None:
                self._commit_result(key, item.result, "EXECUTION_RECONCILED")
                reconciliations.append(ExecutionReconciliation(key, item.result))
            elif intent.result != item.result:
                try:
                    self._commit_result(key, item.result, "EXECUTION_RECONCILED")
                except ValueError as exc:
                    if str(exc) not in {
                        "EXECUTION_TERMINAL_STATE_CONFLICT",
                        "EXECUTION_STATE_TRANSITION_CONFLICT",
                    }:
                        raise
                    consistency = ExecutionConsistency(
                        False,
                        (f"EXECUTION_PROVIDER_STATE_CONFLICT:{key}",),
                    )
                    self.enforce_reconciliation_safety_with_errors(consistency)
                    raise RuntimeError("EXECUTION_PROVIDER_STATE_CONFLICT") from exc
                reconciliations.append(ExecutionReconciliation(key, item.result))
        return tuple(reconciliations)

    def enforce_reconciliation_safety_with_errors(
        self, consistency: ExecutionConsistency
    ) -> bool:
        """Durably halt using an externally constructed reconciliation result."""
        from .execution_reconciliation import reconciliation_halt_reason
        reason = reconciliation_halt_reason(consistency)
        connection = self._transaction()
        try:
            connection.execute(
                """
                INSERT INTO execution_halt(id, active, reason)
                VALUES (1, 1, ?)
                ON CONFLICT(id) DO UPDATE SET active = 1, reason = excluded.reason
                """,
                (reason,),
            )
            self.ledger._append_in_connection(
                connection,
                "EXECUTION_HALT_ASSERTED",
                self._event_id("EXECUTION_HALT_ASSERTED", reason),
                {"reason": reason, "errors": list(consistency.errors)},
            )
            connection.execute("COMMIT")
        except ValueError as exc:
            if str(exc) != "LEDGER_EVENT_ID_EXISTS":
                try:
                    connection.execute("ROLLBACK")
                finally:
                    connection.close()
                raise
            connection.execute("COMMIT")
        except Exception:
            try:
                connection.execute("ROLLBACK")
            finally:
                connection.close()
            raise
        else:
            connection.close()
        return True

    def reconcile(
        self,
        broker_order_ids: list[str],
    ) -> tuple[ExecutionReconciliation, ...]:
        requested = set(broker_order_ids)
        if len(requested) != len(broker_order_ids):
            consistency = ExecutionConsistency(
                False, ("EXECUTION_RECONCILE_DUPLICATE_REQUESTED_ORDER_ID",)
            )
            self.enforce_reconciliation_safety_with_errors(consistency)
            raise RuntimeError("EXECUTION_RECONCILE_DUPLICATE_REQUESTED_ORDER_ID")

        results = self.broker.reconcile(broker_order_ids)
        reconciliations: list[ExecutionReconciliation] = []
        seen: set[str] = set()

        for result in results:
            broker_order_id = result.broker_order_id
            if broker_order_id is None:
                continue
            if broker_order_id not in requested:
                consistency = ExecutionConsistency(
                    False,
                    (f"EXECUTION_RECONCILE_UNREQUESTED_ORDER:{broker_order_id}",),
                )
                self.enforce_reconciliation_safety_with_errors(consistency)
                raise RuntimeError("EXECUTION_RECONCILE_UNREQUESTED_ORDER")
            if broker_order_id in seen:
                consistency = ExecutionConsistency(
                    False,
                    (f"EXECUTION_RECONCILE_DUPLICATE_RESULT:{broker_order_id}",),
                )
                self.enforce_reconciliation_safety_with_errors(consistency)
                raise RuntimeError("EXECUTION_RECONCILE_DUPLICATE_RESULT")
            seen.add(broker_order_id)

            matched_key = None
            for key in self.store.pending_keys():
                intent = self.store.get(key)
                if intent is None or intent.result is None:
                    continue
                if intent.result.broker_order_id == broker_order_id:
                    matched_key = key
                    break

            if matched_key is None:
                consistency = ExecutionConsistency(
                    False,
                    (f"EXECUTION_RECONCILE_UNBOUND_ORDER:{broker_order_id}",),
                )
                self.enforce_reconciliation_safety_with_errors(consistency)
                raise RuntimeError("EXECUTION_RECONCILE_UNBOUND_ORDER")

            try:
                self._commit_result(
                    matched_key, result, "EXECUTION_RECONCILED"
                )
            except ValueError as exc:
                if str(exc) not in {
                    "EXECUTION_TERMINAL_STATE_CONFLICT",
                    "EXECUTION_STATE_TRANSITION_CONFLICT",
                }:
                    raise
                consistency = ExecutionConsistency(
                    False,
                    (f"EXECUTION_RECONCILE_STATE_CONFLICT:{matched_key}",),
                )
                self.enforce_reconciliation_safety_with_errors(consistency)
                raise RuntimeError("EXECUTION_RECONCILE_STATE_CONFLICT") from exc

            reconciliations.append(
                ExecutionReconciliation(matched_key, result)
            )

        return tuple(reconciliations)
