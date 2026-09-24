import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum


class PromotionMode(str, Enum):
    RESEARCH = "RESEARCH"
    PAPER = "PAPER"
    LIVE = "LIVE"


#: Environment variable holding the HMAC key that signs LIVE execution
#: authorizations. LIVE authorization paths fail closed when it is absent:
#: issuance is refused and validation returns False. The key is never logged
#: or committed; it only ever travels from the environment into HMAC.
AUTH_SIGNING_KEY_ENV = "ATLAB_AUTH_SIGNING_KEY"


@dataclass(frozen=True)
class PromotionEvidence:
    tests_green: bool
    paper_run_complete: bool
    risk_limits_active: bool
    kill_switch_verified: bool
    replay_deterministic: bool
    no_live_credentials: bool
    live_execution_implemented: bool = False
    live_controls_verified: bool = False
    human_approval: bool = False
    live_credentials_configured: bool = False


@dataclass(frozen=True)
class PromotionResult:
    eligible: bool
    target: PromotionMode
    failed_requirements: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionAuthorization:
    """Immutable authorization artifact required before LIVE broker submission.

    ``signature`` is HMAC-SHA256 over ``authorization_id`` keyed by
    ``ATLAB_AUTH_SIGNING_KEY``. An empty signature means unsigned and never
    validates for LIVE use.
    """

    authorization_id: str
    target: PromotionMode
    evidence_fingerprint: str
    issued_at: str | None = None
    expires_at: str | None = None
    signature: str = ""


class PromotionGate:
    """Explicit promotion gate; LIVE requires a verified execution path and approval."""

    BASE_REQUIREMENTS = (
        ("tests_green", "TESTS_NOT_GREEN"),
        ("paper_run_complete", "PAPER_RUN_INCOMPLETE"),
        ("risk_limits_active", "RISK_LIMITS_INACTIVE"),
        ("kill_switch_verified", "KILL_SWITCH_NOT_VERIFIED"),
        ("replay_deterministic", "REPLAY_NOT_DETERMINISTIC"),
    )

    PAPER_REQUIREMENTS = (
        ("no_live_credentials", "LIVE_CREDENTIALS_PRESENT"),
    )

    LIVE_REQUIREMENTS = (
        ("live_credentials_configured", "LIVE_CREDENTIALS_NOT_CONFIGURED"),
        ("live_execution_implemented", "LIVE_EXECUTION_NOT_IMPLEMENTED"),
        ("live_controls_verified", "LIVE_CONTROLS_NOT_VERIFIED"),
        ("human_approval", "HUMAN_APPROVAL_REQUIRED"),
    )

    def evaluate(
        self, evidence: PromotionEvidence, target: PromotionMode
    ) -> PromotionResult:
        failed = tuple(
            reason
            for field, reason in self.BASE_REQUIREMENTS
            if not getattr(evidence, field)
        )

        if target is PromotionMode.PAPER:
            failed += tuple(
                reason
                for field, reason in self.PAPER_REQUIREMENTS
                if not getattr(evidence, field)
            )
        elif target is PromotionMode.LIVE:
            failed += tuple(
                reason
                for field, reason in self.LIVE_REQUIREMENTS
                if not getattr(evidence, field)
            )
        elif target is PromotionMode.RESEARCH:
            pass
        else:
            raise ValueError(f"PROMOTION_MODE_UNSUPPORTED:{target}")

        return PromotionResult(not failed, target, failed)

    @staticmethod
    def _evidence_fingerprint(evidence: PromotionEvidence) -> str:
        payload = {
            field: getattr(evidence, field)
            for field in evidence.__dataclass_fields__
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @staticmethod
    def authorization_fingerprint(authorization: ExecutionAuthorization) -> str:
        if not authorization.issued_at or not authorization.expires_at:
            raise ValueError("EXECUTION_AUTHORIZATION_TIMESTAMP_REQUIRED")
        return hashlib.sha256(
            f"{authorization.target.value}:{authorization.evidence_fingerprint}:{authorization.issued_at}:{authorization.expires_at}".encode()
        ).hexdigest()

    @staticmethod
    def validate_authorization(authorization: ExecutionAuthorization) -> bool:
        if authorization.target is not PromotionMode.LIVE:
            return False
        if not authorization.authorization_id or not authorization.evidence_fingerprint:
            return False
        if not authorization.issued_at or not authorization.expires_at:
            return False
        try:
            issued = datetime.fromisoformat(authorization.issued_at)
            expires = datetime.fromisoformat(authorization.expires_at)
            now = datetime.now(UTC)
            if issued.tzinfo is None or expires.tzinfo is None:
                return False
            if expires <= issued or now >= expires or issued > now + timedelta(seconds=5):
                return False
        except ValueError:
            return False
        expected = PromotionGate.authorization_fingerprint(authorization)
        if authorization.authorization_id != expected:
            return False
        # H2: the authorization must carry a valid HMAC signature. Without a
        # configured signing key, or with a missing/forged signature,
        # validation fails closed.
        if not authorization.signature:
            return False
        key = PromotionGate.signing_key()
        if key is None:
            return False
        expected_signature = PromotionGate.sign_authorization_id(
            authorization.authorization_id, key
        )
        return hmac.compare_digest(expected_signature, authorization.signature)

    @staticmethod
    def signing_key() -> bytes | None:
        """Return the HMAC signing key from the environment, if configured."""
        key = os.environ.get(AUTH_SIGNING_KEY_ENV)
        if not key:
            return None
        return key.encode("utf-8")

    @staticmethod
    def sign_authorization_id(authorization_id: str, key: bytes) -> str:
        return hmac.new(
            key, authorization_id.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def authorize(
        self,
        evidence: PromotionEvidence,
        target: PromotionMode,
        ttl_seconds: int = 300,
        now: datetime | None = None,
    ) -> ExecutionAuthorization:
        result = self.evaluate(evidence, target)
        if target is not PromotionMode.LIVE:
            raise RuntimeError("EXECUTION_AUTHORIZATION_LIVE_ONLY")
        if not result.eligible:
            raise RuntimeError("PROMOTION_NOT_ELIGIBLE:" + "|".join(result.failed_requirements))
        if ttl_seconds <= 0:
            raise ValueError("EXECUTION_AUTHORIZATION_TTL_INVALID")
        issued = now or datetime.now(UTC)
        if issued.tzinfo is None:
            raise ValueError("EXECUTION_AUTHORIZATION_TIMEZONE_REQUIRED")
        issued = issued.astimezone(UTC)
        expires = issued + timedelta(seconds=ttl_seconds)
        issued_at = issued.isoformat()
        expires_at = expires.isoformat()
        fingerprint = self._evidence_fingerprint(evidence)
        authorization_id = hashlib.sha256(
            f"{target.value}:{fingerprint}:{issued_at}:{expires_at}".encode()
        ).hexdigest()
        # H2: LIVE authorizations are HMAC-signed. Without a configured
        # signing key, issuance fails closed.
        key = self.signing_key()
        if key is None:
            raise RuntimeError("EXECUTION_AUTHORIZATION_SIGNING_KEY_MISSING")
        signature = self.sign_authorization_id(authorization_id, key)
        return ExecutionAuthorization(
            authorization_id=authorization_id,
            target=target,
            evidence_fingerprint=fingerprint,
            issued_at=issued_at,
            expires_at=expires_at,
            signature=signature,
        )
