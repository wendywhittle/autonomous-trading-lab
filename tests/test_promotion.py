from atlab.promotion import PromotionEvidence, PromotionGate, PromotionMode


def evidence(**overrides):
    values = {
        "tests_green": True,
        "paper_run_complete": True,
        "risk_limits_active": True,
        "kill_switch_verified": True,
        "replay_deterministic": True,
        "no_live_credentials": True,
        "live_credentials_configured": False,
        "live_execution_implemented": False,
        "live_controls_verified": False,
    }
    values.update(overrides)
    return PromotionEvidence(**values)


def test_promotion_gate_allows_paper_with_required_evidence():
    result = PromotionGate().evaluate(evidence(), PromotionMode.PAPER)
    assert result.eligible
    assert result.failed_requirements == ()


def test_promotion_gate_blocks_missing_evidence():
    result = PromotionGate().evaluate(
        evidence(replay_deterministic=False), PromotionMode.PAPER
    )
    assert not result.eligible
    assert result.failed_requirements == ("REPLAY_NOT_DETERMINISTIC",)


def test_live_promotion_requires_live_credentials_execution_controls_and_approval():
    result = PromotionGate().evaluate(
        evidence(
            human_approval=True,
            live_execution_implemented=True,
            live_controls_verified=True,
        ),
        PromotionMode.LIVE,
    )
    assert not result.eligible
    assert result.failed_requirements == ("LIVE_CREDENTIALS_NOT_CONFIGURED",)


def test_live_promotion_can_authorize_only_when_all_live_requirements_are_present():
    approved = evidence(
        no_live_credentials=False,
        live_credentials_configured=True,
        live_execution_implemented=True,
        live_controls_verified=True,
        human_approval=True,
    )
    result = PromotionGate().evaluate(approved, PromotionMode.LIVE)
    assert result.eligible

    authorization = PromotionGate().authorize(approved, PromotionMode.LIVE)
    assert authorization.target is PromotionMode.LIVE
    assert authorization.authorization_id
    assert authorization.evidence_fingerprint



def test_live_authorization_rejects_tampering():
    approved = evidence(
        no_live_credentials=False,
        live_credentials_configured=True,
        live_execution_implemented=True,
        live_controls_verified=True,
        human_approval=True,
    )
    authorization = PromotionGate().authorize(approved, PromotionMode.LIVE)
    assert PromotionGate.validate_authorization(authorization)

    tampered = type(authorization)(
        authorization_id=authorization.authorization_id,
        target=authorization.target,
        evidence_fingerprint="tampered",
    )
    assert not PromotionGate.validate_authorization(tampered)
