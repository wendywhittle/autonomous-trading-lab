import json

import pytest

from atlab.risk_state import RiskSessionState


def _state() -> RiskSessionState:
    return RiskSessionState(
        session_start_equity=10000.0,
        high_water_mark=10000.0,
        current_equity=9900.0,
    )


def test_risk_session_state_round_trip(tmp_path):
    path = tmp_path / "risk_session.json"
    state = _state()
    state.save(path)

    loaded = RiskSessionState.load(path)

    assert loaded == state
    assert loaded.fingerprint() == state.fingerprint()


def test_risk_session_state_tamper_detected(tmp_path):
    # M7: a manual edit (e.g. raising session_start_equity to hide a
    # drawdown breach) fails closed on load instead of passing silently.
    path = tmp_path / "risk_session.json"
    _state().save(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    data["session_start_equity"] = 20000.0
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="RISK_SESSION_STATE_CORRUPT"):
        RiskSessionState.load(path)


def test_risk_session_state_missing_fingerprint_rejected(tmp_path):
    # Files written before the fingerprint existed cannot be verified —
    # fail closed rather than trust them.
    path = tmp_path / "risk_session.json"
    _state().save(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    del data["fingerprint"]
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="RISK_SESSION_STATE_CORRUPT"):
        RiskSessionState.load(path)


def test_risk_session_state_fingerprint_stable_across_int_float(tmp_path):
    int_state = RiskSessionState(
        session_start_equity=10000,
        high_water_mark=10000,
        current_equity=9900,
    )
    assert int_state.fingerprint() == _state().fingerprint()
