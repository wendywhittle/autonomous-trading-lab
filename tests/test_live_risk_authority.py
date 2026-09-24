from concurrent.futures import ThreadPoolExecutor

import pytest

from atlab.execution import ExecutionIntentStore
from atlab.risk import RiskLimits
from atlab.risk_state import RiskStateSnapshot


def snapshot(cash=1000):
    return RiskStateSnapshot("TEST",0,1000,1000,1000,cash)


def test_live_risk_authority_initializes_and_versions(tmp_path):
    store=ExecutionIntentStore(tmp_path/"risk.sqlite3")
    assert store.initialize_live_risk_state(snapshot())==0
    assert store.get_live_risk_state()[1]==0
    assert store.mutate_live_risk_state(snapshot(cash=900),0)==1
    assert store.get_live_risk_state()[1]==1
    with pytest.raises(ValueError,match="RISK_STATE_VERSION_CONFLICT"):
        store.mutate_live_risk_state(snapshot(cash=800),0)


def test_live_limits_are_authoritative_and_versioned(tmp_path):
    store=ExecutionIntentStore(tmp_path/"risk.sqlite3")
    limits=RiskLimits()
    assert store.initialize_live_risk_limits(limits)==0
    assert store.get_live_risk_limits()[1]==0
    changed=RiskLimits(max_order_notional=99)
    assert store.mutate_live_risk_limits(changed,0)==1
    assert store.get_live_risk_limits()[0]==changed
    with pytest.raises(ValueError,match="RISK_LIMITS_VERSION_CONFLICT"):
        store.mutate_live_risk_limits(limits,0)


def test_concurrent_state_mutations_have_one_winner(tmp_path):
    path=tmp_path/"risk.sqlite3"; store=ExecutionIntentStore(path); store.initialize_live_risk_state(snapshot())
    def mutate(value):
        try:
            return store.mutate_live_risk_state(snapshot(value),0)
        except Exception as exc:
            return str(exc)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results=list(pool.map(mutate,[900+i for i in range(8)]))
    assert results.count(1)==1
    assert sum(isinstance(x,str) and x=="RISK_STATE_VERSION_CONFLICT" for x in results)==7
    assert store.get_live_risk_state()[1]==1


def test_restart_preserves_live_authority(tmp_path):
    path=tmp_path/"risk.sqlite3"; first=ExecutionIntentStore(path); first.initialize_live_risk_state(snapshot()); first.initialize_live_risk_limits(RiskLimits())
    first.mutate_live_risk_state(snapshot(900),0)
    first.mutate_live_risk_limits(RiskLimits(max_order_notional=99),0)
    restarted=ExecutionIntentStore(path)
    assert restarted.get_live_risk_state()[1]==1
    assert restarted.get_live_risk_limits()[1]==1


def test_uninitialized_live_authority_fails_closed(tmp_path):
    store=ExecutionIntentStore(tmp_path/"risk.sqlite3")
    with pytest.raises(RuntimeError,match="LIVE_RISK_STATE_NOT_INITIALIZED"):
        store.get_live_risk_state()
    with pytest.raises(RuntimeError,match="LIVE_RISK_LIMITS_NOT_INITIALIZED"):
        store.get_live_risk_limits()
