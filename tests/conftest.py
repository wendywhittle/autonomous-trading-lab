"""Shared test fixtures for the autonomous-trading-lab suite."""

import os

import pytest

#: Fixed HMAC key used to sign LIVE execution authorizations in tests (H2).
#: Tests that exercise the fail-closed path remove it explicitly with
#: ``monkeypatch.delenv("ATLAB_AUTH_SIGNING_KEY")``.
TEST_AUTH_SIGNING_KEY = "atlab-test-signing-key-do-not-use-in-production"


@pytest.fixture(autouse=True)
def _auth_signing_key_env(monkeypatch):
    monkeypatch.setenv("ATLAB_AUTH_SIGNING_KEY", TEST_AUTH_SIGNING_KEY)
    # Preserve any pre-existing value is intentionally not done here: tests
    # must be deterministic regardless of the ambient environment.
    assert os.environ["ATLAB_AUTH_SIGNING_KEY"] == TEST_AUTH_SIGNING_KEY
