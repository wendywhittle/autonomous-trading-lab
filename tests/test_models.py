from datetime import UTC, datetime

from atlab.models import MarketObservation


def _obs(**kwargs) -> MarketObservation:
    base = {
        "symbol": "TEST",
        "timestamp": datetime(2024, 1, 1, tzinfo=UTC),
        "price": 100.0,
        "source": "test",
    }
    base.update(kwargs)
    return MarketObservation(**base)


def test_volume_default_is_float_zero():
    # M12: pydantic does not validate defaults, so the int default 0
    # fingerprinted differently from an explicit 0.0 for identical
    # observations.
    observation = _obs()

    assert observation.volume == 0.0
    assert isinstance(observation.volume, float)


def test_volume_default_serializes_identically_to_explicit_zero():
    assert _obs().model_dump(mode="json") == _obs(volume=0.0).model_dump(mode="json")


def test_naive_timestamp_normalized_to_utc():
    # M13: the same instant must fingerprint the same regardless of
    # whether the source supplied a timezone offset.
    naive = _obs(timestamp=datetime(2024, 1, 1, 12, 30, 0))  # noqa: DTZ001 - naive input is the point of this test
    aware = _obs(timestamp=datetime(2024, 1, 1, 12, 30, 0, tzinfo=UTC))

    assert naive.timestamp == aware.timestamp
    assert naive.timestamp.tzinfo is not None
    assert (
        naive.model_dump(mode="json")["timestamp"]
        == aware.model_dump(mode="json")["timestamp"]
    )


def test_aware_timestamp_with_offset_kept_verbatim():
    observation = _obs(timestamp=datetime(2024, 1, 1, 12, 30, 0, tzinfo=UTC))

    assert observation.timestamp.utcoffset().total_seconds() == 0
