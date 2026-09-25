"""Tests for chronological splits and walk-forward windows."""

from datetime import UTC, datetime

import pytest

from atlab.models import MarketObservation
from atlab.research.splits import (
    assert_chronological,
    chronological_split,
    walk_forward_windows,
    window_bounds,
)


def obs(day):
    return MarketObservation(
        symbol="T",
        timestamp=datetime(2024, 1, day, tzinfo=UTC),
        price=100.0 + day,
        source="test",
    )


def test_assert_chronological_accepts_ordered():
    assert_chronological([obs(1), obs(2), obs(3)])


def test_assert_chronological_rejects_disorder():
    with pytest.raises(ValueError, match="OBSERVATIONS_NOT_CHRONOLOGICAL"):
        assert_chronological([obs(1), obs(3), obs(2)])


def test_chronological_split_fractions():
    split = chronological_split(100, train_frac=0.6, validate_frac=0.2)
    assert split.train == (0, 60)
    assert split.validate == (60, 80)
    assert split.test == (80, 100)


def test_chronological_split_guarantees_non_empty_sections():
    split = chronological_split(5, train_frac=0.6, validate_frac=0.2)
    assert split.train[1] - split.train[0] >= 1
    assert split.validate[1] - split.validate[0] >= 1
    assert split.test[1] - split.test[0] >= 1


def test_chronological_split_rejects_bad_fractions():
    with pytest.raises(ValueError, match="SPLIT_EMPTY_DATA"):
        chronological_split(0)
    with pytest.raises(ValueError, match="SPLIT_INVALID_TRAIN_FRAC"):
        chronological_split(10, train_frac=0.0)
    with pytest.raises(ValueError, match="SPLIT_NO_TEST_FRACTION"):
        chronological_split(10, train_frac=0.6, validate_frac=0.4)


def test_walk_forward_expanding_windows():
    windows = list(
        walk_forward_windows(10, train_size=3, validate_size=2, test_size=2, step=2, expanding=True)
    )
    assert [(w.train, w.validate, w.test) for w in windows] == [
        ((0, 3), (3, 5), (5, 7)),
        ((0, 5), (5, 7), (7, 9)),  # anchored start, growing training set
    ]


def test_walk_forward_rolling_windows():
    windows = list(
        walk_forward_windows(
            10, train_size=3, validate_size=2, test_size=2, step=2, expanding=False
        )
    )
    assert [(w.train, w.validate, w.test) for w in windows] == [
        ((0, 3), (3, 5), (5, 7)),
        ((2, 5), (5, 7), (7, 9)),
    ]


def test_walk_forward_embargo_shifts_validation():
    windows = list(
        walk_forward_windows(12, train_size=3, validate_size=2, test_size=2, step=2, embargo=1)
    )
    first = windows[0]
    assert first.train == (0, 3)
    assert first.validate == (4, 6)  # one dead-zone bar after training
    assert first.test == (6, 8)


def test_walk_forward_stops_when_test_exceeds_data():
    windows = list(walk_forward_windows(6, train_size=3, validate_size=2, test_size=2, step=1))
    assert windows == []


def test_walk_forward_rejects_bad_sizes():
    with pytest.raises(ValueError, match="WALK_FORWARD_INVALID_TRAIN_SIZE"):
        list(walk_forward_windows(10, train_size=0, validate_size=1, test_size=1, step=1))
    with pytest.raises(ValueError, match="WALK_FORWARD_INVALID_EMBARGO"):
        list(
            walk_forward_windows(10, train_size=3, validate_size=1, test_size=1, step=1, embargo=-1)
        )
    with pytest.raises(ValueError, match="WALK_FORWARD_EMPTY_DATA"):
        list(walk_forward_windows(0, train_size=3, validate_size=1, test_size=1, step=1))


def test_window_bounds_returns_slice_timestamps():
    observations = [obs(1), obs(2), obs(3)]
    start, end = window_bounds(observations, 0, 2)
    assert start == observations[0].timestamp
    assert end == observations[1].timestamp
    assert window_bounds(observations, 2, 2) == (None, None)
