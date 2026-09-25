"""Chronological data splitting for time-series research.

The cardinal rule, ported from the Sniper v1 spec: never randomly shuffle
time-series events. Every split and every walk-forward window below is
strictly chronological — the test period always starts after the validation
period, which always starts after the training period.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime

from ..models import MarketObservation


def assert_chronological(observations: Sequence[MarketObservation]) -> None:
    """Raise unless observations are in non-decreasing timestamp order."""
    for previous, current in itertools.pairwise(observations):
        if current.timestamp < previous.timestamp:
            raise ValueError("OBSERVATIONS_NOT_CHRONOLOGICAL")


@dataclass(frozen=True)
class ChronoSplit:
    """One chronological train/validate/test split, as half-open index ranges."""

    n: int
    train_end: int  #: exclusive end of the training slice ``[0, train_end)``
    validate_end: int  #: exclusive end of the validation slice

    @property
    def train(self) -> tuple[int, int]:
        return (0, self.train_end)

    @property
    def validate(self) -> tuple[int, int]:
        return (self.train_end, self.validate_end)

    @property
    def test(self) -> tuple[int, int]:
        return (self.validate_end, self.n)


def chronological_split(
    n: int, *, train_frac: float = 0.6, validate_frac: float = 0.2
) -> ChronoSplit:
    """Split ``n`` ordered observations into train/validate/test fractions.

    The test fraction is whatever remains after train and validate. Every
    section is guaranteed at least one observation.
    """
    if n < 1:
        raise ValueError("SPLIT_EMPTY_DATA")
    if not 0.0 < train_frac < 1.0:
        raise ValueError("SPLIT_INVALID_TRAIN_FRAC")
    if not 0.0 <= validate_frac < 1.0:
        raise ValueError("SPLIT_INVALID_VALIDATE_FRAC")
    if train_frac + validate_frac >= 1.0:
        raise ValueError("SPLIT_NO_TEST_FRACTION")
    train_end = int(n * train_frac)
    validate_end = train_end + int(n * validate_frac)
    # Guarantee non-empty sections after truncation.
    train_end = max(1, min(train_end, n - 2))
    validate_end = max(train_end + (1 if validate_frac > 0 else 0), min(validate_end, n - 1))
    if validate_end >= n:
        raise ValueError("SPLIT_SECTION_EMPTY")
    return ChronoSplit(n=n, train_end=train_end, validate_end=validate_end)


@dataclass(frozen=True)
class WalkForwardWindow:
    """One walk-forward step: train -> validate -> test, all half-open ranges."""

    train_start: int
    train_end: int
    validate_start: int
    validate_end: int
    test_start: int
    test_end: int

    @property
    def train(self) -> tuple[int, int]:
        return (self.train_start, self.train_end)

    @property
    def validate(self) -> tuple[int, int]:
        return (self.validate_start, self.validate_end)

    @property
    def test(self) -> tuple[int, int]:
        return (self.test_start, self.test_end)


def walk_forward_windows(
    n: int,
    *,
    train_size: int,
    validate_size: int,
    test_size: int,
    step: int,
    expanding: bool = True,
    embargo: int = 0,
) -> Iterator[WalkForwardWindow]:
    """Yield successive walk-forward windows over ``n`` ordered observations.

    Window ``i`` (0-based):

    - train: ``[anchor, anchor + train_size)`` where ``anchor`` is 0 for an
      expanding window or ``i * step`` for a rolling window,
    - validate: ``[train_end + embargo, train_end + embargo + validate_size)``,
    - test: ``[validate_end, validate_end + test_size)``.

    The ``embargo`` is a dead zone of bars between the end of training and the
    start of validation, so labels computed near the boundary cannot leak
    training information into evaluation. Iteration stops at the first window
    whose test slice would run past the end of the data.
    """
    for name, value in (
        ("train_size", train_size),
        ("validate_size", validate_size),
        ("test_size", test_size),
        ("step", step),
    ):
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"WALK_FORWARD_INVALID_{name.upper()}")
    if not isinstance(embargo, int) or embargo < 0:
        raise ValueError("WALK_FORWARD_INVALID_EMBARGO")
    if n < 1:
        raise ValueError("WALK_FORWARD_EMPTY_DATA")

    index = 0
    while True:
        if expanding:
            # Anchored start, growing training set: the frame still advances
            # by ``step`` so the test slice moves forward every window.
            train_start, train_end = 0, train_size + index * step
        else:
            train_start, train_end = index * step, index * step + train_size
        validate_start = train_end + embargo
        validate_end = validate_start + validate_size
        test_start = validate_end
        test_end = test_start + test_size
        if test_end > n:
            return
        yield WalkForwardWindow(
            train_start=train_start,
            train_end=train_end,
            validate_start=validate_start,
            validate_end=validate_end,
            test_start=test_start,
            test_end=test_end,
        )
        index += 1


def window_bounds(
    observations: Sequence[MarketObservation], start: int, end: int
) -> tuple[datetime | None, datetime | None]:
    """Return the (first, last) timestamps of an observation slice."""
    if start >= end:
        return (None, None)
    ordered = observations[start:end]
    return (ordered[0].timestamp, ordered[-1].timestamp)
