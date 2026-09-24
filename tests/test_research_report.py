"""Tests for the confidence-bucket report."""

import pytest

from atlab.research.report import (
    ScoredTrade,
    confidence_bucket_table,
    format_bucket_table,
)


def test_buckets_aggregate_gross_costs_and_net():
    trades = [
        ScoredTrade(confidence=0.71, gross_pnl=2.0, cost=0.5),
        ScoredTrade(confidence=0.73, gross_pnl=-1.0, cost=0.5),
        ScoredTrade(confidence=0.82, gross_pnl=3.0, cost=0.5),
    ]
    rows = confidence_bucket_table(trades, bucket_width=0.1, min_confidence=0.7)
    by_bucket = {row.bucket: row for row in rows}
    assert by_bucket["70-80%"].trades == 2
    assert by_bucket["70-80%"].gross_edge == pytest.approx(1.0)
    assert by_bucket["70-80%"].costs == pytest.approx(1.0)
    assert by_bucket["70-80%"].net_edge == pytest.approx(0.0)
    assert by_bucket["70-80%"].expectancy_per_trade == pytest.approx(0.0)
    assert by_bucket["80-90%"].trades == 1
    assert by_bucket["80-90%"].net_edge == pytest.approx(2.5)


def test_boundary_confidence_falls_in_higher_bucket():
    trades = [ScoredTrade(confidence=0.70, gross_pnl=1.0, cost=0.0)]
    rows = confidence_bucket_table(trades, bucket_width=0.1, min_confidence=0.5)
    by_bucket = {row.bucket: row for row in rows}
    assert by_bucket["70-80%"].trades == 1
    assert by_bucket["60-70%"].trades == 0


def test_confidence_one_lands_in_final_bucket():
    trades = [ScoredTrade(confidence=1.0, gross_pnl=1.0, cost=0.0)]
    rows = confidence_bucket_table(trades, bucket_width=0.1, min_confidence=0.5)
    assert rows[-1].bucket == "90-100%"
    assert rows[-1].trades == 1


def test_empty_buckets_are_visible_not_hidden():
    rows = confidence_bucket_table([], bucket_width=0.25, min_confidence=0.5)
    assert [row.bucket for row in rows] == ["50-75%", "75-100%"]
    assert all(row.trades == 0 for row in rows)
    assert all(row.expectancy_per_trade == 0.0 for row in rows)


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="SCORED_TRADE_CONFIDENCE_OUT_OF_RANGE"):
        ScoredTrade(confidence=1.5, gross_pnl=0.0, cost=0.0)
    with pytest.raises(ValueError, match="SCORED_TRADE_NEGATIVE_COST"):
        ScoredTrade(confidence=0.6, gross_pnl=0.0, cost=-1.0)
    with pytest.raises(ValueError, match="BUCKET_INVALID_WIDTH"):
        confidence_bucket_table([], bucket_width=0.0)
    with pytest.raises(ValueError, match="BUCKET_INVALID_MIN_CONFIDENCE"):
        confidence_bucket_table([], min_confidence=1.0)


def test_format_bucket_table_renders_rows():
    rows = confidence_bucket_table(
        [ScoredTrade(confidence=0.72, gross_pnl=2.0, cost=0.5)],
        bucket_width=0.1,
        min_confidence=0.7,
    )
    text = format_bucket_table(rows)
    assert "bucket" in text
    assert "70-80%" in text
    assert "exp/trade" in text
