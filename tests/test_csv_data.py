from pathlib import Path

import pytest

from atlab.adapters import CsvMarketData


def write_csv(path: Path, rows: list[str]) -> Path:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


HEADER = "timestamp,symbol,open,high,low,close,volume"


def test_csv_loads_and_orders_chronologically(tmp_path):
    path = write_csv(
        tmp_path / "data.csv",
        [
            HEADER,
            "2024-01-03T00:00:00+00:00,AAA,100,101,99,100.5,1000",
            "2024-01-02T00:00:00+00:00,AAA,99,100,98,99.5,900",
        ],
    )
    observations = list(CsvMarketData(path).observations("AAA"))
    assert [item.price for item in observations] == [99.5, 100.5]
    assert observations[0].timestamp.isoformat() == "2024-01-02T00:00:00+00:00"
    assert observations[0].metadata == {"open": 99.0, "high": 100.0, "low": 98.0, "close": 99.5}
    assert observations[0].volume == 900.0


def test_csv_filters_by_symbol(tmp_path):
    path = write_csv(
        tmp_path / "data.csv",
        [
            HEADER,
            "2024-01-02T00:00:00+00:00,AAA,100,101,99,100,1000",
            "2024-01-02T00:00:00+00:00,BBB,200,201,199,200,1000",
        ],
    )
    data = CsvMarketData(path)
    assert [item.symbol for item in data.observations("AAA")] == ["AAA"]
    assert [item.symbol for item in data.observations("BBB")] == ["BBB"]
    assert list(data.observations("CCC")) == []


def test_csv_duplicate_timestamps_keep_file_order(tmp_path):
    path = write_csv(
        tmp_path / "data.csv",
        [
            HEADER,
            "2024-01-02T00:00:00+00:00,AAA,100,101,99,100,1000",
            "2024-01-02T00:00:00+00:00,AAA,100,101,99,101,1000",
        ],
    )
    observations = list(CsvMarketData(path).observations("AAA"))
    assert [item.price for item in observations] == [100.0, 101.0]


def test_csv_naive_timestamp_assumed_utc(tmp_path):
    path = write_csv(
        tmp_path / "data.csv",
        [HEADER, "2024-01-02T00:00:00,AAA,100,101,99,100,1000"],
    )
    (observation,) = CsvMarketData(path).observations("AAA")
    assert observation.timestamp.utcoffset().total_seconds() == 0


def test_csv_missing_column_rejected(tmp_path):
    path = write_csv(tmp_path / "data.csv", ["timestamp,symbol,open,high,low", "2024-01-02T00:00:00+00:00,AAA,100,101,99"])
    with pytest.raises(ValueError, match="CSV_MISSING_COLUMNS"):
        CsvMarketData(path)


def test_csv_bad_timestamp_rejected(tmp_path):
    path = write_csv(tmp_path / "data.csv", [HEADER, "not-a-date,AAA,100,101,99,100,1000"])
    with pytest.raises(ValueError, match="CSV_BAD_TIMESTAMP"):
        CsvMarketData(path)


def test_csv_bad_numeric_rejected(tmp_path):
    path = write_csv(tmp_path / "data.csv", [HEADER, "2024-01-02T00:00:00+00:00,AAA,100,101,99,abc,1000"])
    with pytest.raises(ValueError, match="CSV_BAD_NUMERIC"):
        CsvMarketData(path)


def test_csv_non_positive_close_rejected(tmp_path):
    path = write_csv(tmp_path / "data.csv", [HEADER, "2024-01-02T00:00:00+00:00,AAA,100,101,99,0,1000"])
    with pytest.raises(ValueError, match="CSV_NON_POSITIVE_PRICE"):
        CsvMarketData(path)


def test_csv_high_low_inconsistency_rejected(tmp_path):
    path = write_csv(tmp_path / "data.csv", [HEADER, "2024-01-02T00:00:00+00:00,AAA,100,99,101,100,1000"])
    with pytest.raises(ValueError, match="CSV_HIGH_LOW_INCONSISTENT"):
        CsvMarketData(path)


def test_csv_volume_defaults_to_zero(tmp_path):
    path = write_csv(
        tmp_path / "data.csv",
        ["timestamp,symbol,open,high,low,close", "2024-01-02T00:00:00+00:00,AAA,100,101,99,100"],
    )
    (observation,) = CsvMarketData(path).observations("AAA")
    assert observation.volume == 0.0


def test_csv_rejects_infinite_price(tmp_path):
    # M8: float("inf") parses and pydantic's gt=0 accepts it — the adapter
    # must reject non-finite numerics before they poison returns/equity.
    path = write_csv(
        tmp_path / "data.csv",
        ["timestamp,symbol,open,high,low,close", "2024-01-02T00:00:00+00:00,AAA,100,101,99,inf"],
    )
    with pytest.raises(ValueError, match="CSV_NON_FINITE_NUMERIC:close"):
        CsvMarketData(path)


def test_csv_rejects_nan_value(tmp_path):
    path = write_csv(
        tmp_path / "data.csv",
        ["timestamp,symbol,open,high,low,close", "2024-01-02T00:00:00+00:00,AAA,nan,101,99,100"],
    )
    with pytest.raises(ValueError, match="CSV_NON_FINITE_NUMERIC:open"):
        CsvMarketData(path)


def test_csv_rejects_infinite_volume(tmp_path):
    path = write_csv(
        tmp_path / "data.csv",
        [HEADER, "2024-01-02T00:00:00+00:00,AAA,100,101,99,100,inf"],
    )
    with pytest.raises(ValueError, match="CSV_NON_FINITE_NUMERIC:volume"):
        CsvMarketData(path)
