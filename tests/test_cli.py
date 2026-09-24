import json
from pathlib import Path

import pytest

from atlab.cli import main

HEADER = "timestamp,symbol,open,high,low,close,volume"
ROWS = [
    "2024-01-02T00:00:00+00:00,AAA,100,101,99,100,1000",
    "2024-01-03T00:00:00+00:00,AAA,100,102,99,101,1000",
    "2024-01-04T00:00:00+00:00,AAA,101,103,100,102,1000",
    "2024-01-05T00:00:00+00:00,AAA,102,102,100,101,1000",
]


def make_csv(tmp_path: Path) -> str:
    path = tmp_path / "bars.csv"
    path.write_text("\n".join([HEADER, *ROWS]) + "\n", encoding="utf-8")
    return str(path)


def test_cli_backtest_reports_metrics(tmp_path, capsys):
    csv_path = make_csv(tmp_path)
    code = main(
        ["backtest", "--csv", csv_path, "--symbol", "AAA", "--workdir", str(tmp_path / "wd")]
    )
    assert code == 0
    out = capsys.readouterr().out
    for token in ("observations=4", "trades=", "total_return=", "max_drawdown=", "sharpe_ratio=", "win_rate="):
        assert token in out


def test_cli_run_persists_state(tmp_path, capsys):
    csv_path = make_csv(tmp_path)
    workdir = tmp_path / "run-state"
    code = main(["run", "--csv", csv_path, "--symbol", "AAA", "--workdir", str(workdir)])
    assert code == 0
    out = capsys.readouterr().out
    assert "cycles=3" in out
    assert (workdir / "ledger.sqlite3").exists()
    assert (workdir / "portfolio.json").exists()
    assert (workdir / "risk_session.json").exists()


def test_cli_backtest_unknown_symbol_fails(tmp_path, capsys):
    csv_path = make_csv(tmp_path)
    code = main(
        ["backtest", "--csv", csv_path, "--symbol", "ZZZ", "--workdir", str(tmp_path / "wd")]
    )
    assert code == 2
    assert "CSV_NO_OBSERVATIONS_FOR_SYMBOL" in capsys.readouterr().err


def test_cli_backtest_rejects_bad_csv(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("timestamp,symbol,open\n2024-01-02T00:00:00+00:00,AAA,100\n", encoding="utf-8")
    code = main(
        ["backtest", "--csv", str(bad), "--symbol", "AAA", "--workdir", str(tmp_path / "wd")]
    )
    assert code == 2
    assert "CSV_MISSING_COLUMNS" in capsys.readouterr().err


# --- H9: the `promote` command evaluates real, locally-verified evidence. ---


@pytest.mark.promote_cli
def test_cli_promote_paper_reports_eligible(tmp_path, capsys):
    csv_path = make_csv(tmp_path)
    workdir = tmp_path / "promote-wd"
    assert (
        main(["run", "--csv", csv_path, "--symbol", "AAA", "--workdir", str(workdir)])
        == 0
    )
    code = main(
        [
            "promote",
            "--target",
            "paper",
            "--csv",
            csv_path,
            "--symbol",
            "AAA",
            "--workdir",
            str(workdir),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "target=PAPER eligible=True" in out
    assert "PROMOTION_ELIGIBLE:paper" in out
    record = json.loads((workdir / "promotion_record.json").read_text(encoding="utf-8"))
    assert record["eligible"] is True
    assert record["target"] == "PAPER"
    assert record["evidence"]["tests_green"]["provenance"] == "verified"
    assert record["evidence"]["kill_switch_verified"]["value"] is True


@pytest.mark.promote_cli
def test_cli_promote_live_refuses_without_live_controls(tmp_path, capsys):
    csv_path = make_csv(tmp_path)
    workdir = tmp_path / "promote-live-wd"
    code = main(
        [
            "promote",
            "--target",
            "live",
            "--csv",
            csv_path,
            "--symbol",
            "AAA",
            "--workdir",
            str(workdir),
            "--attest-human-approval",
            "Test Operator <test>: end-to-end check",
        ]
    )
    assert code == 1
    out = capsys.readouterr().out
    assert "target=LIVE eligible=False" in out
    assert "LIVE_EXECUTION_NOT_IMPLEMENTED" in out
    record = json.loads((workdir / "promotion_record.json").read_text(encoding="utf-8"))
    assert record["eligible"] is False
    # Attested-but-unverified evidence stays explicit in the record.
    assert record["evidence"]["human_approval"]["provenance"] == "attested"
    assert record["evidence"]["human_approval"]["value"] is True
    assert record["evidence"]["live_execution_implemented"]["provenance"] == "verified"
