from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RiskStateSnapshot:
    symbol: str
    current_position_notional: float
    equity: float
    session_start_equity: float
    high_water_mark: float
    available_cash: float

    def __post_init__(self) -> None:
        import math
        values = (self.current_position_notional, self.equity, self.session_start_equity, self.high_water_mark, self.available_cash)
        if not self.symbol or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("INVALID_RISK_STATE_SNAPSHOT")
        if self.high_water_mark < self.equity:
            raise ValueError("INVALID_RISK_STATE_SNAPSHOT")

    def fingerprint(self) -> str:
        payload = {
            "symbol": self.symbol,
            "current_position_notional": self.current_position_notional,
            "equity": self.equity,
            "session_start_equity": self.session_start_equity,
            "high_water_mark": self.high_water_mark,
            "available_cash": self.available_cash,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class RiskSessionState:
    """Persisted session equity state used by deterministic risk controls."""

    session_start_equity: float
    high_water_mark: float
    current_equity: float

    def __post_init__(self) -> None:
        if (
            self.session_start_equity < 0
            or self.high_water_mark < 0
            or self.current_equity < 0
            or self.high_water_mark < self.current_equity
        ):
            raise ValueError("INVALID_RISK_SESSION_STATE")

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "current_equity": self.current_equity,
                    "high_water_mark": self.high_water_mark,
                    "session_start_equity": self.session_start_equity,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        temporary.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> RiskSessionState:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        required = {"session_start_equity", "high_water_mark", "current_equity"}
        if set(data) != required:
            raise ValueError("INVALID_RISK_SESSION_STATE")
        return cls(
            session_start_equity=float(data["session_start_equity"]),
            high_water_mark=float(data["high_water_mark"]),
            current_equity=float(data["current_equity"]),
        )
