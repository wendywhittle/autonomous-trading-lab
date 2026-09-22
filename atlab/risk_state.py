from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


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
    def load(cls, path: str | Path) -> "RiskSessionState":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        required = {"session_start_equity", "high_water_mark", "current_equity"}
        if set(data) != required:
            raise ValueError("INVALID_RISK_SESSION_STATE")
        return cls(
            session_start_equity=float(data["session_start_equity"]),
            high_water_mark=float(data["high_water_mark"]),
            current_equity=float(data["current_equity"]),
        )
