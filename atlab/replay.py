"""Ledger sequence validation.

``validate_ledger_sequence`` checks that a sequence of ledger events is
contiguous starting at 0 and carries unique event IDs. It does not
reconstruct decisions or portfolio states from the event stream — for a
deterministic end-to-end check, run the engine twice and compare outputs
(see ``evidence.check_replay_deterministic``).
"""

from collections.abc import Iterable

from .models import LedgerEvent


def validate_ledger_sequence(events: Iterable[LedgerEvent]) -> list[str]:
    ordered = list(events)
    actual = [event.sequence for event in ordered]
    if actual != list(range(len(ordered))):
        raise ValueError("LEDGER_SEQUENCE_INVALID")
    if len({event.event_id for event in ordered}) != len(ordered):
        raise ValueError("LEDGER_EVENT_ID_DUPLICATE")
    return [event.event_id for event in ordered]
