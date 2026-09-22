from collections.abc import Iterable

from .models import LedgerEvent


def replay(events: Iterable[LedgerEvent]) -> list[str]:
    ordered = list(events)
    actual = [event.sequence for event in ordered]
    if actual != list(range(len(ordered))):
        raise ValueError("LEDGER_SEQUENCE_INVALID")
    if len({event.event_id for event in ordered}) != len(ordered):
        raise ValueError("LEDGER_EVENT_ID_DUPLICATE")
    return [event.event_id for event in ordered]
