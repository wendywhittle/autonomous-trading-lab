from .models import LedgerEvent


def replay(events: list[LedgerEvent]) -> list[str]:
    ordered = sorted(events, key=lambda event: event.sequence)
    actual = [event.sequence for event in ordered]
    if actual != list(range(len(ordered))):
        raise ValueError("LEDGER_SEQUENCE_INVALID")
    return [event.event_id for event in ordered]
