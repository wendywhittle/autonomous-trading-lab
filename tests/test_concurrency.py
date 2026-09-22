from concurrent.futures import ThreadPoolExecutor, as_completed

from atlab.ledger import ImmutableLedger


def test_ledger_concurrent_appends_are_serialized(tmp_path):
    ledger_path = tmp_path / "ledger.sqlite3"
    ledger = ImmutableLedger(ledger_path)
    event_ids = [f"concurrent-{index}" for index in range(64)]

    def append_event(event_id: str):
        return ledger.append("TEST", event_id, {"event_id": event_id})

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(append_event, event_id) for event_id in event_ids]
        events = [future.result() for future in as_completed(futures)]

    assert {event.event_id for event in events} == set(event_ids)
    assert sorted(event.sequence for event in events) == list(range(len(event_ids)))

    persisted = ledger.read()
    assert [event.sequence for event in persisted] == list(range(len(event_ids)))
    assert {event.event_id for event in persisted} == set(event_ids)
    assert [event.payload["event_id"] for event in persisted] == [
        event.event_id for event in persisted
    ]


def test_ledger_concurrent_duplicate_event_id_allows_only_one_append(tmp_path):
    ledger = ImmutableLedger(tmp_path / "ledger.sqlite3")

    def append_duplicate():
        try:
            ledger.append("TEST", "same-event", {"value": 1})
            return "success"
        except ValueError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=16) as executor:
        outcomes = list(executor.map(lambda _: append_duplicate(), range(16)))

    assert outcomes.count("success") == 1
    assert outcomes.count("LEDGER_EVENT_ID_EXISTS") == 15
    persisted = ledger.read()
    assert len(persisted) == 1
    assert persisted[0].event_id == "same-event"
    assert persisted[0].sequence == 0
