"""Frontend SSE client: protocol parsing seam."""

from frontend.api_client import iter_sse_events


def test_iter_sse_events_parses_data_lines():
    lines = [
        'data: {"token": "Odpo"}',
        "",
        ": ping",
        'data: {"token": "wiedź"}',
        'data: {"done": true, "sources": []}',
    ]
    events = list(iter_sse_events(lines))
    assert [e.get("token") for e in events[:2]] == ["Odpo", "wiedź"]
    assert events[-1]["done"] is True


def test_iter_sse_events_handles_unicode():
    events = list(iter_sse_events(['data: {"token": "Стаття 8 — права суб\'єкта"}']))
    assert events[0]["token"] == "Стаття 8 — права суб'єкта"
