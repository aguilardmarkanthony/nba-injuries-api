import io
import json
from urllib.error import HTTPError, URLError

import pytest

import app as api
from injury_cache import InjuryCache
from scripts import check_freshness as monitor


@pytest.mark.parametrize(
    "age,status,state",
    [
        (0, 200, "fresh"),
        (119, 200, "fresh"),
        (120, 200, "delayed"),
        (299, 200, "delayed"),
        (300, 503, "stale"),
        (901, 503, "stale"),
    ],
)
def test_probe_observes_snapshot_without_refreshing_it(client, snapshot, monkeypatch, age, status, state):
    now, calls = [1000.0], []
    cache = InjuryCache(lambda: calls.append(1) or snapshot, clock=lambda: now[0])
    cache.get()
    monkeypatch.setattr(api, "cache", cache)
    monkeypatch.setattr(api, "FRESHNESS_ALERT_SECONDS", 300)
    now[0] += age
    client.headers.clear()
    response = client.get("/v1/freshness")
    assert response.status_code == status
    assert response.json()["status"] == state
    assert response.json()["cache"]["ageSeconds"] == age
    assert response.headers["cache-control"] == "no-store"
    assert "teams" not in response.json()
    assert len(calls) == 1
    if status == 503:
        assert response.headers["retry-after"] == "30"


def test_missing_snapshot_is_unavailable_without_starting_a_fetch(client, monkeypatch):
    def fail_if_called():
        raise AssertionError("The probe must not fetch")

    monkeypatch.setattr(api, "cache", InjuryCache(fail_if_called))
    response = client.get("/v1/freshness")
    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"
    assert response.json()["cache"] is None and response.json()["updatedAt"] is None
    assert response.json()["consecutiveRefreshFailures"] == 0


def test_repeated_refresh_failures_alert_and_success_clears_failure_count(client, snapshot, monkeypatch):
    now, failing = [100.0], [False]

    def loader():
        if failing[0]:
            raise ValueError("private upstream body")
        return snapshot

    cache = InjuryCache(loader, ttl=10, max_age=900, clock=lambda: now[0])
    cache.get()
    monkeypatch.setattr(api, "cache", cache)
    failing[0] = True
    for moment in [111.0, 142.0, 173.0]:
        now[0] = moment
        cache.refresh()
    response = client.get("/v1/freshness")
    assert response.status_code == 503 and response.json()["status"] == "degraded"
    assert response.json()["consecutiveRefreshFailures"] == 3
    assert "private upstream" not in response.text
    now[0], failing[0] = 204.0, False
    cache.refresh()
    recovered = client.get("/v1/freshness")
    assert recovered.status_code == 200 and recovered.json()["consecutiveRefreshFailures"] == 0


def good_report():
    return {
        "status": "fresh",
        "source": "ESPN",
        "updatedAt": 1791746877,
        "cache": {"ageSeconds": 20},
        "alertAfterSeconds": 300,
        "consecutiveRefreshFailures": 0,
    }


def test_monitor_accepts_a_valid_report():
    assert monitor.validate_report(good_report(), 200).startswith("PASS:")


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "stale"),
        ("status", ["fresh"]),
        ("cache", None),
        ("cache", {"ageSeconds": 300}),
        ("cache", {"ageSeconds": -1}),
        ("cache", {"ageSeconds": True}),
        ("alertAfterSeconds", 0),
        ("consecutiveRefreshFailures", 3),
        ("updatedAt", None),
    ],
)
def test_monitor_rejects_stale_or_malformed_success_responses(field, value):
    report = {**good_report(), field: value}
    with pytest.raises(monitor.MonitorFailure):
        monitor.validate_report(report, 200)


def test_monitor_retries_startup_then_passes(monkeypatch):
    results = [monitor.MonitorFailure("starting"), "PASS"]
    sleeps = []

    def probe(*args):
        result = results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(monitor, "probe", probe)
    monkeypatch.setattr(monitor.time, "sleep", sleeps.append)
    assert monitor.check("https://example.test") == 0
    assert sleeps == [15]


def test_monitor_exits_nonzero_after_bounded_failures(monkeypatch, capsys):
    calls = []

    def probe(*args):
        calls.append(1)
        raise monitor.MonitorFailure("unavailable")

    monkeypatch.setattr(monitor, "probe", probe)
    monkeypatch.setattr(monitor.time, "sleep", lambda _: None)
    assert monitor.check("https://example.test") == 1
    assert len(calls) == 3
    assert "ALERT" in capsys.readouterr().out


@pytest.mark.parametrize(
    "body,code",
    [
        (b"<html>Starting service</html>", 200),
        (b"x" * 65_537, 200),
        (json.dumps(good_report()).encode(), 503),
    ],
    ids=["startup-html", "oversized-body", "http-503"],
)
def test_monitor_rejects_startup_pages_oversized_bodies_and_http_errors(monkeypatch, body, code):
    def open_request(*args, **kwargs):
        if code != 200:
            raise HTTPError("https://example.test", code, "unavailable", {}, io.BytesIO(body))
        response = io.BytesIO(body)
        response.status = code
        return response

    monkeypatch.setattr(monitor, "urlopen", open_request)
    with pytest.raises(monitor.MonitorFailure):
        monitor.probe("https://example.test", 1)


def test_monitor_sanitizes_transport_errors(monkeypatch):
    def fail(*args, **kwargs):
        raise URLError("private upstream response")

    monkeypatch.setattr(monitor, "urlopen", fail)
    with pytest.raises(monitor.MonitorFailure, match="Could not reach") as caught:
        monitor.probe("https://example.test", 1)
    assert "private upstream" not in str(caught.value)
