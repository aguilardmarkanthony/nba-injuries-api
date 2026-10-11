import pytest
from fastapi.testclient import TestClient

import app as api
from injury_cache import InjuryCache
from teams import CANONICAL_TO_SOURCE


@pytest.fixture
def snapshot():
    teams = {
        source: [
            {
                "player": f"Example {canonical} Player",
                "playerId": "123",
                "status": "OUT",
                "reason": "Ankle",
                "returnDate": None,
            }
        ]
        for canonical, source in CANONICAL_TO_SOURCE.items()
    }
    return {
        "updatedAt": 1791746877,
        "source": "ESPN",
        "teams": teams,
        "teamCount": 30,
        "playerEntryCount": 30,
    }


@pytest.fixture
def client(monkeypatch, snapshot):
    monkeypatch.setattr(api, "cache", InjuryCache(lambda: snapshot))
    monkeypatch.setattr(api, "RAPIDAPI_PROXY_SECRET", "test-proxy-secret")
    monkeypatch.setattr(api, "DIRECT_API_KEYS", {"test-direct-key"})
    with TestClient(api.app, headers={"X-RapidAPI-Proxy-Secret": "test-proxy-secret"}) as client:
        yield client
