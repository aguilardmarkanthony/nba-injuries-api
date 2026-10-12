import pytest
from fastapi.testclient import TestClient

import app as api
from injury_cache import InjuryCache
from teams import ALIASES, CANONICAL_TO_SOURCE


@pytest.mark.parametrize("alias,canonical", ALIASES.items())
def test_every_team_alias_returns_matching_injuries(client, alias, canonical):
    response = client.get("/v1/injuries", params={"team": f" {alias.lower()} "})
    assert response.status_code == 200
    data = response.json()
    assert data["team"] == alias
    assert data["canonicalTeam"] == canonical
    assert data["count"] == 1
    assert data["injuries"][0]["player"] == f"Example {canonical} Player"
    assert data["source"] == "ESPN"


def test_all_teams_preserve_v1_source_keys_and_counts(client):
    data = client.get("/v1/injuries").json()
    assert set(data["teams"]) == set(CANONICAL_TO_SOURCE.values())
    assert "GS" in data["teams"] and "GSW" not in data["teams"]
    assert data["playerEntryCount"] == sum(map(len, data["teams"].values()))
    assert data["teamCount"] == len(data["teams"])
    assert data["cache"]["stale"] is False


@pytest.mark.parametrize("team", ["INVALID", "", "   ", "123", "🔥"])
def test_invalid_team_is_not_a_successful_empty_result(client, team):
    response = client.get("/v1/injuries", params={"team": team})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_TEAM"
    assert len(response.json()["error"]["validTeams"]) == 30


def test_valid_team_without_injuries_is_successful(client, snapshot):
    snapshot["teams"].pop("GS")
    data = client.get("/v1/injuries?team=GSW").json()
    assert data["count"] == 0 and data["injuries"] == []


def test_team_directory_is_complete_even_when_source_fails(client, monkeypatch):
    def unavailable():
        raise RuntimeError("offline")

    monkeypatch.setattr(api, "cache", InjuryCache(unavailable))
    response = client.get("/v1/teams")
    assert response.status_code == 200
    data = response.json()
    assert len(data["teams"]) == len(data["canonicalTeams"]) == len(data["details"]) == 30
    assert next(row for row in data["details"] if row["code"] == "GSW")["aliases"] == ["GS", "GSW"]


def test_player_search_trims_and_matches_case_insensitively(client):
    response = client.get("/v1/player/search", params={"name": "  example gsw  "})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["results"][0]["team"] == "GS"
    assert data["results"][0]["canonicalTeam"] == "GSW"


@pytest.mark.parametrize("name", ["  ", " a ", "\t\t"])
def test_player_search_rejects_trimmed_short_names(client, name):
    response = client.get("/v1/player/search", params={"name": name})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_SEARCH"


def test_missing_search_and_long_team_have_documented_validation_error(client):
    for path in ["/v1/player/search", "/v1/injuries?team=" + "A" * 17]:
        response = client.get(path)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"
        assert "input" not in str(response.json())


@pytest.mark.parametrize(
    "headers,query",
    [
        ({}, ""),
        ({"X-RapidAPI-Key": "anything"}, ""),
        ({"X-RapidAPI-Proxy-Secret": "wrong"}, ""),
        ({"X-API-Key": "free-demo-key"}, ""),
        ({"X-API-Key": "pro-demo-key"}, ""),
        ({}, "?api_key=test-direct-key"),
    ],
)
def test_origin_rejects_missing_spoofed_demo_and_query_keys(client, headers, query):
    with TestClient(api.app) as unauthenticated:
        response = unauthenticated.get("/v1/injuries" + query, headers=headers)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHORIZED"


def test_explicit_direct_key_works(client):
    with TestClient(api.app) as direct:
        assert direct.get("/v1/teams", headers={"X-API-Key": "test-direct-key"}).status_code == 200


def test_empty_proxy_configuration_does_not_authenticate_empty_header(client, monkeypatch):
    monkeypatch.setattr(api, "RAPIDAPI_PROXY_SECRET", "")
    assert client.get("/v1/teams").status_code == 401


def test_unavailable_feed_returns_503_and_retry_header(client, monkeypatch):
    def unavailable():
        raise ValueError("internal details must not leak")

    monkeypatch.setattr(api, "cache", InjuryCache(unavailable))
    response = client.get("/v1/injuries")
    assert response.status_code == 503
    assert response.headers["retry-after"] == "30"
    assert response.json()["error"]["code"] == "UPSTREAM_UNAVAILABLE"
    assert "internal details" not in response.text
    assert client.get("/v1/health").status_code == 200
    assert client.get("/v1/ready").status_code == 503


def test_health_does_not_expose_auth_configuration(client):
    data = client.get("/v1/health").json()
    assert data == {"status": "ok", "version": "2.4.0", "cacheTtlSeconds": 120}


def test_openapi_documents_filters_and_response_models(client):
    spec = client.get("/openapi.json").json()
    operation = spec["paths"]["/v1/injuries"]["get"]
    assert operation["parameters"][0]["name"] == "team"
    assert operation["parameters"][0]["required"] is False
    assert {"200", "400", "401", "422", "503"} <= set(operation["responses"])
