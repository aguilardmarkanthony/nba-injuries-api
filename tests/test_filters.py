import pytest

import app as api
from injury_status import parse_status, status_code


@pytest.mark.parametrize(
    "value,expected",
    [
        ("OUT", "OUT"),
        (" out ", "OUT"),
        ("Out For Season", "OFS"),
        ("out for the season", "OFS"),
        ("Game-Time Decision", "GTD"),
        ("Day-To-Day", "DAY_TO_DAY"),
        ("DTD", "DAY_TO_DAY"),
        ("Questionable", "QUESTIONABLE"),
        ("Doubtful", "DOUBTFUL"),
        ("Probable", "PROBABLE"),
        ("UNKNOWN", "UNKNOWN"),
    ],
)
def test_supported_status_spellings(value, expected):
    assert parse_status(value) == status_code(value) == expected


@pytest.mark.parametrize("value", ["", " ", "injured", "brand new source status"])
def test_unknown_source_status_is_preserved_but_invalid_filters_are_rejected(value):
    assert status_code(value) == "UNKNOWN"
    with pytest.raises(ValueError):
        parse_status(value)


def mixed_snapshot(snapshot):
    row = snapshot["teams"]["GS"][0]
    snapshot["teams"] = {
        "GS": [{**row, "status": "Out", "playerId": "101"}, {**row, "status": "GTD", "playerId": "102"}],
        "NO": [{**row, "status": "Questionable", "playerId": "103"}],
        "NY": [{**row, "status": "Some new source value", "playerId": None}],
    }
    snapshot["teamCount"] = 3
    snapshot["playerEntryCount"] = 4


def test_status_filter_combines_with_alias_without_mutating_cached_data(client, snapshot):
    mixed_snapshot(snapshot)
    data = client.get("/v1/injuries", params={"team": "GSW", "status": " out "}).json()
    assert data["canonicalTeam"] == "GSW" and data["count"] == 1
    assert data["injuries"][0]["status"] == "Out"
    assert data["injuries"][0]["statusCode"] == "OUT"
    assert len(snapshot["teams"]["GS"]) == 2
    assert client.get("/v1/injuries?team=GSW").json()["count"] == 2


def test_all_team_filter_updates_both_counts_and_preserves_source_keys(client, snapshot):
    mixed_snapshot(snapshot)
    result = client.get("/v1/injuries?status=OUT").json()
    assert set(result["teams"]) == {"GS"}
    assert result["teamCount"] == result["playerEntryCount"] == 1
    assert snapshot["playerEntryCount"] == 4


def test_gtd_and_questionable_remain_distinct(client, snapshot):
    mixed_snapshot(snapshot)
    result = client.get("/v1/injuries?status=questionable").json()
    assert set(result["teams"]) == {"NO"}
    assert result["teams"]["NO"][0]["statusCode"] == "QUESTIONABLE"


def test_unknown_filter_keeps_original_unrecognized_status(client, snapshot):
    mixed_snapshot(snapshot)
    result = client.get("/v1/injuries?status=UNKNOWN").json()
    assert result["teams"]["NY"][0]["status"] == "Some new source value"
    assert result["teams"]["NY"][0]["statusCode"] == "UNKNOWN"


def test_multiple_teams_deduplicate_aliases_and_combine_with_status(client, snapshot):
    mixed_snapshot(snapshot)
    result = client.get("/v1/injuries", params={"teams": " gsw, GS, nop "}).json()
    assert set(result["teams"]) == {"GS", "NO"}
    assert result["teamCount"] == 2 and result["playerEntryCount"] == 3
    filtered = client.get("/v1/injuries?teams=GSW,NOP&status=OUT").json()
    assert filtered["teamCount"] == filtered["playerEntryCount"] == 1


@pytest.mark.parametrize(
    "query,code",
    [
        ("team=GS&teams=NO", "INVALID_FILTER"),
        ("status=", "INVALID_STATUS"),
        ("status=not-real", "INVALID_STATUS"),
        ("teams=", "INVALID_TEAM"),
        ("teams=GS,", "INVALID_TEAM"),
        ("teams=GS,INVALID", "INVALID_TEAM"),
        ("teams=" + ",".join(["GS"] * 31), "INVALID_TEAM"),
    ],
)
def test_filter_errors_fail_before_reading_upstream(client, monkeypatch, query, code):
    def should_not_fetch():
        raise AssertionError("Invalid input fetched source data")

    monkeypatch.setattr(api, "get_snapshot", should_not_fetch)
    response = client.get("/v1/injuries?" + query)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == code
    if code == "INVALID_STATUS":
        assert "OUT" in response.json()["error"]["validStatuses"]


def test_valid_status_with_no_matches_is_empty_success(client):
    single = client.get("/v1/injuries?team=GSW&status=PROBABLE")
    assert single.status_code == 200 and single.json()["injuries"] == []
    multiple = client.get("/v1/injuries?teams=GSW,NOP&status=PROBABLE").json()
    assert multiple["teams"] == {} and multiple["teamCount"] == multiple["playerEntryCount"] == 0


def test_player_id_lookup_matches_exact_identity_with_team_context(client, snapshot):
    mixed_snapshot(snapshot)
    result = client.get("/v1/player/101").json()
    assert result["playerId"] == "101" and result["count"] == 1
    assert result["results"][0]["team"] == "GS"
    assert result["results"][0]["canonicalTeam"] == "GSW"
    assert result["results"][0]["statusCode"] == "OUT"
    assert client.get("/v1/player/10").json()["results"] == []
    assert client.get("/v1/player/999999999999").status_code == 200


@pytest.mark.parametrize("player_id", ["abc", "-1", "0", "001", "1.2", "1234567890123", "１２３"])
def test_invalid_player_ids_have_validation_errors(client, player_id):
    response = client.get("/v1/player/" + player_id)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_player_lookup_requires_authentication_and_does_not_shadow_search(client):
    assert client.get("/v1/player/search?name=Example").json()["count"] == 30
    client.headers.clear()
    assert client.get("/v1/player/123").status_code == 401
