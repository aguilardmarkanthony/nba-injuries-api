import pytest

from injury_feed import FeedError, athlete_id, build_injury_map


def row(abbreviation="GS", **extra):
    return {
        "athlete": {"id": "123", "displayName": "Example Player", "team": {"abbreviation": abbreviation}},
        "status": "Out",
        "details": {"fantasyStatus": {"description": "OUT"}, "type": "Ankle"},
        **extra,
    }


@pytest.mark.parametrize("grouped", [True, False])
def test_supported_source_shapes_and_canonical_source_alias(grouped):
    items = [row("GSW")]
    if grouped:
        items = [{"displayName": "Golden State Warriors", "injuries": items}]
    parsed = build_injury_map({"injuries": items})
    assert parsed == {
        "GS": [
            {
                "player": "Example Player",
                "playerId": "123",
                "status": "OUT",
                "reason": "Ankle",
                "returnDate": None,
            }
        ]
    }


def test_group_name_fallback_when_athlete_team_is_missing():
    injury = row()
    injury["athlete"].pop("team")
    parsed = build_injury_map({"injuries": [{"displayName": "Golden State Warriors", "injuries": [injury]}]})
    assert parsed["GS"][0]["player"] == "Example Player"


def test_optional_fields_and_comment_fallback():
    parsed = build_injury_map({"injuries": [row(details=None, shortComment="Source comment")]})
    assert parsed["GS"][0]["status"] == "Out"
    assert parsed["GS"][0]["reason"] == "Source comment"
    assert parsed["GS"][0]["returnDate"] is None


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        {},
        {"error": "temporarily unavailable"},
        {"injuries": None},
        {"injuries": {}},
        {"injuries": [None]},
        {"injuries": [{"injuries": None}]},
        {"injuries": [row("UNKNOWN")]},
        {"injuries": [row(athlete={})]},
        {"injuries": [row(details=[])]},
    ],
)
def test_malformed_source_is_rejected_instead_of_replacing_cache_with_empty_data(data):
    with pytest.raises(FeedError):
        build_injury_map(data)


def test_explicit_empty_source_snapshot_is_valid():
    assert build_injury_map({"injuries": []}) == {}


@pytest.mark.parametrize(
    "href,expected",
    [
        ("https://www.espn.com/nba/player/_/id/123/example-player", "123"),
        ("https://www.espn.com/nba/player/stats/_/id/456/example-player", "456"),
        ("https://other.example/nba/player/_/id/123/example-player", None),
        ("https://www.espn.com/nba/player/_/id/not-a-number/", None),
        ("https://[malformed", None),
    ],
)
def test_player_id_comes_only_from_explicit_espn_identifiers(href, expected):
    assert athlete_id({"links": [{"href": href}]}) == expected
