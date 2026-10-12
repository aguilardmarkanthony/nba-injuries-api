import json
from pathlib import Path

from openapi_spec_validator import validate

from scripts.export_openapi import documents


def test_committed_specs_match_the_app_and_preserve_provider_routing():
    root = Path(__file__).resolve().parents[1]
    specs = documents()
    for filename, spec in specs.items():
        validate(spec)
        assert json.loads((root / filename).read_text(encoding="utf-8")) == spec
        assert set(spec["paths"]) == {
            "/v1/injuries",
            "/v1/teams",
            "/v1/player/search",
            "/v1/player/{player_id}",
            "/v1/health",
            "/v1/freshness",
        }
    assert specs["openapi.json"]["servers"][0]["url"] == "https://nba-injuries-api.p.rapidapi.com"
    provider = specs["rapidapi-import.json"]
    assert provider["servers"][0]["url"] == "https://nba-injuries-api-1.onrender.com"
    assert "security" not in provider
    assert "securitySchemes" not in provider["components"]
    injuries = provider["paths"]["/v1/injuries"]["get"]
    params = {item["name"]: item for item in injuries["parameters"]}
    assert params["team"]["example"] == "GSW" and "example" not in params["teams"]
    freshness = provider["paths"]["/v1/freshness"]["get"]
    diagnostic = freshness["responses"]["503"]["content"]["application/json"]
    assert diagnostic["example"]["status"] == "stale"
    assert "error" not in diagnostic["example"]
