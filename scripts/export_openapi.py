"""Generate consumer and provider OpenAPI 3.0.3 files from the actual FastAPI routes."""

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app  # noqa: E402


def compatible_schema(value):
    if isinstance(value, list):
        return [compatible_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    value = {key: compatible_schema(item) for key, item in value.items()}
    choices = value.get("anyOf", [])
    if any(choice == {"type": "null"} for choice in choices):
        non_null = [choice for choice in choices if choice != {"type": "null"}]
        value.pop("anyOf")
        if len(non_null) == 1:
            choice = non_null[0]
            value.update({"allOf": [choice]} if "$ref" in choice else choice)
        else:
            value["anyOf"] = non_null
        value["nullable"] = True
    return value


def documents():
    spec = compatible_schema(copy.deepcopy(app.openapi()))
    spec["openapi"] = "3.0.3"
    spec["servers"] = [{"url": "https://nba-injuries-api.p.rapidapi.com"}]
    spec["components"]["securitySchemes"] = {
        "RapidAPIKey": {"type": "apiKey", "in": "header", "name": "X-RapidAPI-Key"},
        "RapidAPIHost": {"type": "apiKey", "in": "header", "name": "X-RapidAPI-Host"},
    }
    spec["security"] = [{"RapidAPIKey": [], "RapidAPIHost": []}]
    for path in spec["paths"].values():
        for operation in path.values():
            operation.pop("security", None)
    for parameter in spec["paths"]["/v1/injuries"]["get"]["parameters"]:
        # Do not prefill both mutually exclusive filters in API playgrounds.
        if parameter["name"] == "team":
            parameter["example"] = "GSW"
    spec["paths"]["/v1/player/search"]["get"]["parameters"][0]["example"] = "James"
    spec["paths"]["/v1/player/{player_id}"]["get"]["parameters"][0]["example"] = "1966"
    injury = {
        "player": "Example Player",
        "playerId": "123",
        "status": "OUT",
        "statusCode": "OUT",
        "reason": "Ankle",
        "returnDate": None,
    }
    meta = {
        "updatedAt": 1791746877,
        "source": "ESPN",
        "cache": {"ageSeconds": 12, "ttlSeconds": 120, "maxAgeSeconds": 900, "stale": False},
    }
    spec["paths"]["/v1/injuries"]["get"]["responses"]["200"]["content"]["application/json"]["examples"] = {
        "allTeams": {
            "summary": "Illustrative all-team snapshot",
            "value": {**meta, "teams": {"GS": [injury]}, "teamCount": 1, "playerEntryCount": 1},
        },
        "singleTeam": {
            "summary": "Illustrative canonical alias filter",
            "value": {**meta, "team": "GSW", "canonicalTeam": "GSW", "count": 1, "injuries": [injury]},
        },
    }
    for path in spec["paths"].values():
        for operation in path.values():
            if operation["operationId"] == "checkFreshness":
                for code in ("200", "503"):
                    operation["responses"][code]["content"]["application/json"]["example"] = {
                        "status": "fresh" if code == "200" else "stale",
                        "version": spec["info"]["version"],
                        "source": "ESPN",
                        "updatedAt": meta["updatedAt"],
                        "cache": {
                            **meta["cache"],
                            "ageSeconds": 12 if code == "200" else 300,
                            "stale": code == "503",
                        },
                        "alertAfterSeconds": 300,
                        "consecutiveRefreshFailures": 0,
                    }
                continue
            for code, response in operation["responses"].items():
                if code in {"400", "401", "422", "503"}:
                    examples = {
                        "400": ("INVALID_TEAM", "Unknown NBA team abbreviation."),
                        "401": ("UNAUTHORIZED", "Authentication is missing or invalid."),
                        "422": ("VALIDATION_ERROR", "Invalid request parameters."),
                        "503": (
                            "UPSTREAM_UNAVAILABLE",
                            "Injury data is temporarily unavailable. Please retry later.",
                        ),
                    }
                    error_code, message = examples[code]
                    if operation["operationId"] == "searchInjuredPlayers" and code == "400":
                        error_code, message = (
                            "INVALID_SEARCH",
                            "Provide at least two characters after trimming whitespace.",
                        )
                    response["content"]["application/json"]["example"] = {
                        "error": {"code": error_code, "message": message}
                    }
    provider = copy.deepcopy(spec)
    provider["info"]["version"] = "v1"
    provider["servers"] = [{"url": "https://nba-injuries-api-1.onrender.com"}]
    # Gateway credentials are injected by RapidAPI. Importing consumer auth here
    # would accidentally require a second authentication policy at the origin.
    provider.pop("security")
    provider["components"].pop("securitySchemes")
    return {"openapi.json": spec, "rapidapi-import.json": provider}


def main():
    from openapi_spec_validator import validate

    for filename, spec in documents().items():
        validate(spec)
        (ROOT / filename).write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Validated and generated {filename}")


if __name__ == "__main__":
    main()
