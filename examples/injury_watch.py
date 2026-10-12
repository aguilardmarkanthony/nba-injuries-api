"""Fetch filtered injuries or look up an ESPN player ID through RapidAPI."""

import argparse
import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

HOST = "nba-injuries-api.p.rapidapi.com"


class APIError(RuntimeError):
    pass


def fetch_injuries(api_key: str, *, team=None, teams=None, status=None, player_id=None) -> dict:
    if not api_key.strip():
        raise ValueError("Set RAPIDAPI_KEY to your private subscription key")
    if sum(value is not None for value in (team, teams, player_id)) > 1:
        raise ValueError("Choose one of team, teams, or player_id")
    if player_id is not None:
        if not re.fullmatch(r"[1-9][0-9]{0,11}", player_id):
            raise ValueError("player_id must be a positive ESPN ID of 1-12 digits")
        if status is not None:
            raise ValueError("status applies to injury-list requests, not player lookup")
        path = f"/v1/player/{player_id}"
        params = {}
    else:
        path = "/v1/injuries"
        params = {
            key: value
            for key, value in {"team": team, "teams": teams, "status": status}.items()
            if value is not None
        }
    url = f"https://{HOST}{path}" + ("?" + urlencode(params) if params else "")
    request = Request(
        url,
        headers={"X-RapidAPI-Key": api_key.strip(), "X-RapidAPI-Host": HOST, "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=75) as response:
            data = json.load(response)
    except HTTPError as error:
        code = error.code
        error.close()
        retry = " Respect Retry-After and your plan limits before retrying." if code in {429, 503} else ""
        raise APIError(f"API returned HTTP {code}.{retry}") from None
    except (URLError, TimeoutError, OSError):
        raise APIError("API connection failed; verify connectivity and try again later") from None
    except (ValueError, UnicodeError):
        raise APIError("API returned an invalid JSON response") from None
    if not isinstance(data, dict):
        raise APIError("API returned an unexpected response shape")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--team", help="Single team, e.g. GSW")
    selection.add_argument("--teams", help="Comma-separated teams, e.g. GSW,NOP")
    selection.add_argument("--player-id", help="ESPN athlete ID")
    parser.add_argument("--status", help="Injury status code, e.g. OUT")
    args = parser.parse_args()
    try:
        data = fetch_injuries(os.environ.get("RAPIDAPI_KEY", ""), **vars(args))
    except (ValueError, APIError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
