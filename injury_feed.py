"""Fetch and validate ESPN's current injury snapshot before it enters the cache."""

import json
import re
from typing import Any
from urllib.parse import urlparse

import requests

from teams import CANONICAL_TO_SOURCE, NAME_TO_CANONICAL, source_team

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"


class FeedError(ValueError):
    """The source response cannot safely be represented as an injury snapshot."""


def load_data(path: str | None = None) -> dict:
    if path:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    with requests.get(ESPN_URL, timeout=(3.05, 10)) as response:
        response.raise_for_status()
        return response.json()


def text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def object_field(record: dict, key: str) -> dict:
    value = record.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise FeedError(f"Invalid {key} object in source data")
    return value


def athlete_id(athlete: dict) -> str | None:
    value = athlete.get("id")
    if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).isdigit():
        return str(value)
    # The current injury feed omits athlete.id but supplies ESPN profile links.
    # Accept only ESPN's explicit numeric player-id path; never derive IDs from names.
    links = athlete.get("links")
    if isinstance(links, list):
        for link in links:
            if not isinstance(link, dict):
                continue
            try:
                url = urlparse(text(link.get("href")))
            except ValueError:
                continue
            if url.hostname not in {"espn.com", "www.espn.com"}:
                continue
            match = re.search(r"/nba/player/(?:[^/]+/)?_/id/(\d+)(?:/|$)", url.path)
            if match:
                return match.group(1)
    return None


def build_injury_map(data: dict) -> dict[str, list[dict]]:
    if not isinstance(data, dict) or not isinstance(data.get("injuries"), list):
        raise FeedError("Source response is missing its injuries array")
    result: dict[str, list[dict]] = {}
    for item in data["injuries"]:
        if not isinstance(item, dict):
            raise FeedError("Invalid source injury record")
        grouped = "injuries" in item and "athlete" not in item
        rows = item.get("injuries") if grouped else [item]
        if not isinstance(rows, list):
            raise FeedError("Invalid source team injury list")
        fallback = text(item.get("abbreviation")) if grouped else ""
        if grouped and not fallback:
            fallback = text(object_field(item, "team").get("abbreviation"))
        if grouped and not fallback:
            canonical = NAME_TO_CANONICAL.get(text(item.get("displayName")).casefold())
            fallback = CANONICAL_TO_SOURCE.get(canonical, "")
        for row in rows:
            if not isinstance(row, dict):
                raise FeedError("Invalid source injury row")
            athlete = object_field(row, "athlete")
            team = object_field(athlete, "team")
            code = text(team.get("abbreviation")) or fallback
            name = text(athlete.get("displayName")) or text(athlete.get("shortName"))
            if not name or not code:
                raise FeedError("Source injury is missing a player name or team")
            try:
                code = source_team(code)
            except ValueError as exc:
                raise FeedError("Source injury has an unknown team") from exc
            details = object_field(row, "details")
            fantasy = object_field(details, "fantasyStatus")
            result.setdefault(code, []).append(
                {
                    "player": name,
                    "status": text(fantasy.get("description")) or text(row.get("status")),
                    "reason": text(details.get("type"))
                    or text(details.get("detail"))
                    or text(row.get("shortComment")),
                    "returnDate": text(details.get("returnDate")) or None,
                    "playerId": athlete_id(athlete),
                }
            )
    return result
