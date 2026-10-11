# NBA Injuries API

Retrieve current ESPN-sourced NBA injury reports, filter by team, discover all 30 team codes, and search injured players by name. Responses include snapshot timestamps, cache age, and stale-data indicators.

## Authentication and first request

Subscribe to a plan and send `X-RapidAPI-Key` and `X-RapidAPI-Host` with every gateway request. No additional origin API key is required through RapidAPI.

```sh
curl 'https://nba-injuries-api.p.rapidapi.com/v1/injuries?team=GSW' \
  -H "X-RapidAPI-Key: $RAPIDAPI_KEY" \
  -H 'X-RapidAPI-Host: nba-injuries-api.p.rapidapi.com'
```

## Endpoints

### GET /v1/injuries

Omit `team` for all injury entries. Provide an optional, case-insensitive abbreviation to filter one team. Leading and trailing whitespace is trimmed. Invalid or empty abbreviations return HTTP 400, rather than a misleading empty success.

Both canonical and historical ESPN aliases work:

| Canonical code | Accepted source alias |
| --- | --- |
| GSW | GS |
| NOP | NO |
| NYK | NY |
| SAS | SA |
| UTA | UTAH |
| WAS | WSH |

Other supported codes: ATL, BOS, BKN, CHA, CHI, CLE, DAL, DEN, DET, HOU, IND, LAC, LAL, MEM, MIA, MIL, MIN, OKC, ORL, PHI, PHX, POR, SAC, TOR.

For compatibility, all-team responses retain historical source keys such as `GS` and `NO`. Single-team responses retain the requested code in `team` and include its normalized identity in `canonicalTeam`.

Illustrative single-team response (not a live injury report):

```json
{
  "team": "GSW",
  "canonicalTeam": "GSW",
  "count": 1,
  "injuries": [{"player": "Example Player", "playerId": "123", "status": "OUT", "reason": "Ankle", "returnDate": null}],
  "updatedAt": 1791746877,
  "source": "ESPN",
  "cache": {"ageSeconds": 12, "ttlSeconds": 120, "maxAgeSeconds": 900, "stale": false}
}
```

The all-team response contains `teams` (an object mapping source abbreviations to injury arrays), `teamCount`, `playerEntryCount`, `updatedAt`, `source`, and `cache`. `teamCount` counts teams with entries in the snapshot, so it may be less than 30. A valid team with no reported entries returns HTTP 200 with `count: 0` and `injuries: []`.

### GET /v1/teams

Returns all 30 supported teams without requiring a source fetch. `teams` contains the historical source codes, `canonicalTeams` contains normalized codes, and `details` supplies each team's name, canonical `code`, `sourceCode`, and accepted `aliases`.

### GET /v1/player/search?name=James

`name` is required: 2–100 characters, with at least two characters after trimming. Search is case-insensitive and matches a substring of the player's name. Results include `team`, `canonicalTeam`, and injury fields, plus response-level `query`, `count`, `updatedAt`, `source`, and `cache`.

This searches the current injury snapshot, not the entire NBA player roster. No matching injury entries produces an empty `results` list.

### GET /v1/health

Returns process liveness, API implementation version, and `cacheTtlSeconds`. It does not certify upstream availability or fresh injury data.

## Freshness and field meanings

- Snapshots are refreshed in the background approximately every 120 seconds while the service is running. Source publication delays are additional.
- `updatedAt` is a Unix timestamp in seconds marking successful retrieval and parsing, not an individual injury update time.
- `cache.ageSeconds` gives snapshot age. `cache.stale` becomes true when the refresh interval has elapsed. A previous valid snapshot can be served for up to 900 seconds during refresh or source failure. Older data is not served.
- `status` preserves the source value. Common codes are `GTD` (game-time decision), `OUT`, and `OFS` (out for season); source text may vary.
- `returnDate` is an estimated source-provided return date, not a confirmed recovery date. It can be null.
- `playerId` is an ESPN athlete identifier extracted from the source record or ESPN profile link; null when unavailable.
- `reason` contains the source's injury description and can be empty when unavailable.
- No injury entry is not a guarantee of player availability. This API is not an official NBA injury-report service.
- Idle periods on the current free hosting instance may cause a slower first request. Subsequent cached requests are normally faster; no latency guarantee is made.

## Errors and retries

Backend errors use `{"error":{"code":"...","message":"..."}}`, with optional details. RapidAPI-generated authentication, subscription, or quota errors may use the gateway's own format.

| HTTP status | Code | Meaning |
| --- | --- | --- |
| 400 | INVALID_TEAM | Invalid or empty team abbreviation; response includes `validTeams` |
| 400 | INVALID_SEARCH | Search text is shorter than two characters after trimming |
| 401 | UNAUTHORIZED | Missing or invalid origin authorization |
| 422 | VALIDATION_ERROR | Missing required parameter or invalid parameter length |
| 429 | Gateway-defined | RapidAPI plan quota or rate limit reached |
| 503 | UPSTREAM_UNAVAILABLE | No sufficiently recent snapshot is available |

For 503, honor the `Retry-After` header (normally 30 seconds). For 429, follow the gateway's quota/rate-limit information. Avoid rapid retry loops. Polling more frequently than the refresh interval does not guarantee fresher source data and still consumes your plan's request allowance.

## Release 2.3.0 / public API v1

Fixed canonical team aliases and invalid-team responses; added team-directory and player-search documentation, player IDs, cache freshness metadata, source-failure handling, and secure origin defaults. Existing v1 success fields and all-team source keys are preserved. Query-string API keys and public demo credentials are no longer accepted at the origin.

Source, changelog, and specifications: https://github.com/aguilardmarkanthony/nba-injuries-api
