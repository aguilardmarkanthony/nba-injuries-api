# NBA Injuries API

Retrieve current ESPN-sourced NBA injury reports, filter by team and status, discover all 30 team codes, and find injured players by name or ESPN ID. Responses include snapshot timestamps, normalized status codes, cache age, and stale-data indicators.

## Authentication and first request

Subscribe to a plan and send `X-RapidAPI-Key` and `X-RapidAPI-Host` with every gateway request. No additional origin API key is required through RapidAPI.

BASIC: 5 evaluation requests/month free. PRO: $5/month for 500,000. Both have hard limits. Existing subscribers keep their prior plans.

```sh
curl 'https://nba-injuries-api.p.rapidapi.com/v1/injuries?team=GSW' \
  -H "X-RapidAPI-Key: $RAPIDAPI_KEY" \
  -H 'X-RapidAPI-Host: nba-injuries-api.p.rapidapi.com'
```

## Endpoints

### GET /v1/injuries

Omit the filters for all injury entries. Provide an optional, case-insensitive `team` abbreviation to filter one team. Leading and trailing whitespace is trimmed. Invalid or empty abbreviations return HTTP 400.

| Parameter | Meaning |
| --- | --- |
| `team` | One abbreviation, such as `GSW`; maximum 16 characters |
| `teams` | 1-30 comma-separated abbreviations, such as `GSW,NOP`; maximum 256 characters; aliases are deduplicated |
| `status` | One normalized status code or readable alias; maximum 64 characters; may combine with `team` or `teams` |

Use either `team` or `teams`. Supplying both returns `INVALID_FILTER`. Empty items such as `teams=GSW,` are invalid.

Examples: `/v1/injuries?team=GSW&status=OUT`, `/v1/injuries?teams=GSW,NOP`, and `/v1/injuries?status=QUESTIONABLE`.

Status codes are case-insensitive: `OUT`, `GTD`, `OFS`, `QUESTIONABLE`, `DOUBTFUL`, `PROBABLE`, `DAY_TO_DAY`, and `UNKNOWN`. Readable aliases include `Game-Time Decision` for GTD, `Out For Season` for OFS, and `Day-To-Day` or DTD for DAY_TO_DAY. GTD and QUESTIONABLE remain separate categories. Unsupported filter values return `INVALID_STATUS` with `validStatuses`.

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
  "injuries": [{"player": "Example Player", "playerId": "123", "status": "OUT", "statusCode": "OUT", "reason": "Ankle", "returnDate": null}],
  "updatedAt": 1791746877,
  "source": "ESPN",
  "cache": {"ageSeconds": 12, "ttlSeconds": 120, "maxAgeSeconds": 900, "stale": false}
}
```

The all-team response contains `teams` (an object mapping source abbreviations to injury arrays), `teamCount`, `playerEntryCount`, `updatedAt`, `source`, and `cache`. `teamCount` counts teams with entries in the snapshot, so it may be less than 30. A valid team with no reported entries returns HTTP 200 with `count: 0` and `injuries: []`.

With `teams` or a status-only filter, the response uses that same all-team envelope. Only teams with matching entries are included, and both counts describe the filtered results. No matches returns HTTP 200 with `teams: {}`, `teamCount: 0`, and `playerEntryCount: 0`. Filtering does not modify the shared snapshot.

### GET /v1/teams

Returns all 30 supported teams without requiring a source fetch. `teams` contains the historical source codes, `canonicalTeams` contains normalized codes, and `details` supplies each team's name, canonical `code`, `sourceCode`, and accepted `aliases`.

### GET /v1/player/search?name=James

`name` is required: 2–100 characters, with at least two characters after trimming. Search is case-insensitive and matches a substring of the player's name. Results include `team`, `canonicalTeam`, and injury fields, plus response-level `query`, `count`, `updatedAt`, `source`, and `cache`.

This searches the current injury snapshot, not the entire NBA player roster. No matching injury entries produces an empty `results` list.

### GET /v1/player/{player_id}

Look up current injury entries by exact ESPN athlete ID, for example `/v1/player/1966`. The ID must contain 1-12 ASCII digits, be positive, and have no leading zero. Invalid IDs return 422.

The response contains `playerId`, `count`, `results`, `updatedAt`, `source`, and `cache`. Each result includes team context and the usual injury fields. No matching entry returns an empty 200 response; it does not confirm that the player exists or is healthy. Obtain IDs from injury/search responses rather than guessing from player names.

### GET /v1/health

Returns process liveness, API implementation version, and `cacheTtlSeconds`. It does not certify upstream availability or fresh injury data.

### GET /v1/freshness

Checks the existing background cache without making an on-demand source request. Returns `status`, `version`, `source`, `updatedAt`, `cache`, `alertAfterSeconds`, and `consecutiveRefreshFailures`.

| Status | HTTP | Meaning |
| --- | --- | --- |
| `fresh` | 200 | Snapshot is younger than its refresh interval |
| `delayed` | 200 | Refresh interval elapsed, but snapshot is below the alert threshold and fewer than three refresh attempts have failed consecutively |
| `stale` | 503 | Snapshot age reached the alert threshold, normally 300 seconds |
| `degraded` | 503 | Three or more consecutive refresh attempts failed before the age threshold was reached |
| `unavailable` | 503 | No successful snapshot has been loaded; `cache` and `updatedAt` are null |

**This endpoint uses the same diagnostic response shape for 200 and 503**, rather than the normal error envelope. It returns `Cache-Control: no-store` and a retry header on 503. This measures retrieval freshness, not when ESPN last changed a player's injury report.

## Freshness and field meanings

- Snapshots are refreshed in the background approximately every 120 seconds while the service is running. Source publication delays are additional.
- `updatedAt` is a Unix timestamp in seconds marking successful retrieval and parsing, not an individual injury update time.
- `cache.ageSeconds` gives snapshot age. `cache.stale` becomes true when the refresh interval has elapsed. A previous valid snapshot can be served for up to 900 seconds during refresh or source failure. Older data is not served.
- `status` preserves the source value. Common codes are `GTD` (game-time decision), `OUT`, and `OFS` (out for season); source text may vary.
- `statusCode` is the normalized filter code. Unrecognized or empty source text maps to `UNKNOWN`, while the original `status` is retained. This field is included on list, search, and player-ID responses.
- `returnDate` is an estimated source-provided return date, not a confirmed recovery date. It can be null.
- `playerId` is an ESPN athlete identifier extracted from the source record or ESPN profile link; null when unavailable.
- `reason` contains the source's injury description and can be empty when unavailable.
- No injury entry is not a guarantee of player availability. This API is not an official NBA injury-report service.
- Idle periods on the current free hosting instance may cause a slower first request. Subsequent cached requests are normally faster; no latency guarantee is made.

## Errors and retries

Backend errors use `{"error":{"code":"...","message":"..."}}`, with optional details, except for the diagnostic 503 from `/v1/freshness` described above. RapidAPI-generated authentication, subscription, or quota errors may use the gateway's own format.

| HTTP status | Code | Meaning |
| --- | --- | --- |
| 400 | INVALID_TEAM | Invalid or empty team abbreviation; response includes `validTeams` |
| 400 | INVALID_SEARCH | Search text is shorter than two characters after trimming |
| 400 | INVALID_STATUS | Unsupported or empty status filter; response includes `validStatuses` |
| 400 | INVALID_FILTER | Both `team` and `teams` were supplied |
| 401 | UNAUTHORIZED | Missing or invalid origin authorization |
| 422 | VALIDATION_ERROR | Missing required parameter, invalid length, or invalid player-ID format |
| 429 | Gateway-defined | RapidAPI plan quota or rate limit reached |
| 503 | UPSTREAM_UNAVAILABLE | No sufficiently recent snapshot is available |

For 503, honor the `Retry-After` header (normally 30 seconds). For 429, follow the gateway's quota/rate-limit information. Avoid rapid retry loops. Polling more frequently than the refresh interval does not guarantee fresher source data and still consumes your plan's request allowance.

## Working Python example

Download `examples/injury_watch.py` from the GitHub repository. Set `RAPIDAPI_KEY` in your environment, then run:

```sh
python examples/injury_watch.py --team GSW --status OUT
python examples/injury_watch.py --teams GSW,NOP --status GTD
python examples/injury_watch.py --player-id 1966
```

The example uses Python's standard library, makes one gateway request, prints JSON, and exits with an error on a failed request. It never places the API key in the URL or automatically retries billable requests.

## Release 2.4.0 / public API v1

Added optional status and multi-team filters, normalized `statusCode`, exact player-ID lookup, a separate freshness endpoint, production monitoring, and a runnable Python integration. Existing unfiltered and single-team response envelopes and historical team keys remain supported.

Source, changelog, and specifications: https://github.com/aguilardmarkanthony/nba-injuries-api
