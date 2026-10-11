# NBA Injuries API

Current NBA injury snapshots sourced from ESPN, with team filters, a complete team directory, and player-name search. The public v1 contract preserves ESPN's historical team keys while accepting canonical NBA abbreviations in requests.

| Resource | Location |
| --- | --- |
| GitHub source | https://github.com/aguilardmarkanthony/nba-injuries-api |
| RapidAPI listing | https://rapidapi.com/aguilardmarkanthony/api/nba-injuries-api |
| Consumer gateway | https://nba-injuries-api.p.rapidapi.com |
| Render origin | https://nba-injuries-api-1.onrender.com |
| Render service | `srv-d6ce52ctgctc73evpb40`, deployed from GitHub `main` |
| Consumer reference | [docs/rapidapi.md](docs/rapidapi.md) |
| Consumer specification | [openapi.json](openapi.json) |
| Provider import specification | [rapidapi-import.json](rapidapi-import.json) |
| Latest production verification | [October 12, 2026 release](docs/RELEASE-2026-10-12.md) |

## Quick start

Subscribe through RapidAPI and provide both gateway headers. Keep the key on your server or in an environment variable.

```sh
curl 'https://nba-injuries-api.p.rapidapi.com/v1/injuries?team=GSW' \
  -H "X-RapidAPI-Key: $RAPIDAPI_KEY" \
  -H 'X-RapidAPI-Host: nba-injuries-api.p.rapidapi.com'
```

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/v1/injuries` | All injury entries, optionally filtered by `team` |
| GET | `/v1/teams` | All 30 teams, including canonical codes and aliases |
| GET | `/v1/player/search?name=James` | Case-insensitive substring search within current injury entries |
| GET | `/v1/health` | Process liveness and release version |

`/v1/ready` is an origin readiness probe. It returns 503 when no sufficiently recent snapshot is available; bounded stale data remains ready and is explicitly marked stale.

## Local development

Python 3.10 or newer is required. CI validates Python 3.11 and 3.13.

```sh
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install -r requirements-dev.txt
# Set DIRECT_API_KEYS to a private development key in your environment.
uvicorn app:app --host 127.0.0.1 --port 8000
```

For local requests, send your explicitly configured key as `X-API-Key`. The application reads process environment variables; it does not automatically load `.env` files. `.env.example` documents supported settings.

| Variable | Default | Meaning |
| --- | --- | --- |
| `RAPIDAPI_PROXY_SECRET` | empty | Origin-only secret supplied by RapidAPI; configure in Render, never in client code |
| `DIRECT_API_KEYS` | empty | Optional comma-separated private keys for direct origin access |
| `DEMO_KEYS` | empty | Legacy compatibility setting for private direct keys; the two old public demo keys are always rejected |
| `CACHE_TTL` | 120 | Snapshot refresh interval in seconds |
| `MAX_CACHE_AGE` | 900 | Maximum snapshot age allowed during refresh/source failure; must be at least `CACHE_TTL` |

Query-string authentication is disabled. A client-supplied `X-RapidAPI-Key` alone does not authenticate direct origin requests. The backend validates RapidAPI's proxy secret using constant-time comparison.

## Reliability and data semantics

- A single background worker per application process attempts to refresh snapshots. Concurrent requests share one refresh rather than creating a burst of source requests.
- Fresh cached reads avoid an upstream round trip. During refresh or failure, a prior snapshot may be served up to `MAX_CACHE_AGE` seconds old with `cache.stale: true`. Failed refreshes retry after 30 seconds.
- Without a usable snapshot, data routes return 503 with `Retry-After: 30`. Cold request waiters wait at most 15 seconds for another refresh. ESPN requests use a 3.05-second connection timeout and a 10-second read timeout.
- Malformed upstream responses are rejected without overwriting the last valid snapshot. An explicit empty `injuries` array is a valid zero-entry snapshot.
- `updatedAt` is the time a snapshot was successfully fetched and parsed. It is not a guarantee that ESPN changed an injury at that time.
- `playerId` uses an explicit ESPN athlete ID or a numeric ID in an ESPN player-profile link. It is null if neither is supplied.
- Source status and estimated return dates are preserved. An empty result for a valid team means the source snapshot contains no entries for that team; it does not certify player availability.
- The cache is in memory and per process. Multiple workers/instances need shared storage before scaling out.
- The current Render service uses Free compute. It can sleep after idle periods, so an initial request can still take about a minute despite these cache improvements. A paid always-on instance is a separate hosting decision.

## Verification and release

```sh
python -m pytest -q
ruff check .
ruff format --check .
python -m pip check
pip-audit -r requirements.txt
python scripts/export_openapi.py
```

Review the diff, commit all tested source/documentation/spec changes, and push to GitHub. After CI passes, merge to `main`, verify the Render deploy and authenticated API behavior, then update the existing RapidAPI v1 listing. Import `rapidapi-import.json` into the provider definition: **do not use the consumer `openapi.json` as the origin configuration**, because its server is the RapidAPI gateway.

Set the Render health-check path to `/v1/health`. Monitor `/v1/ready` and freshness metadata separately from liveness. See [CHANGELOG.md](CHANGELOG.md) for compatibility changes and [docs/ROADMAP.md](docs/ROADMAP.md) for proposed enhancements.

The previously tracked `payload.json` billing sample is excluded from the current repository tree. Historical commits may still contain it; no history rewrite was performed.
