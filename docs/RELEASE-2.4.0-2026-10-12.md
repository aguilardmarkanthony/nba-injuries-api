# Production release verification: 2.4.0

Verified on October 12, 2026. Counts and timings below describe the observed snapshot, not future values or a latency guarantee.

- [Implementation PR #2](https://github.com/aguilardmarkanthony/nba-injuries-api/pull/2) merged as `2606e9e68c9d3bda2bf04991304d53b1323a6e37`.
- [Main CI](https://github.com/aguilardmarkanthony/nba-injuries-api/actions/runs/38189385740) passed on Python 3.11 and 3.13. All 159 tests passed, along with Ruff lint/format, dependency compatibility, and specification drift checks. The runtime dependency audit found no known vulnerabilities. Both OpenAPI 3.0.3 specifications also passed local validation.
- Render's public health route and the authenticated RapidAPI gateway both reported version `2.4.0`.
- The existing RapidAPI v1 definition was updated from `rapidapi-import.json`. All six endpoints appeared in the public listing, including `checkFreshness` and `getPlayerInjuries`.
- Listing descriptions, the GitHub website link, and `/v1/health` monitoring path were read back and matched `docs/rapidapi-listing.json`.
- The saved marketplace documentation exactly matched all 9,654 characters of `docs/rapidapi.md`; the public page showed the new sections and complete GitHub link. An initial 10,040-character upload was truncated by the editor. The repeated previous-release paragraph was removed, the document was republished, and the full saved copy was verified. Earlier changes remain in the changelog.

## Authenticated gateway checks

All 15 endpoint checks passed through `nba-injuries-api.p.rapidapi.com` using the existing subscription. A sixteenth live check exercised the committed Python integration example.

| Request | Verified result |
| --- | --- |
| `/v1/health` | 200; version `2.4.0` |
| `/v1/injuries` | 200; 133 entries across 30 teams; counts matched arrays and every entry had a documented `statusCode` |
| `/v1/injuries?status=OUT` | 200; 18 OUT entries across 14 teams |
| `/v1/injuries?teams=GSW,NOP&status=GTD` | 200; 2 GTD entries across the requested teams; historical `GS`/`NO` keys preserved |
| `/v1/injuries?teams=GS,GSW,NOP&status=GTD` | 200; same entries, with aliases deduplicated |
| `/v1/injuries?team=GSW&status=GTD` | 200; 1 entry; existing single-team envelope and canonical `GSW` identity preserved |
| `/v1/injuries?status=invalid` | 400; `INVALID_STATUS` and supported status list |
| `/v1/injuries?team=GS&teams=NO` | 400; `INVALID_FILTER` |
| `/v1/injuries?teams=GS,` | 400; `INVALID_TEAM` |
| `/v1/player/4585618` | 200; exact ID match, team context, and normalized status |
| `/v1/player/999999999999` | 200; empty results |
| `/v1/player/abc` | 422; `VALIDATION_ERROR` |
| `/v1/player/search?name=James` | 200; 1 matching result; static search route remains accessible |
| `/v1/teams` | 200; all 30 directory entries |
| `/v1/freshness` | 200; fresh snapshot, version `2.4.0`, no consecutive refresh failures |
| `examples/injury_watch.py`, `fetch_injuries(teams="GSW,NOP", status="GTD")` | Successful live request; 2 matching entries and consistent counts |

Observed warm endpoint request durations ranged from 317 to 936 milliseconds. These measurements followed deployment and do not measure idle cold starts. Keys were used only in request headers/process memory and were excluded from result files and GitHub.

An additional unauthenticated direct-origin request to `/v1/player/4585618` returned 401, confirming the new data route requires authorization.

## Monitoring

The `Production freshness` workflow is active on GitHub, with a schedule at minutes 17 and 47 of each hour (UTC). Its [first manual production run](https://github.com/aguilardmarkanthony/nba-injuries-api/actions/runs/38189599105) passed: snapshot age 41 seconds, alert threshold 300 seconds, and zero consecutive refresh failures.

The monitor uses the public origin without a RapidAPI subscription key. It records a failed workflow if bounded retries cannot obtain a sufficiently fresh snapshot. Failure paths are covered by automated tests; no production outage was induced. Notification delivery follows the account's GitHub Actions preferences and was not separately verified. Scheduled Actions are best effort and do not guarantee continuous coverage.

Render remains on Free compute. Monitoring does not eliminate idle startup delays; paid compute, durable history, and webhooks remain proposals. See [monitoring operations](MONITORING.md), [code review](REVIEW-2.4.0.md), and [roadmap](ROADMAP.md).
