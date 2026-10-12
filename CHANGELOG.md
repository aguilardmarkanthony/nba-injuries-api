# Changelog

## Marketplace configuration — 2026-10-12

- Reduce the public BASIC plan from 30 to 5 requests per month at the owner's request, retaining the monthly hard limit and $0 subscription price.
- Existing subscribers retain their previous RapidAPI plan until they change plans. PRO remains $9.99/month with 500,000 requests/month.
- Update consumer documentation and record the verified provider settings in `docs/PRICING.md`. This is a marketplace configuration change; backend version remains 2.4.0.

## 2.4.0 — 2026-10-12

- Add optional status and comma-separated multi-team filters, with alias deduplication, validation, matching-result counts, and preserved v1 response envelopes.
- Add a normalized `statusCode` without changing source status text; keep GTD and QUESTIONABLE distinct and map unfamiliar source values to UNKNOWN.
- Add exact ESPN player-ID lookup for current injury entries. An empty 200 response does not imply player health or roster membership.
- Add `/v1/freshness`, which observes the background cache without refreshing it, and counts consecutive source failures. Missing, stale, or repeatedly failing snapshots return diagnostic 503 responses.
- Add a production GitHub Actions freshness monitor with bounded retries and a 30-minute schedule; notification delivery follows GitHub account settings.
- Add a standard-library Python integration example, regression coverage, and regenerated RapidAPI specifications and reference documentation.

## 2.3.0 — 2026-10-12

- Accept canonical and source aliases for all 30 teams, preserving v1 source keys in all-team responses.
- Reject invalid/empty team filters with 400 and distinguish valid teams with no entries.
- Return a complete team directory independently of source availability; validate trimmed player searches.
- Add canonical team identities, optional ESPN player IDs, and explicit cache freshness metadata.
- Refresh snapshots in the background with one source fetch at a time, bounded waiting, a 30-second failure backoff, and a 15-minute maximum snapshot age.
- Reject malformed source data without replacing a valid cache; return documented 503 errors when no usable snapshot exists.
- Remove default public demo credentials and query-string authentication. Retain explicitly configured private direct-origin keys; use constant-time comparisons.
- Remove authentication configuration details from public health output and add separate readiness probing.
- Update runtime dependencies, add regression/contract tests and CI, and generate validated consumer/provider OpenAPI specifications.
- Remove the billing test payload from the current tracked tree and ignore private payloads in future commits. Existing Git history is preserved.

### Compatibility notes

Success response fields are additive. All-team keys stay `GS`, `NO`, `NY`, `SA`, `UTAH`, and `WSH`; canonical aliases work for filtering. Invalid teams now return 400 instead of an empty 200. `/v1/teams` now lists all supported teams, including those absent from the injury snapshot. The origin error format is standardized. Old public demo keys and `?api_key=` are intentionally disabled.
