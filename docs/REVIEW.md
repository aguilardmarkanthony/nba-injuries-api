# Release review — 2026-10-12

## Confirmed production issues addressed

- Canonical team filters silently returned zero results for teams stored under source aliases. Regression coverage checks every supported spelling and all 30 identities.
- Invalid team strings and whitespace-only player searches produced misleading successes. Validation now distinguishes these cases from genuine empty search/team results.
- Every expired-cache request could start an upstream fetch. Refreshes are serialized and performed by a background worker, with bounded stale responses and retry backoff.
- Source exceptions could surface as unstructured 500 errors. Malformed source responses now retain the prior snapshot, and a documented 503 is used when none is usable.
- Public demo keys were enabled by default in code. They are rejected even if retained in legacy environment configuration. Direct access is explicitly configured and query-key authentication is disabled.
- The team directory depended on which teams had injuries. It now includes the complete supported registry without a source request.

## Findings during the change review

- Added a 15-second bound for cold callers waiting on another refresh; a blocked worker must not hold every request indefinitely.
- The live ESPN feed omits `athlete.id`. Added strict extraction from ESPN profile URLs and tests rejecting foreign hosts, malformed links, and nonnumeric IDs. A live source check found IDs for all 132 entries in the reviewed snapshot.
- Split the consumer and provider OpenAPI files to prevent an import from setting the RapidAPI gateway as its own origin.
- Removed the deprecated HTTP test client dependency in favor of the client supported by the current Starlette release.
- Removed the billing payload from the current tracked tree; its local copy and Git history remain intact. History sanitization is not part of this release.

## Verification and limitations

The regression suite covers alias equivalence, preserved v1 keys, invalid and empty input, missing source data, malformed source payloads, authentication bypass attempts, cache concurrency, stale-data bounds, source retry backoff, background-worker lifecycle, and generated specification drift. Ruff, package compatibility checks, OpenAPI validation, and a runtime dependency audit are included in the release workflow.

The source timestamps describe retrieval, not the time an injury changed. The cache is per process and in memory. Render Free can still incur idle cold starts; this code change does not promise a latency SLA or paid infrastructure. An unavailable source cannot be made fresh by caching. Proposed history, webhooks, shared storage, pricing changes, and canonical v2 responses are documented separately in ROADMAP.md.
