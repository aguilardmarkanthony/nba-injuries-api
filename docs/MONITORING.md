# Freshness monitoring

`/v1/health` remains Render's liveness check. `/v1/ready` permits bounded stale snapshots for serving requests. `/v1/freshness` separately checks whether the background refresh is keeping up; it never fetches source data itself and cannot hide a stalled worker by refreshing during a probe.

The freshness endpoint is public at the origin and contains no player data, credentials, exception messages, or source response bodies. Through RapidAPI, gateway subscription/authentication rules still apply.

## Thresholds

- Fresh: snapshot age is below `CACHE_TTL` (120 seconds by default).
- Delayed but acceptable: age has reached the TTL but is below `FRESHNESS_ALERT_SECONDS` (normally 300), with fewer than three consecutive refresh failures.
- Alert/HTTP 503: no snapshot, age at or above the alert threshold, or at least three consecutive failed source refreshes.
- A successful source refresh resets the failure count. Source exceptions and parsing failures both count; retry-backoff skips do not.

Both success and failure use the same documented freshness schema. There is no source fetch or retry delay inside this endpoint. Responses use `Cache-Control: no-store`.

## Production workflow and notifications

The [Production freshness workflow](https://github.com/aguilardmarkanthony/nba-injuries-api/actions/workflows/freshness.yml) runs at minutes 17 and 47 of each hour (UTC). Its public-origin requests do not consume RapidAPI subscription calls. Each run makes up to three attempts with a 75-second request timeout and 15 seconds between failures, allowing for an idle Render startup. An exhausted check exits nonzero and creates a failed workflow run and diagnostic annotation. The job is capped at six minutes.

GitHub sends web/email workflow notifications according to the account's existing preferences. In GitHub notification settings, enable Actions notifications and optionally choose only failed workflows. This release does not change global notification preferences or configure SMS, email recipients, or messaging integrations. [GitHub notification settings](https://docs.github.com/en/subscriptions-and-notifications/how-tos/managing-github-actions-notifications).

Scheduled Actions can be delayed or dropped under load. Public-repository schedules may be disabled after 60 days without repository activity. This is a best-effort operational check, not an uptime SLA or a continuous alert service. [GitHub schedule behavior](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).

Run the workflow manually after each release, or run `python scripts/check_freshness.py` locally. A passing result confirms retrieval freshness at the check time; it cannot determine whether ESPN has published the latest real-world injury information. The checks can wake the Free instance but do not remove its idle behavior between checks.

## Responding to an alert

1. Inspect the workflow output and `/v1/freshness`. Liveness may remain healthy while source retrieval is failing.
2. Check Render logs for source timeout or validation-failure events. Do not add raw source bodies or credentials to logs.
3. Determine whether the source changed its schema, is unavailable, or the refresh worker stalled. Preserve bounded stale-data handling during recovery.
4. Fix and test the underlying issue, push through GitHub, verify deployment, and rerun the monitor. Do not raise the age limit simply to silence an alert.

## Hosting proposal

The current Render service remains on Free compute. The entry paid web-service compute option is listed at US$7/month for 512 MB and 0.5 CPU as of October 12, 2026; taxes and usage-dependent charges may be additional. This would remove idle spin-down, but source freshness would still require monitoring. Upgrading the workspace plan alone does not change the service's compute plan. No billing change is part of this release. [Render pricing](https://render.com/pricing), [Free instance limitations](https://render.com/docs/free).
