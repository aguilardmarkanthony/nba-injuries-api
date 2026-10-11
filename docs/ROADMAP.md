# Proposed enhancements

These are proposals, not features claimed by the current API.

1. **Remove idle cold starts:** move the existing Render service from Free to always-on compute after approving the hosting cost. Set latency and freshness targets based on actual monitoring.
2. **Persist injury changes:** store append-only snapshots keyed by ESPN player ID and team identity. Expose history and changes-since queries with cursor pagination and retention limits. Verify coverage and source permissions before promising historical completeness.
3. **Status filters:** add documented source-status filters and stable normalized status fields while retaining the original status text. Define mappings before introducing them.
4. **Change webhooks:** notify subscribers of status transitions with signatures, idempotency keys, retry backoff, delivery history, and per-plan limits.
5. **Shared cache and resilience:** add durable/shared snapshots before running multiple API processes or instances. Monitor source parse failures and freshness; test outage recovery and cold starts independently.
6. **Developer onboarding:** add JavaScript/Python quick starts, a small integration example, and a realistic trial allowance. The current 30-request free allowance and grandfathered 500,000-request free plan merit review; subscriber prices and quotas require a deliberate migration plan.
7. **Canonical v2:** consider a flat, paginated injury collection with canonical team identifiers and one consistent response envelope. Keep the existing v1 contract supported during migration.

Prioritize correctness and active customer usage before expanding infrastructure or promising real-time updates. The October 12 review observed nine calls in the prior 30-day dashboard window, eight of which were review tests; subscriber count alone is not evidence of active usage.
