# Proposed enhancements

Release 2.4.0 implements status and multi-team filters, normalized status codes, player-ID lookup, a freshness probe and scheduled GitHub monitor, and a runnable Python example. The remaining items below are proposals.

1. **Remove idle cold starts:** move the existing Render service from Free to always-on compute after approving the hosting cost. Set latency and freshness targets based on actual monitoring.
2. **Persist injury changes:** store append-only snapshots keyed by ESPN player ID and team identity. Expose history and changes-since queries with cursor pagination and retention limits. Verify coverage and source permissions before promising historical completeness.
3. **Client libraries:** expand the Python example into supported Python/TypeScript clients with typed responses and documented retry policies, guided by customer demand.
4. **Change webhooks:** notify subscribers of status transitions with signatures, idempotency keys, retry backoff, delivery history, and per-plan limits.
5. **Shared cache and resilience:** add durable/shared snapshots before running multiple API processes or instances. Build on the existing freshness monitor with continuous monitoring and outage/cold-start recovery checks.
6. **Developer onboarding:** add a JavaScript integration guided by customer demand. The existing Python example supplies a working starting point. At the owner's request, the public BASIC allowance changed from 30 to 5 requests/month on October 12, 2026; existing subscribers keep their prior contracts. See [plan settings](PRICING.md). Evaluate activation under the new allowance before proposing further quota changes.
7. **Canonical v2:** consider a flat, paginated injury collection with canonical team identifiers and one consistent response envelope. Keep the existing v1 contract supported during migration.

Prioritize correctness and active customer usage before expanding infrastructure or promising real-time updates. The October 12 review observed nine calls in the prior 30-day dashboard window, eight of which were review tests; subscriber count alone is not evidence of active usage.
