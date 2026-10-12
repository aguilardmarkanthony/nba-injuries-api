# Release 2.4.0 code review

## Compatibility and correctness

- Existing unfiltered and single-team v1 envelopes and historical ESPN team keys are preserved. Status codes are additive; source text is unchanged.
- GTD is not silently treated as QUESTIONABLE. Unknown source statuses are retained and receive UNKNOWN; unsupported request filters are rejected.
- Filtering creates new collections without modifying the shared cache. Counts are recalculated from matching entries. Multi-team aliases are deduplicated, and empty items or simultaneous `team`/`teams` are rejected before a source read.
- The player-ID route follows `/v1/player/search` to avoid shadowing it. IDs are matched exactly, and the response does not equate a missing injury with a healthy player or verified roster membership.
- Specification examples do not prefill both mutually exclusive team parameters in API playgrounds.

## Monitoring and example review

- The freshness endpoint observes cached state without calling `get()` or initiating a source request, so a stalled background worker cannot be hidden by the monitor itself.
- Failure counters are protected by the cache lock, increment only on attempted refresh failures, and reset after success. No source bodies or exception messages are exposed.
- Missing snapshots, age boundaries, delayed refreshes, repeated failures, and recovery are covered by regression tests. A dedicated 503 diagnostic schema and examples avoid misdocumenting freshness errors as the standard error envelope.
- Hardened monitor validation against malformed JSON field types, oversized responses, startup HTML pages, and unexpected HTTP statuses. Retries are bounded; production monitoring uses no API credentials.
- The Python integration encodes query parameters, validates player-ID paths, sends keys only in headers, and suppresses raw transport/error bodies that might contain sensitive values. It makes one request and does not retry billable calls automatically.
- GitHub notification delivery depends on account preferences. Monitoring and paid hosting limitations are stated explicitly in the operator documentation.

## Source check

The live source review returned 133 entries: 114 GTD, 18 OUT, and 1 OFS. All observed statuses mapped to a documented code. Counts vary over time; fixtures and examples do not promise these production counts.

## Remaining proposals

Paid compute, durable injury history, webhooks, pricing changes, and additional client libraries remain separate proposals. The implementation does not claim source publication-time freshness or historical coverage.
