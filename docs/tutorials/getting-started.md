# Build an NBA team injury widget with status and freshness checks

Fetch the Golden State injury list, understand an empty result, and keep an old snapshot from looking like a new report. The starter makes one request.

## 1. Get your RapidAPI key

Open this API's Pricing tab and choose a plan. Basic offers five sample requests per month; use the first example before spending calls on optional follow-ups. Open the endpoint playground, select your RapidAPI application, and copy its **X-RapidAPI-Key** from the generated code snippet. The key belongs to your RapidAPI application; you do not need a provider secret or a separate key from APILabPH.

For this tutorial, the gateway host is `nba-injuries-api.p.rapidapi.com`. You can test `GET /v1/injuries` in the playground first. Each playground run and each script run is a separate API call. Check your plan page for current quotas and billing.

## 2. Run the example

Install Node.js 22 or later. No npm packages are needed. In a local working folder, create a file called `.env` with the line below, replacing the placeholder with your own key. Add `.env` to `.gitignore` before using Git. Keep the key on your server or local computer; never put it in public browser code, screenshots or a repository.

```text
RAPIDAPI_KEY=YOUR_RAPIDAPI_KEY
```

Save the following as `quickstart.mjs` in the same folder:

```javascript
const key = process.env.RAPIDAPI_KEY;
if (!key || key === 'YOUR_RAPIDAPI_KEY') {
  throw new Error('Set RAPIDAPI_KEY to the key from your RapidAPI application');
}
const host = 'nba-injuries-api.p.rapidapi.com';
const url = new URL("/v1/injuries?team=GSW", 'https://' + host);
const response = await fetch(url, {
  method: 'GET',
  headers: {
    'X-RapidAPI-Key': key,
    'X-RapidAPI-Host': host,
  },
  signal: AbortSignal.timeout(45000),
});
if (!response.ok) {
  let error = {};
  try { error = await response.json(); } catch {}
  const code = error?.error?.code || (typeof error?.error === 'string' ? error.error : null) || error?.message || 'REQUEST_FAILED';
  throw new Error(`HTTP ${response.status}: ${code}`);
}
const result = await response.json();
console.log(JSON.stringify(result, null, 2));
```

Run it from that folder:

```sh
node --env-file=.env quickstart.mjs
```

## 3. Read the result

The single-team result has `team`, `canonicalTeam`, `count` and `injuries`. Each injury entry includes the original `status`, normalized `statusCode`, `reason`, optional `playerId` and optional `returnDate`. Treat a return date as an estimate.

`updatedAt` is a Unix timestamp in seconds for successful snapshot retrieval. In JavaScript, use `new Date(result.updatedAt * 1000)` to display it. Inspect `cache.ageSeconds` and `cache.stale`; this is retrieval freshness, not the time ESPN changed the injury. `count: 0` means no entries were reported for that filter, not confirmed player availability.

## 4. Adapt it to your application

Add `status=OUT` to select only out designations. For several teams, replace `team=GSW` with `teams=GSW,NOP`; never send both `team` and `teams`. Multi-team responses use a `teams` object with `teamCount` and `playerEntryCount`, instead of the single-team `injuries` array.

Get supported codes from `/v1/teams`. Use `/v1/player/search?name=James` for injury-entry name searches, or `/v1/player/PLAYER_ID` using an ESPN ID returned by an injury record. These endpoints search injury entries, not the complete player roster. Each follow-up is another request.

## Limits and troubleshooting

This is an independent ESPN-sourced integration. Source delays, missing fields and slower cold starts can occur. `/v1/freshness` is a diagnostic endpoint; its 503 response still contains a freshness report, rather than the normal error shape. Never turn a source failure into a healthy-player indicator. A 400 `INVALID_FILTER` or `INVALID_STATUS` calls for corrected input, not retries.

For 401/403, check the key, host, selected application and subscription. For 429, check the plan quota and rate limit before retrying; a used-up monthly allowance will not be fixed by rapid retries. A local timeout does not prove the server never processed the request. Avoid automatic retry loops. Additional calls can consume your allowance.

If you need help, use this listing's Discussions tab with the endpoint, HTTP status, error code and request ID when supplied. Do not post API keys or private payloads. See the Overview for the complete endpoint contract.
