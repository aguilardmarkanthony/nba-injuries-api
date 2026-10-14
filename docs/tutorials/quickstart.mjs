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
