import os
import time
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse

from test_espn_injuries import load_data, build_injury_map

# ---------------- CONFIG ----------------
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL", "120"))

# Temporary browser testing switch:
# - 1 = allow ?api_key=... (use this while testing)
# - 0 = disable query param auth (recommended before marketplace launch)
ALLOW_QUERY_KEY = os.environ.get("ALLOW_QUERY_KEY", "1").strip() == "1"

# Demo keys for quick smoke tests (optional)
DEMO_KEYS = set(
    k.strip()
    for k in os.environ.get("DEMO_KEYS", "free-demo-key,pro-demo-key").split(",")
    if k.strip()
)

# ✅ RapidAPI-origin verification secret (copy from RapidAPI -> Gateway tab)
RAPIDAPI_PROXY_SECRET = os.environ.get("RAPIDAPI_PROXY_SECRET", "").strip()

app = FastAPI(title="Basketball Injuries API", version="2.2.0")


# ---------------- Root ----------------
@app.get("/")
def root():
    return {
        "message": "Basketball Injuries API is live",
        "docs": "/docs",
        "health": "/v1/health",
        "browserTestExample": "/v1/injuries?api_key=free-demo-key" if ALLOW_QUERY_KEY else "disabled",
        "note": "RapidAPI requests are validated via X-RapidAPI-Proxy-Secret.",
    }


# ---------------- Auth ----------------
def is_from_rapidapi(request: Request) -> bool:
    """
    RapidAPI forwards requests to your origin with X-RapidAPI-Proxy-Secret.
    Verify it matches the secret shown in RapidAPI Studio -> Gateway tab.
    """
    if not RAPIDAPI_PROXY_SECRET:
        return False
    incoming = (request.headers.get("X-RapidAPI-Proxy-Secret") or "").strip()
    return bool(incoming) and incoming == RAPIDAPI_PROXY_SECRET


def get_client_key(request: Request) -> Optional[str]:
    # Preferred: your own header (useful for direct testing / future custom auth)
    k = request.headers.get("X-API-Key")
    if k:
        return k.strip()

    # NOTE: RapidAPI Playground shows X-RapidAPI-Key in the UI,
    # but it is NOT guaranteed to be forwarded to your origin.
    k = request.headers.get("X-RapidAPI-Key")
    if k:
        return k.strip()

    # Browser testing: query parameter (only if enabled)
    if ALLOW_QUERY_KEY:
        k = request.query_params.get("api_key") or request.query_params.get("key")
        if k:
            return k.strip()

    return None


def require_key(request: Request) -> str:
    # ✅ If the request came through RapidAPI proxy, accept it.
    # Billing/auth is handled by RapidAPI subscription.
    if is_from_rapidapi(request):
        return "rapidapi"

    # Otherwise require your normal key logic (browser/local/direct usage)
    api_key = get_client_key(request)
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Send X-API-Key header, or use ?api_key=... if enabled.",
        )

    # Accept demo keys for browser testing
    if api_key in DEMO_KEYS:
        return api_key

    raise HTTPException(status_code=401, detail="Invalid API key")


# ---------------- Cache ----------------
_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None}


def get_injuries_cached(ttl: int = CACHE_TTL_SECONDS) -> Dict[str, Any]:
    now = time.time()
    if _CACHE["data"] is not None and (now - _CACHE["ts"]) < ttl:
        return _CACHE["data"]

    data = load_data(None)
    mp = build_injury_map(data)

    payload = {
        "updatedAt": int(now),
        "source": "ESPN",
        "teams": mp,
        "teamCount": len(mp),
        "playerEntryCount": sum(len(v) for v in mp.values()),
    }

    _CACHE["ts"] = now
    _CACHE["data"] = payload
    return payload


# ---------------- Routes ----------------
@app.get("/v1/health")
def health(request: Request):
    # health is usually public, but if you want to restrict it, uncomment:
    # require_key(request)

    return {
        "status": "ok",
        "cacheTtlSeconds": CACHE_TTL_SECONDS,
        "allowQueryKey": ALLOW_QUERY_KEY,
        "rapidapiProxySecretConfigured": bool(RAPIDAPI_PROXY_SECRET),
    }


@app.get("/v1/teams")
def teams(request: Request):
    require_key(request)
    payload = get_injuries_cached()
    return {"teams": sorted(list(payload.get("teams", {}).keys()))}


@app.get("/v1/injuries")
def injuries(
    request: Request,
    team: Optional[str] = Query(default=None, description="Team abbreviation like LAL, BOS"),
):
    require_key(request)
    payload = get_injuries_cached()

    if team:
        t = team.upper().strip()
        rows = payload["teams"].get(t, [])
        return JSONResponse(
            {
                "team": t,
                "count": len(rows),
                "injuries": rows,
                "updatedAt": payload["updatedAt"],
            }
        )

    return JSONResponse(payload)


@app.get("/v1/player/search")
def player_search(
    request: Request,
    name: str = Query(..., min_length=2),
):
    require_key(request)

    payload = get_injuries_cached()
    q = name.lower().strip()
    matches = []

    for team, items in payload.get("teams", {}).items():
        for it in items:
            player = str(it.get("player", "")).strip()
            if q in player.lower():
                matches.append({"team": team, **it})

    return {
        "query": name,
        "count": len(matches),
        "results": matches,
        "updatedAt": payload["updatedAt"],
    }