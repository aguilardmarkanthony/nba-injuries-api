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
DEMO_KEYS = set(k.strip() for k in os.environ.get("DEMO_KEYS", "free-demo-key,pro-demo-key").split(",") if k.strip())

app = FastAPI(title="Basketball Injuries API", version="2.1.0")

@app.get("/")
def root():
    return {
        "message": "Basketball Injuries API is live",
        "docs": "/docs",
        "health": "/v1/health",
        "example": "/v1/injuries?api_key=free-demo-key",
    }

# ---------------- Auth ----------------
def get_client_key(request: Request) -> Optional[str]:
    # Preferred: your own header
    k = request.headers.get("X-API-Key")
    if k:
        return k.strip()

    # RapidAPI header (when testing through RapidAPI Console later)
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
    api_key = get_client_key(request)
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Send X-API-Key header (or X-RapidAPI-Key), or use ?api_key=... if enabled.",
        )

    # For now: accept demo keys (so you can test in browser immediately).
    # Later: replace this with your DB/subscription validation.
    if api_key in DEMO_KEYS:
        return api_key

    # If you want to allow any RapidAPI key without validating here, you can uncomment this:
    # if request.headers.get("X-RapidAPI-Key"):
    #     return api_key

    # Otherwise, block unknown keys by default:
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
@app.get("/")
def root():
    return {
        "message": "Basketball Injuries API is live",
        "docs": "/docs",
        "health": "/v1/health",
        "browserTestExample": "/v1/injuries?api_key=free-demo-key" if ALLOW_QUERY_KEY else "disabled",
    }

@app.get("/v1/health")
def health():
    return {"status": "ok", "cacheTtlSeconds": CACHE_TTL_SECONDS, "allowQueryKey": ALLOW_QUERY_KEY}


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