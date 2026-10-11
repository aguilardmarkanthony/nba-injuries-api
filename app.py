"""NBA Injuries API: compatible v1 data with validated filters and freshness metadata."""

import hmac
import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from injury_cache import CacheUnavailable, InjuryCache
from injury_feed import build_injury_map, load_data
from teams import CANONICAL_TO_SOURCE, TEAM_NAMES, canonical_team, directory

VERSION = "2.3.0"
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL", "120"))
MAX_CACHE_AGE_SECONDS = int(os.environ.get("MAX_CACHE_AGE", "900"))
RAPIDAPI_PROXY_SECRET = os.environ.get("RAPIDAPI_PROXY_SECRET", "").strip()
# Direct-origin keys are opt-in. Legacy DEMO_KEYS configuration remains supported,
# but the old publicly documented demo values are never accepted.
DIRECT_API_KEYS = {
    key.strip()
    for key in (os.environ.get("DIRECT_API_KEYS", "") + "," + os.environ.get("DEMO_KEYS", "")).split(",")
    if key.strip() and key.strip() not in {"free-demo-key", "pro-demo-key"}
}
direct_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def load_snapshot() -> dict:
    injuries = build_injury_map(load_data())
    return {
        "updatedAt": int(time.time()),
        "source": "ESPN",
        "teams": injuries,
        "teamCount": len(injuries),
        "playerEntryCount": sum(map(len, injuries.values())),
    }


cache = InjuryCache(load_snapshot, ttl=CACHE_TTL_SECONDS, max_age=MAX_CACHE_AGE_SECONDS)


@asynccontextmanager
async def lifespan(application: FastAPI):
    cache.start()
    try:
        yield
    finally:
        cache.stop()


app = FastAPI(
    title="NBA Injuries API",
    version=VERSION,
    lifespan=lifespan,
    description="Current ESPN-sourced NBA injury snapshots, team lookup, and player search. "
    "Responses include cache age and stale-data indicators. Return dates are source estimates.",
)


class Injury(BaseModel):
    player: str
    status: str = Field(
        description="Source status, commonly GTD (game-time decision), OUT, or OFS (out for season)."
    )
    reason: str
    returnDate: str | None = Field(
        description="Source-provided estimated return date; null when unavailable."
    )
    playerId: str | None = Field(
        default=None, description="ESPN athlete identifier, when supplied by the source."
    )


class CacheMetadata(BaseModel):
    ageSeconds: int
    ttlSeconds: int
    maxAgeSeconds: int
    stale: bool


class SnapshotMetadata(BaseModel):
    updatedAt: int = Field(
        description="Unix timestamp in seconds when this snapshot was successfully fetched and parsed; not the source's injury update time."
    )
    source: str
    cache: CacheMetadata


class AllInjuries(SnapshotMetadata):
    teams: dict[str, list[Injury]]
    teamCount: int = Field(
        description="Number of teams represented by injury entries in this snapshot; may be less than 30."
    )
    playerEntryCount: int


class TeamInjuries(SnapshotMetadata):
    team: str = Field(
        description="The requested abbreviation, trimmed and uppercased, retained for v1 compatibility."
    )
    canonicalTeam: str
    count: int
    injuries: list[Injury]


class TeamDetails(BaseModel):
    code: str
    name: str
    sourceCode: str
    aliases: list[str]


class TeamsResponse(BaseModel):
    teams: list[str] = Field(
        description="All 30 supported v1 source abbreviations, including teams without injury entries."
    )
    canonicalTeams: list[str]
    details: list[TeamDetails]


class SearchInjury(Injury):
    team: str
    canonicalTeam: str


class SearchResponse(SnapshotMetadata):
    query: str
    count: int
    results: list[SearchInjury]


class ErrorDetails(BaseModel):
    code: str
    message: str
    validTeams: list[str] | None = None
    details: list[dict] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetails


class HealthResponse(BaseModel):
    status: str
    version: str
    cacheTtlSeconds: int


ERROR_RESPONSES = {
    code: {"model": ErrorResponse, "description": description}
    for code, description in {
        400: "Invalid team or search value",
        401: "Authentication is missing or invalid",
        422: "Request parameter validation failed",
        503: "No sufficiently recent source snapshot is available",
    }.items()
}


def fail(status: int, code: str, message: str, **extra):
    raise HTTPException(status_code=status, detail={"code": code, "message": message, **extra})


def require_key(request: Request, direct_key: str | None = Depends(direct_key_header)):
    incoming = request.headers.get("X-RapidAPI-Proxy-Secret", "").strip()
    if RAPIDAPI_PROXY_SECRET and hmac.compare_digest(incoming.encode(), RAPIDAPI_PROXY_SECRET.encode()):
        return
    if direct_key and any(
        hmac.compare_digest(direct_key.strip().encode(), key.encode()) for key in DIRECT_API_KEYS
    ):
        return
    fail(
        401,
        "UNAUTHORIZED",
        "Use the RapidAPI gateway with X-RapidAPI-Key and X-RapidAPI-Host, or a configured direct-origin X-API-Key.",
    )


@app.exception_handler(HTTPException)
async def handle_http_error(request: Request, exc: HTTPException):
    error = exc.detail if isinstance(exc.detail, dict) else {"code": "HTTP_ERROR", "message": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content={"error": error}, headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    details = [
        {"location": list(error["loc"]), "message": error["msg"], "type": error["type"]}
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid request parameters.",
                "details": details,
            }
        },
    )


def get_snapshot() -> tuple[dict, dict]:
    try:
        return cache.get()
    except CacheUnavailable:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "UPSTREAM_UNAVAILABLE",
                "message": "Injury data is temporarily unavailable. Please retry later.",
            },
            headers={"Retry-After": str(cache.retry_seconds)},
        ) from None


def metadata(payload: dict, cache_info: dict) -> dict:
    return {"updatedAt": payload["updatedAt"], "source": payload["source"], "cache": cache_info}


@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "NBA Injuries API",
        "version": VERSION,
        "docs": "/docs",
        "health": "/v1/health",
        "marketplace": "https://rapidapi.com/aguilardmarkanthony/api/nba-injuries-api",
    }


@app.get(
    "/v1/health",
    response_model=HealthResponse,
    summary="Health Check",
    operation_id="healthCheck",
    tags=["Service"],
)
def health():
    """Liveness only: a healthy process does not guarantee fresh upstream data."""
    return {"status": "ok", "version": VERSION, "cacheTtlSeconds": CACHE_TTL_SECONDS}


@app.get("/v1/ready", include_in_schema=False)
def ready():
    payload, cache_info = get_snapshot()
    return {"status": "ready", **metadata(payload, cache_info)}


@app.get(
    "/v1/teams",
    response_model=TeamsResponse,
    dependencies=[Depends(require_key)],
    summary="List Teams",
    operation_id="listTeams",
    tags=["Teams"],
    responses=ERROR_RESPONSES,
)
def teams():
    """List all 30 supported teams and canonical/source abbreviation mappings. No source fetch is needed."""
    return {
        "teams": sorted(CANONICAL_TO_SOURCE.values()),
        "canonicalTeams": sorted(TEAM_NAMES),
        "details": directory(),
    }


@app.get(
    "/v1/injuries",
    response_model=AllInjuries | TeamInjuries,
    dependencies=[Depends(require_key)],
    summary="Get Injuries",
    operation_id="getInjuries",
    tags=["Injuries"],
    responses=ERROR_RESPONSES,
)
def injuries(
    team: str | None = Query(
        default=None,
        max_length=16,
        description="Optional NBA abbreviation, case-insensitive. Canonical and source aliases both work: GSW/GS, NOP/NO, NYK/NY, SAS/SA, UTA/UTAH, WAS/WSH. Omit for all teams.",
    ),
):
    canonical = None
    if team is not None:
        try:
            canonical = canonical_team(team)
        except ValueError:
            fail(
                400,
                "INVALID_TEAM",
                "Unknown NBA team abbreviation. See /v1/teams for supported aliases.",
                validTeams=sorted(TEAM_NAMES),
            )
    payload, cache_info = get_snapshot()
    if canonical is not None:
        rows = payload["teams"].get(CANONICAL_TO_SOURCE[canonical], [])
        return {
            "team": team.strip().upper(),
            "canonicalTeam": canonical,
            "count": len(rows),
            "injuries": rows,
            **metadata(payload, cache_info),
        }
    return {**payload, "cache": cache_info}


@app.get(
    "/v1/player/search",
    response_model=SearchResponse,
    dependencies=[Depends(require_key)],
    summary="Search Injured Players",
    operation_id="searchInjuredPlayers",
    tags=["Players"],
    responses=ERROR_RESPONSES,
)
def player_search(
    name: str = Query(
        ...,
        min_length=2,
        max_length=100,
        description="Case-insensitive substring of a player's name. Must contain at least two characters after trimming. Searches the current injury snapshot, not all NBA players.",
    ),
):
    query = name.strip()
    if len(query) < 2:
        fail(400, "INVALID_SEARCH", "Provide at least two characters after trimming whitespace.")
    payload, cache_info = get_snapshot()
    results = [
        {"team": team, "canonicalTeam": canonical_team(team), **injury}
        for team, rows in payload["teams"].items()
        for injury in rows
        if query.casefold() in injury["player"].casefold()
    ]
    return {"query": query, "count": len(results), "results": results, **metadata(payload, cache_info)}
