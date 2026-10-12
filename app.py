"""NBA Injuries API: compatible v1 data with validated filters and freshness metadata."""

import hmac
import os
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, computed_field

from injury_cache import CacheUnavailable, InjuryCache
from injury_feed import build_injury_map, load_data
from injury_status import STATUS_CODES, StatusCode, parse_status, status_code
from teams import CANONICAL_TO_SOURCE, TEAM_NAMES, canonical_team, directory

VERSION = "2.4.0"
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
FRESHNESS_ALERT_SECONDS = int(
    os.environ.get("FRESHNESS_ALERT_SECONDS", str(min(cache.max_age, max(cache.ttl, 300))))
)
if not cache.ttl <= FRESHNESS_ALERT_SECONDS <= cache.max_age:
    raise ValueError("FRESHNESS_ALERT_SECONDS must be between CACHE_TTL and MAX_CACHE_AGE")


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
    description="Current ESPN-sourced NBA injury snapshots with team/status filters, player lookup, and freshness checks. "
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

    @computed_field(
        description="Stable filter code derived from status; unrecognized source text maps to UNKNOWN."
    )
    @property
    def statusCode(self) -> StatusCode:
        return status_code(self.status)


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
        description="Number of teams represented by returned injury entries; reflects applied filters and may be less than 30."
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


class PlayerResponse(SnapshotMetadata):
    playerId: str
    count: int
    results: list[SearchInjury]


class ErrorDetails(BaseModel):
    code: str
    message: str
    validTeams: list[str] | None = None
    validStatuses: list[str] | None = None
    details: list[dict] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetails


class HealthResponse(BaseModel):
    status: str
    version: str
    cacheTtlSeconds: int


class FreshnessResponse(BaseModel):
    status: Literal["fresh", "delayed", "stale", "degraded", "unavailable"]
    version: str
    source: str
    updatedAt: int | None
    cache: CacheMetadata | None
    alertAfterSeconds: int
    consecutiveRefreshFailures: int


ERROR_RESPONSES = {
    code: {"model": ErrorResponse, "description": description}
    for code, description in {
        400: "Invalid team, status, filter combination, or search value",
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


@app.get(
    "/v1/freshness",
    response_model=FreshnessResponse,
    summary="Check Snapshot Freshness",
    operation_id="checkFreshness",
    tags=["Service"],
    responses={
        503: {
            "model": FreshnessResponse,
            "description": "Snapshot missing, too old, or repeatedly failing to refresh",
        }
    },
)
def freshness(response: Response):
    """Observe background refresh health without fetching. HTTP 503 uses the same diagnostic schema as HTTP 200."""
    observed = cache.inspect()
    info = observed["cache"]
    if info is None:
        state = "unavailable"
    elif info["ageSeconds"] >= FRESHNESS_ALERT_SECONDS:
        state = "stale"
    elif observed["consecutiveRefreshFailures"] >= 3:
        state = "degraded"
    else:
        state = "delayed" if info["stale"] else "fresh"
    response.headers["Cache-Control"] = "no-store"
    if state in {"unavailable", "stale", "degraded"}:
        response.status_code = 503
        response.headers["Retry-After"] = str(cache.retry_seconds)
    return {
        "status": state,
        "version": VERSION,
        "source": "ESPN",
        "alertAfterSeconds": FRESHNESS_ALERT_SECONDS,
        **observed,
    }


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
        description="Optional single NBA abbreviation, case-insensitive. Canonical and source aliases work: GSW/GS, NOP/NO, NYK/NY, SAS/SA, UTA/UTAH, WAS/WSH. Cannot combine with teams.",
    ),
    status: str | None = Query(
        default=None,
        max_length=64,
        description="Optional status code or readable alias, case-insensitive: OUT, GTD, OFS, QUESTIONABLE, DOUBTFUL, PROBABLE, DAY_TO_DAY, UNKNOWN. GTD and QUESTIONABLE are distinct. Omit for every status.",
    ),
    teams: str | None = Query(
        default=None,
        max_length=256,
        description="Optional comma-separated list of up to 30 team abbreviations, e.g. GSW,NOP. Aliases are deduplicated. Returns the all-team envelope with matching source keys. Cannot combine with team.",
    ),
):
    if team is not None and teams is not None:
        fail(400, "INVALID_FILTER", "Use either team or teams, not both.")
    selected_status = None
    if status is not None:
        try:
            selected_status = parse_status(status)
        except ValueError:
            fail(
                400,
                "INVALID_STATUS",
                "Unsupported injury status. See validStatuses.",
                validStatuses=STATUS_CODES,
            )
    canonical = None
    selected_teams = None
    if team is not None or teams is not None:
        try:
            if team is not None:
                canonical = canonical_team(team)
            else:
                values = teams.split(",")
                if len(values) > 30:
                    raise ValueError("Too many teams")
                selected_teams = {CANONICAL_TO_SOURCE[canonical_team(value)] for value in values}
        except ValueError:
            fail(
                400,
                "INVALID_TEAM",
                "Use valid NBA abbreviations; teams accepts 1-30 comma-separated values without empty items. See /v1/teams.",
                validTeams=sorted(TEAM_NAMES),
            )
    payload, cache_info = get_snapshot()

    def matches(row):
        return selected_status is None or status_code(row["status"]) == selected_status

    if canonical is not None:
        rows = [row for row in payload["teams"].get(CANONICAL_TO_SOURCE[canonical], []) if matches(row)]
        return {
            "team": team.strip().upper(),
            "canonicalTeam": canonical,
            "count": len(rows),
            "injuries": rows,
            **metadata(payload, cache_info),
        }
    if selected_status is None and selected_teams is None:
        return {**payload, "cache": cache_info}
    filtered = {
        source: matching
        for source, rows in payload["teams"].items()
        if selected_teams is None or source in selected_teams
        if (matching := [row for row in rows if matches(row)])
    }
    return {
        **metadata(payload, cache_info),
        "teams": filtered,
        "teamCount": len(filtered),
        "playerEntryCount": sum(map(len, filtered.values())),
    }


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


@app.get(
    "/v1/player/{player_id}",
    response_model=PlayerResponse,
    dependencies=[Depends(require_key)],
    summary="Find Injuries by Player ID",
    operation_id="getPlayerInjuries",
    tags=["Players"],
    responses={code: response for code, response in ERROR_RESPONSES.items() if code != 400},
)
def player_lookup(
    player_id: str = Path(
        ...,
        min_length=1,
        max_length=12,
        pattern=r"^[1-9][0-9]*$",
        description="Positive ESPN athlete ID, 1-12 ASCII digits with no leading zero. Searches current injury entries only.",
    ),
):
    """An empty 200 means no matching injury entry, not that the player exists or is healthy."""
    payload, cache_info = get_snapshot()
    results = [
        {"team": team, "canonicalTeam": canonical_team(team), **injury}
        for team, rows in payload["teams"].items()
        for injury in rows
        if injury.get("playerId") == player_id
    ]
    return {"playerId": player_id, "count": len(results), "results": results, **metadata(payload, cache_info)}
