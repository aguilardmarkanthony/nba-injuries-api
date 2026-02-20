# app.py  (Phase 1 + Phase 3: SQLite keys + history + webhooks + changes + utility endpoints)
import os
import time
import math
import json
import hashlib
import sqlite3
import secrets
from typing import Optional, Dict, Any, Tuple, List

import hmac
import hashlib
import json
import secrets
import sqlite3
from fastapi import Request

import httpx
from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.responses import JSONResponse

from test_espn_injuries import load_data, build_injury_map

# ---------------- CONFIG ----------------
# ---------------- CONFIG ----------------
LEMON_WEBHOOK_SECRET = (os.environ.get("LEMON_WEBHOOK_SECRET") or "").strip()

if not LEMON_WEBHOOK_SECRET:
    raise RuntimeError("LEMON_WEBHOOK_SECRET is not set for this process")
DB_PATH = os.environ.get("DB_PATH", "nba_injuries.db")
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL", "120"))
MAX_SNAPSHOTS = int(os.environ.get("MAX_SNAPSHOTS", "50"))
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "change-me-admin-secret")

# webhook settings
WEBHOOK_TIMEOUT_SECONDS = float(os.environ.get("WEBHOOK_TIMEOUT_SECONDS", "5"))
WEBHOOK_MAX_ATTEMPTS = int(os.environ.get("WEBHOOK_MAX_ATTEMPTS", "2"))

PLAN_LIMITS = {
    "free": {"rpm": 30, "rpd": 500, "full_payload": False, "changes": False, "webhooks": False, "history": False},
    "pro":  {"rpm": 600, "rpd": 50000, "full_payload": True, "changes": True, "webhooks": True, "history": True},
}

app = FastAPI(title="Basketball Injuries API", version="1.3.0")
import base64
import binascii
import hmac
import hashlib

def verify_lemon_signature(raw_body: bytes, signature: str) -> bool:
    if not LEMON_WEBHOOK_SECRET:
        return False

    sig = (signature or "").strip()

    # Common pattern: "sha256=<hex>"
    if sig.lower().startswith("sha256="):
        sig = sig.split("=", 1)[1].strip()

    mac = hmac.new(
        LEMON_WEBHOOK_SECRET.encode("utf-8"),
        msg=raw_body,
        digestmod=hashlib.sha256
    )

    expected_hex = mac.hexdigest()
    expected_b64 = base64.b64encode(mac.digest()).decode("utf-8")

    # Compare against both possible encodings
    if hmac.compare_digest(expected_hex, sig):
        return True
    if hmac.compare_digest(expected_b64, sig):
        return True

    # Some providers send base64 but with whitespace/newlines; normalize
    try:
        # If sig is base64 of raw digest bytes, compare bytes too
        sig_bytes = base64.b64decode(sig, validate=True)
        return hmac.compare_digest(mac.digest(), sig_bytes)
    except Exception:
        return False



# ---------------- SQLITE ----------------
def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def db_init() -> None:
    conn = db_connect()
    try:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            api_key     TEXT PRIMARY KEY,
            plan        TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'active',
            created_at  INTEGER NOT NULL,
            note        TEXT
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS usage_daily (
            api_key   TEXT NOT NULL,
            day_id    INTEGER NOT NULL,
            count     INTEGER NOT NULL,
            PRIMARY KEY (api_key, day_id)
        )
        """)

        # Phase 3: history snapshots
        cur.execute("""
        CREATE TABLE IF NOT EXISTS injury_snapshots (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ts         INTEGER NOT NULL,
            payload    TEXT NOT NULL
        )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_injury_snapshots_ts ON injury_snapshots(ts)")

        # Phase 3: webhooks registry
        cur.execute("""
        CREATE TABLE IF NOT EXISTS webhooks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            api_key     TEXT NOT NULL,
            url         TEXT NOT NULL,
            events_json TEXT NOT NULL,
            created_at  INTEGER NOT NULL,
            is_active   INTEGER NOT NULL DEFAULT 1
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS lemon_subscriptions (
        subscription_id TEXT PRIMARY KEY,
        order_id TEXT,
        customer_email TEXT,
        api_key TEXT,
        plan TEXT NOT NULL,
        status TEXT NOT NULL,
        updated_at INTEGER NOT NULL
        )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_webhooks_api_key ON webhooks(api_key)")
        conn.commit()
    finally:
        conn.close()


@app.on_event("startup")
def _startup():
    db_init()
    seed_demo_keys()


def seed_demo_keys() -> None:
    now = int(time.time())
    conn = db_connect()
    try:
        cur = conn.cursor()
        for k, plan in [("free-demo-key", "free"), ("pro-demo-key", "pro")]:
            cur.execute("SELECT api_key FROM api_keys WHERE api_key=?", (k,))
            if cur.fetchone() is None:
                cur.execute(
                    "INSERT INTO api_keys(api_key, plan, status, created_at, note) VALUES(?,?,?,?,?)",
                    (k, plan, "active", now, "seeded demo key"),
                )
        conn.commit()
    finally:
        conn.close()


def get_key_record(api_key: str) -> Optional[sqlite3.Row]:
    conn = db_connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT api_key, plan, status, created_at, note FROM api_keys WHERE api_key=?", (api_key,))
        return cur.fetchone()
    finally:
        conn.close()


def create_key(plan: str, note: str = "") -> str:
    if plan not in PLAN_LIMITS:
        raise ValueError("Invalid plan")
    api_key = secrets.token_urlsafe(32)
    now = int(time.time())
    conn = db_connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO api_keys(api_key, plan, status, created_at, note) VALUES(?,?,?,?,?)",
            (api_key, plan, "active", now, note),
        )
        conn.commit()
    finally:
        conn.close()
    return api_key


def revoke_key(api_key: str) -> bool:
    conn = db_connect()
    try:
        cur = conn.cursor()
        cur.execute("UPDATE api_keys SET status='revoked' WHERE api_key=?", (api_key,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def list_keys(limit: int = 200) -> List[Dict[str, Any]]:
    conn = db_connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT api_key, plan, status, created_at, note FROM api_keys ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ---------------- LIMITING ----------------
def day_id_utc(now: float) -> int:
    return int(now // 86400)

def day_reset_epoch_utc(now: float) -> int:
    return int((math.floor(now / 86400) + 1) * 86400)

def minute_reset_epoch(now: float) -> int:
    return int((math.floor(now / 60) + 1) * 60)

_MIN_BUCKET: Dict[str, Dict[str, Any]] = {}

def require_api_key(x_api_key: Optional[str]) -> Tuple[str, str]:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key")
    rec = get_key_record(x_api_key)
    if rec is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if rec["status"] != "active":
        raise HTTPException(status_code=403, detail="API key revoked")
    plan = rec["plan"]
    if plan not in PLAN_LIMITS:
        raise HTTPException(status_code=403, detail="Invalid plan on key")
    return x_api_key, plan

def get_daily_usage(api_key: str, day: int) -> int:
    conn = db_connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT count FROM usage_daily WHERE api_key=? AND day_id=?", (api_key, day))
        row = cur.fetchone()
        return int(row["count"]) if row else 0
    finally:
        conn.close()

def increment_daily_usage(api_key: str, day: int) -> int:
    conn = db_connect()
    try:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO usage_daily(api_key, day_id, count)
        VALUES(?, ?, 1)
        ON CONFLICT(api_key, day_id) DO UPDATE SET count = count + 1
        """, (api_key, day))
        conn.commit()
        cur.execute("SELECT count FROM usage_daily WHERE api_key=? AND day_id=?", (api_key, day))
        row = cur.fetchone()
        return int(row["count"]) if row else 1
    finally:
        conn.close()

def enforce_limits(api_key: str, plan: str) -> Dict[str, str]:
    limits = PLAN_LIMITS[plan]
    rpm, rpd = int(limits["rpm"]), int(limits["rpd"])
    now = time.time()

    bucket = _MIN_BUCKET.get(api_key)
    if bucket is None:
        bucket = {"tokens": float(rpm), "last": now}
        _MIN_BUCKET[api_key] = bucket

    elapsed = now - float(bucket["last"])
    bucket["last"] = now
    refill = (elapsed / 60.0) * float(rpm)
    bucket["tokens"] = min(float(rpm), float(bucket["tokens"]) + refill)

    if bucket["tokens"] < 1.0:
        reset = minute_reset_epoch(now)
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded (per-minute).",
            headers={"Retry-After": str(max(1, reset - int(now))), "X-RateLimit-Reset-Minute": str(reset)},
        )

    bucket["tokens"] -= 1.0
    remaining_min = int(bucket["tokens"])

    d = day_id_utc(now)
    used = get_daily_usage(api_key, d)
    if used >= rpd:
        reset_day = day_reset_epoch_utc(now)
        raise HTTPException(
            status_code=429,
            detail="Daily quota exceeded.",
            headers={"Retry-After": str(max(60, reset_day - int(now))), "X-RateLimit-Reset-Day": str(reset_day)},
        )

    new_count = increment_daily_usage(api_key, d)
    remaining_day = max(0, rpd - new_count)

    return {
        "X-Plan": plan,
        "X-RateLimit-Limit-Minute": str(rpm),
        "X-RateLimit-Remaining-Minute": str(remaining_min),
        "X-RateLimit-Reset-Minute": str(minute_reset_epoch(now)),
        "X-RateLimit-Limit-Day": str(rpd),
        "X-RateLimit-Remaining-Day": str(remaining_day),
        "X-RateLimit-Reset-Day": str(day_reset_epoch_utc(now)),
    }


# ---------------- ESPN CACHE ----------------
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


# ---------------- SNAPSHOT MEMORY (for /changes) ----------------
_SNAPSHOTS: List[Dict[str, Any]] = []

def _stable_injury_key(item: Dict[str, Any]) -> str:
    for k in ("athlete_id", "playerId", "id"):
        if k in item and item[k]:
            return str(item[k])
    name = str(item.get("player", item.get("playerName", item.get("name", "")))).strip()
    reason = str(item.get("reason", item.get("type", ""))).strip()
    status = str(item.get("status", item.get("injuryStatus", ""))).strip()
    base = f"{name}|{reason}|{status}".strip("|")
    if base:
        return base
    return hashlib.sha1(json.dumps(item, sort_keys=True).encode("utf-8")).hexdigest()

def _normalize_snapshot(payload: Dict[str, Any]) -> Dict[str, Any]:
    teams = payload.get("teams", {}) or {}
    snap = {"ts": int(payload.get("updatedAt", int(time.time()))), "teams": teams}
    snap_str = json.dumps(snap, sort_keys=True, default=str)
    snap["hash"] = hashlib.sha1(snap_str.encode("utf-8")).hexdigest()
    return snap

def _remember_snapshot(payload: Dict[str, Any]) -> None:
    snap = _normalize_snapshot(payload)
    if _SNAPSHOTS and _SNAPSHOTS[-1]["hash"] == snap["hash"]:
        return
    _SNAPSHOTS.append(snap)
    if len(_SNAPSHOTS) > MAX_SNAPSHOTS:
        del _SNAPSHOTS[0: len(_SNAPSHOTS) - MAX_SNAPSHOTS]

def _find_snapshot_at_or_before(ts: int) -> Optional[Dict[str, Any]]:
    candidates = [s for s in _SNAPSHOTS if int(s["ts"]) <= int(ts)]
    if not candidates:
        return None
    candidates.sort(key=lambda x: int(x["ts"]))
    return candidates[-1]

def _diff_snapshots(old_snap: Dict[str, Any], new_snap: Dict[str, Any]) -> Dict[str, Any]:
    def indexify(snap: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        out: Dict[str, Dict[str, Dict[str, Any]]] = {}
        teams = snap.get("teams", {}) or {}
        for team, items in teams.items():
            if not isinstance(items, list):
                continue
            m: Dict[str, Dict[str, Any]] = {}
            for it in items:
                if isinstance(it, dict):
                    m[_stable_injury_key(it)] = it
            out[str(team)] = m
        return out

    old_idx = indexify(old_snap)
    new_idx = indexify(new_snap)

    added, removed, updated = [], [], []
    all_teams = set(old_idx.keys()) | set(new_idx.keys())

    for team in sorted(all_teams):
        o = old_idx.get(team, {})
        n = new_idx.get(team, {})
        o_keys, n_keys = set(o.keys()), set(n.keys())

        for k in (n_keys - o_keys):
            added.append({"team": team, "key": k, "item": n[k]})
        for k in (o_keys - n_keys):
            removed.append({"team": team, "key": k, "item": o[k]})
        for k in (o_keys & n_keys):
            oi, ni = o[k], n[k]
            o_status = str(oi.get("status", oi.get("injuryStatus", ""))).strip()
            n_status = str(ni.get("status", ni.get("injuryStatus", ""))).strip()
            o_reason = str(oi.get("reason", oi.get("type", ""))).strip()
            n_reason = str(ni.get("reason", ni.get("type", ""))).strip()
            if o_status != n_status or o_reason != n_reason:
                updated.append({
                    "team": team,
                    "key": k,
                    "before": oi,
                    "after": ni,
                    "changes": {
                        "status": {"from": o_status, "to": n_status} if o_status != n_status else None,
                        "reason": {"from": o_reason, "to": n_reason} if o_reason != n_reason else None,
                    }
                })

    return {"added": added, "removed": removed, "updated": updated}


# ---------------- PHASE 3: HISTORY ----------------
def store_snapshot(payload: Dict[str, Any]) -> None:
    conn = db_connect()
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO injury_snapshots(ts, payload) VALUES(?, ?)", (int(payload["updatedAt"]), json.dumps(payload)))
        conn.commit()
    finally:
        conn.close()


def get_history(from_ts: int, to_ts: int) -> List[Dict[str, Any]]:
    conn = db_connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT ts, payload FROM injury_snapshots WHERE ts BETWEEN ? AND ? ORDER BY ts ASC",
            (int(from_ts), int(to_ts)),
        )
        rows = cur.fetchall()
        out = []
        for r in rows:
            out.append({"ts": int(r["ts"]), "payload": json.loads(r["payload"])})
        return out
    finally:
        conn.close()


# ---------------- PHASE 3: WEBHOOKS ----------------
def register_webhook(api_key: str, url: str, events: List[str]) -> int:
    now = int(time.time())
    conn = db_connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO webhooks(api_key, url, events_json, created_at, is_active) VALUES(?,?,?,?,1)",
            (api_key, url, json.dumps(events), now),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def list_webhooks(api_key: str) -> List[Dict[str, Any]]:
    conn = db_connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, url, events_json, created_at, is_active FROM webhooks WHERE api_key=? ORDER BY created_at DESC",
            (api_key,),
        )
        rows = cur.fetchall()
        out = []
        for r in rows:
            out.append({
                "id": int(r["id"]),
                "url": r["url"],
                "events": json.loads(r["events_json"]),
                "created_at": int(r["created_at"]),
                "is_active": bool(r["is_active"]),
            })
        return out
    finally:
        conn.close()


def deactivate_webhook(api_key: str, webhook_id: int) -> bool:
    conn = db_connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE webhooks SET is_active=0 WHERE api_key=? AND id=?",
            (api_key, int(webhook_id)),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_active_webhooks_for_key(api_key: str) -> List[Dict[str, Any]]:
    conn = db_connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, url, events_json FROM webhooks WHERE api_key=? AND is_active=1",
            (api_key,),
        )
        rows = cur.fetchall()
        out = []
        for r in rows:
            out.append({"id": int(r["id"]), "url": r["url"], "events": json.loads(r["events_json"])})
        return out
    finally:
        conn.close()


def _should_send(events: List[str], event_type: str) -> bool:
    return ("*" in events) or (event_type in events)


def deliver_webhooks(api_key: str, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    hooks = get_active_webhooks_for_key(api_key)
    if not hooks:
        return {"sent": 0, "failed": 0}

    sent = 0
    failed = 0

    for h in hooks:
        if not _should_send(h["events"], event_type):
            continue

        ok = False
        for attempt in range(WEBHOOK_MAX_ATTEMPTS):
            try:
                with httpx.Client(timeout=WEBHOOK_TIMEOUT_SECONDS) as client:
                    r = client.post(h["url"], json=payload)
                if 200 <= r.status_code < 300:
                    ok = True
                    break
            except Exception:
                pass

        if ok:
            sent += 1
        else:
            failed += 1

    return {"sent": sent, "failed": failed}


# ---------------- ROUTES ----------------
@app.get("/v1/health")
def health():
    return {"status": "ok", "db": DB_PATH, "snapshots": len(_SNAPSHOTS), "cacheTtlSeconds": CACHE_TTL_SECONDS}


@app.get("/v1/teams")
def teams():
    # static list derived from current snapshot
    payload = get_injuries_cached()
    return {"teams": sorted(list(payload.get("teams", {}).keys()))}


@app.get("/v1/player/search")
def player_search(
    name: str = Query(..., min_length=2, description="Substring match. Example: tatum"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    api_key, plan = require_api_key(x_api_key)
    headers = enforce_limits(api_key, plan)

    payload = get_injuries_cached()
    q = name.lower().strip()

    matches = []
    for team, items in payload.get("teams", {}).items():
        for it in items:
            player = str(it.get("player", "")).strip()
            if q in player.lower():
                matches.append({"team": team, **it})

    return JSONResponse({"query": name, "count": len(matches), "results": matches, "updatedAt": payload["updatedAt"]}, headers=headers)


@app.get("/v1/me")
def me(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    api_key, plan = require_api_key(x_api_key)
    limits = PLAN_LIMITS[plan]
    now = time.time()
    d = day_id_utc(now)
    used = get_daily_usage(api_key, d)
    return {"plan": plan, "limits": {"rpm": limits["rpm"], "rpd": limits["rpd"]}, "usageToday": used, "dayResetsAt": day_reset_epoch_utc(now)}


@app.get("/v1/injuries")
def injuries(
    team: Optional[str] = Query(default=None, description="Team abbreviation like MEM, LAL"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    api_key, plan = require_api_key(x_api_key)
    headers = enforce_limits(api_key, plan)

    payload = get_injuries_cached()
    _remember_snapshot(payload)
    store_snapshot(payload)

    # If snapshot changed: deliver webhooks (pro only)
    # We'll treat any updated snapshot as "injury_update" event
    if PLAN_LIMITS[plan]["webhooks"] and len(_SNAPSHOTS) >= 2:
        # compare last two
        diff = _diff_snapshots(_SNAPSHOTS[-2], _SNAPSHOTS[-1])
        if diff["added"] or diff["removed"] or diff["updated"]:
            deliver_webhooks(api_key, "injury_update", {"updatedAt": payload["updatedAt"], "changes": diff})

    if team:
        t = team.upper().strip()
        rows = payload["teams"].get(t, [])
        return JSONResponse({"team": t, "count": len(rows), "injuries": rows, "updatedAt": payload["updatedAt"], "plan": plan}, headers=headers)

    if not PLAN_LIMITS[plan]["full_payload"]:
        counts = {abbr: len(rows) for abbr, rows in payload["teams"].items()}
        return JSONResponse({"updatedAt": payload["updatedAt"], "counts": counts, "plan": plan}, headers=headers)

    full = payload.copy()
    full["plan"] = plan
    return JSONResponse(full, headers=headers)


@app.get("/v1/changes")
def changes(
    since: int = Query(..., description="Unix timestamp. Return changes since this time."),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    api_key, plan = require_api_key(x_api_key)
    headers = enforce_limits(api_key, plan)

    if not PLAN_LIMITS[plan]["changes"]:
        raise HTTPException(status_code=403, detail="Upgrade to Pro to access /v1/changes")

    payload = get_injuries_cached()
    _remember_snapshot(payload)
    store_snapshot(payload)

    new_snap = _SNAPSHOTS[-1] if _SNAPSHOTS else _normalize_snapshot(payload)
    old_snap = _find_snapshot_at_or_before(since)

    if old_snap is None:
        return JSONResponse(
            {"since": since, "availableFrom": _SNAPSHOTS[0]["ts"] if _SNAPSHOTS else None, "updatedAt": new_snap["ts"],
             "changes": {"added": [], "removed": [], "updated": []},
             "note": "No snapshot found at/before 'since'. Call /v1/injuries first, then retry with an older timestamp."},
            headers=headers,
        )

    diff = _diff_snapshots(old_snap, new_snap)
    return JSONResponse({"since": int(old_snap["ts"]), "updatedAt": int(new_snap["ts"]), "changes": diff}, headers=headers)


@app.get("/v1/history")
def history(
    team: Optional[str] = Query(default=None, description="Optional team filter"),
    from_ts: int = Query(..., description="Unix timestamp from"),
    to_ts: int = Query(..., description="Unix timestamp to"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    api_key, plan = require_api_key(x_api_key)
    headers = enforce_limits(api_key, plan)

    if not PLAN_LIMITS[plan]["history"]:
        raise HTTPException(status_code=403, detail="Upgrade to Pro to access /v1/history")

    rows = get_history(from_ts, to_ts)

    # optional team filter (extract only that team's injuries per snapshot)
    if team:
        t = team.upper().strip()
        filtered = []
        for r in rows:
            p = r["payload"]
            filtered.append({"ts": r["ts"], "team": t, "injuries": p.get("teams", {}).get(t, [])})
        return JSONResponse({"from": from_ts, "to": to_ts, "team": t, "snapshots": filtered}, headers=headers)

    return JSONResponse({"from": from_ts, "to": to_ts, "snapshots": rows}, headers=headers)


# ---------------- WEBHOOK ROUTES ----------------



@app.post("/v1/webhooks/register")
def webhook_register(
    url: str = Query(..., description="Your webhook URL (must be publicly reachable to receive events)"),
    events: str = Query("injury_update", description="Comma-separated events (injury_update or *)"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    api_key, plan = require_api_key(x_api_key)
    headers = enforce_limits(api_key, plan)

    if not PLAN_LIMITS[plan]["webhooks"]:
        raise HTTPException(status_code=403, detail="Upgrade to Pro to use webhooks")

    ev = [e.strip() for e in events.split(",") if e.strip()]
    wid = register_webhook(api_key, url, ev)
    return JSONResponse({"webhook_id": wid, "url": url, "events": ev}, headers=headers)


@app.get("/v1/webhooks/list")
def webhook_list(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    api_key, plan = require_api_key(x_api_key)
    headers = enforce_limits(api_key, plan)
    if not PLAN_LIMITS[plan]["webhooks"]:
        raise HTTPException(status_code=403, detail="Upgrade to Pro to use webhooks")
    return JSONResponse({"webhooks": list_webhooks(api_key)}, headers=headers)


@app.delete("/v1/webhooks/unregister")
def webhook_unregister(
    id: int = Query(..., description="Webhook ID to deactivate"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    api_key, plan = require_api_key(x_api_key)
    headers = enforce_limits(api_key, plan)

    if not PLAN_LIMITS[plan]["webhooks"]:
        raise HTTPException(status_code=403, detail="Upgrade to Pro to use webhooks")

    ok = deactivate_webhook(api_key, id)
    if not ok:
        raise HTTPException(status_code=404, detail="Webhook not found")
    return JSONResponse({"deactivated": True, "id": id}, headers=headers)

@app.post("/v1/webhooks/test")
def webhook_test(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    api_key, plan = require_api_key(x_api_key)
    headers = enforce_limits(api_key, plan)

    if not PLAN_LIMITS[plan]["webhooks"]:
        raise HTTPException(status_code=403, detail="Upgrade to Pro to use webhooks")

    # send a guaranteed test event to all active webhooks for this API key
    result = deliver_webhooks(
        api_key,
        "injury_update",
        {
            "test": True,
            "sentAt": int(time.time()),
            "message": "Webhook delivery test",
            "example": {
                "player": "Test Player",
                "team": "LAL",
                "status": "OUT",
                "reason": "Testing webhook delivery",
            },
        },
    )

    return JSONResponse({"ok": True, "result": result}, headers=headers)
# ---------------- ADMIN ----------------
def require_admin(x_admin_secret: Optional[str]) -> None:
    if not x_admin_secret or x_admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Invalid admin secret")

@app.post("/v1/admin/create_key")
def admin_create_key(
    plan: str = Query(..., description="free or pro"),
    note: str = Query("", description="Optional note"),
    x_admin_secret: Optional[str] = Header(default=None, alias="X-Admin-Secret"),
):
    require_admin(x_admin_secret)
    plan = plan.lower().strip()
    if plan not in PLAN_LIMITS:
        raise HTTPException(status_code=400, detail="Invalid plan")
    api_key = create_key(plan=plan, note=note)
    return {"api_key": api_key, "plan": plan, "status": "active"}

@app.post("/v1/admin/revoke_key")
def admin_revoke_key(
    api_key: str = Query(..., description="API key to revoke"),
    x_admin_secret: Optional[str] = Header(default=None, alias="X-Admin-Secret"),
):
    require_admin(x_admin_secret)
    ok = revoke_key(api_key)
    if not ok:
        raise HTTPException(status_code=404, detail="Key not found")
    return {"revoked": True}

@app.get("/v1/admin/list_keys")
def admin_list_keys(
    limit: int = Query(200, ge=1, le=1000),
    x_admin_secret: Optional[str] = Header(default=None, alias="X-Admin-Secret"),
):
    require_admin(x_admin_secret)
    return {"keys": list_keys(limit=limit)}

@app.post("/v1/billing/webhook")
async def lemon_webhook(
    request: Request,
    x_signature: Optional[str] = Header(default=None, alias="X-Signature"),
):
    raw = await request.body()

    # SAFE debug (no secret contents)
    print("Secret length:", len(LEMON_WEBHOOK_SECRET))
    print("Secret endswith \\n:", LEMON_WEBHOOK_SECRET.endswith("\n"))
    print("Body bytes:", len(raw))
    print("X-Signature:", x_signature)

    mac = hmac.new(LEMON_WEBHOOK_SECRET.encode("utf-8"), msg=raw, digestmod=hashlib.sha256)
    print("Computed  :", mac.hexdigest())

    if not x_signature or not verify_lemon_signature(raw, x_signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    payload = json.loads(raw.decode("utf-8"))
    event_name = payload.get("meta", {}).get("event_name") or payload.get("event_name")
    data = payload.get("data", {})

    # Extract common fields (Lemon payloads differ slightly per event)
    sub_id = str(data.get("id") or "")
    attrs = data.get("attributes", {}) or {}
    status = (attrs.get("status") or "").lower()
    customer_email = (attrs.get("user_email") or attrs.get("customer_email") or "").strip().lower()
    order_id = str(attrs.get("order_id") or "")

    # Decide plan by your product/variant — for Phase 1, everything = pro
    plan = "pro"

    now = int(time.time())
    conn = db_connect()
    cur = conn.cursor()

    # On paid/active subscription events, ensure an API key exists
    if event_name in {"subscription_created", "subscription_updated", "order_created"}:
        # Mark active if it looks active
        effective_status = "active" if status in {"active", "trialing", "on_trial"} or event_name == "order_created" else (status or "active")

        # Find existing mapping
        row = cur.execute("SELECT api_key FROM lemon_subscriptions WHERE subscription_id=?", (sub_id,)).fetchone()
        api_key = row["api_key"] if row else None

        if not api_key:
            api_key = create_key(plan=plan, note=f"lemon:{sub_id}:{customer_email}")

        cur.execute("""
          INSERT INTO lemon_subscriptions(subscription_id, order_id, customer_email, api_key, plan, status, updated_at)
          VALUES (?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT(subscription_id) DO UPDATE SET
            order_id=excluded.order_id,
            customer_email=excluded.customer_email,
            api_key=excluded.api_key,
            plan=excluded.plan,
            status=excluded.status,
            updated_at=excluded.updated_at
        """, (sub_id, order_id, customer_email, api_key, plan, effective_status, now))

        # Ensure key active
        cur.execute("UPDATE api_keys SET status='active' WHERE api_key=?", (api_key,))

        conn.commit()
        conn.close()
        return {"ok": True}

    # On cancel/expire, deactivate key
    if event_name in {"subscription_cancelled", "subscription_expired"}:
        row = cur.execute("SELECT api_key FROM lemon_subscriptions WHERE subscription_id=?", (sub_id,)).fetchone()
        if row and row["api_key"]:
            cur.execute("UPDATE api_keys SET status='revoked' WHERE api_key=?", (row["api_key"],))

        cur.execute("""
          UPDATE lemon_subscriptions
          SET status=?, updated_at=?
          WHERE subscription_id=?
        """, ("cancelled" if event_name == "subscription_cancelled" else "expired", now, sub_id))

        conn.commit()
        conn.close()
        return {"ok": True}

    conn.close()
    return {"ok": True, "ignored": event_name}