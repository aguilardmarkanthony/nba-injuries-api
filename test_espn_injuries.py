"""
Test ESPN NBA injuries feed.

Usage:
  python test_espn_injuries.py
  python test_espn_injuries.py --team MEM
  python test_espn_injuries.py --file injuries.json --team ATL

What it does:
- Fetches ESPN injuries JSON (or reads a local file)
- Builds a map: TEAM_ABBR -> list of injuries
- Prints a summary + optional team details
"""

import argparse
import json
from collections import defaultdict

import requests


ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"


def load_data(path: str | None):
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    r = requests.get(ESPN_URL, timeout=20)
    r.raise_for_status()
    return r.json()


def build_injury_map(data: dict):
    """Return TEAM_ABBR -> list[ {player,status,reason,returnDate} ].

    ESPN feed shape commonly is:
      { "injuries": [ { "displayName": "Atlanta Hawks", "injuries": [ { athlete: {...} } ... ] } ... ] }

    But some older/alternate shapes may be flat. We support both.
    """
    mp = defaultdict(list)

    items = (data.get("injuries") or [])

    # Case A: team blocks
    if items and isinstance(items[0], dict) and ("injuries" in items[0]) and ("athlete" not in items[0]):
        for team_block in items:
            for inj in (team_block.get("injuries") or []):
                athlete = inj.get("athlete") or {}
                team = athlete.get("team") or {}
                abbr = (team.get("abbreviation") or "").upper().strip()
                name = (athlete.get("displayName") or athlete.get("shortName") or "").strip()
                status = (inj.get("status") or "").strip()
                details = inj.get("details") or {}
                fantasy = details.get("fantasyStatus") or {}
                fantasy_status = (fantasy.get("description") or "").strip()
                reason = (details.get("type") or "") or (details.get("detail") or "") or (inj.get("shortComment") or "")
                if not abbr or not name:
                    continue
                mp[abbr].append(
                    {
                        "player": name,
                        "status": (fantasy_status or status),
                        "reason": reason.strip(),
                        "returnDate": details.get("returnDate"),
                    }
                )
        return dict(mp)

    # Case B: flat injury rows
    for it in items:
        athlete = it.get("athlete") or {}
        team = athlete.get("team") or {}
        abbr = (team.get("abbreviation") or "").upper().strip()
        name = (athlete.get("displayName") or "").strip()
        status = (it.get("status") or "").strip()
        details = it.get("details") or {}
        fantasy = details.get("fantasyStatus") or {}
        fantasy_status = (fantasy.get("description") or "").strip()
        reason = (details.get("type") or "") or (details.get("detail") or "")
        if not abbr or not name:
            continue
        mp[abbr].append(
            {
                "player": name,
                "status": fantasy_status or status,
                "reason": (reason or "").strip(),
                "returnDate": details.get("returnDate"),
            }
        )
    return dict(mp)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", help="Team abbreviation like MEM, POR, LAL")
    ap.add_argument("--file", help="Read JSON from file instead of fetching ESPN")
    args = ap.parse_args()

    data = load_data(args.file)
    mp = build_injury_map(data)

    total = sum(len(v) for v in mp.values())
    print(f"✅ Loaded injuries feed. Teams with entries: {len(mp)} | Total player entries: {total}")

    if args.team:
        t = args.team.upper().strip()
        rows = mp.get(t, [])
        print(f"\n=== {t} injuries: {len(rows)} ===")
        for r in rows:
            print(f"- {r['player']}: {r['status']}" + (f" ({r['reason']})" if r.get("reason") else ""))

    else:
        # Print a quick top summary
        for t in sorted(mp.keys()):
            print(f"{t}: {len(mp[t])}")

if __name__ == "__main__":
    main()
