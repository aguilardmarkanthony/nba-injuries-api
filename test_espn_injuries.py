"""Manual source inspection CLI; automated regression tests live in tests/."""

import argparse

from injury_feed import build_injury_map, load_data
from teams import source_team


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team", help="Canonical or source team abbreviation, for example GSW or GS")
    parser.add_argument("--file", help="Read source JSON from a local file instead of ESPN")
    args = parser.parse_args()
    try:
        team = source_team(args.team) if args.team else None
    except ValueError as exc:
        parser.error(str(exc))
    injuries = build_injury_map(load_data(args.file))
    print(f"Teams with entries: {len(injuries)} | Player entries: {sum(map(len, injuries.values()))}")
    for code, rows in sorted(injuries.items()):
        if team is None:
            print(f"{code}: {len(rows)}")
        elif code == team:
            for row in rows:
                print(f"{row['player']}: {row['status']} ({row['reason']})")


if __name__ == "__main__":
    main()
