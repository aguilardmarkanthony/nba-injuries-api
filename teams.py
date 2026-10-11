"""Stable NBA team identifiers and compatible v1 ESPN abbreviations."""

TEAM_NAMES = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "LA Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}
CANONICAL_TO_SOURCE = {team: team for team in TEAM_NAMES}
CANONICAL_TO_SOURCE.update({"GSW": "GS", "NOP": "NO", "NYK": "NY", "SAS": "SA", "UTA": "UTAH", "WAS": "WSH"})
ALIASES = {source: canonical for canonical, source in CANONICAL_TO_SOURCE.items()}
ALIASES.update({team: team for team in TEAM_NAMES})
NAME_TO_CANONICAL = {name.casefold(): team for team, name in TEAM_NAMES.items()}
NAME_TO_CANONICAL["los angeles clippers"] = "LAC"


def canonical_team(value: str) -> str:
    """Accept canonical or historical source codes, ignoring case and whitespace."""
    code = value.strip().upper()
    if code not in ALIASES:
        raise ValueError("Unknown NBA team abbreviation")
    return ALIASES[code]


def source_team(value: str) -> str:
    return CANONICAL_TO_SOURCE[canonical_team(value)]


def directory() -> list[dict]:
    return [
        {
            "code": code,
            "name": TEAM_NAMES[code],
            "sourceCode": CANONICAL_TO_SOURCE[code],
            "aliases": sorted(alias for alias, canonical in ALIASES.items() if canonical == code),
        }
        for code in sorted(TEAM_NAMES)
    ]
