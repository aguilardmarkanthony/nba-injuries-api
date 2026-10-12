"""Stable filter codes; keep the source status text separate and unchanged."""

import re
from typing import Literal, get_args

StatusCode = Literal["OUT", "GTD", "OFS", "QUESTIONABLE", "DOUBTFUL", "PROBABLE", "DAY_TO_DAY", "UNKNOWN"]
STATUS_CODES = list(get_args(StatusCode))
ALIASES = {
    "OUT": "OUT",
    "GTD": "GTD",
    "GAME_TIME_DECISION": "GTD",
    "GAMETIME_DECISION": "GTD",
    "OFS": "OFS",
    "OUT_FOR_SEASON": "OFS",
    "OUT_FOR_THE_SEASON": "OFS",
    "QUESTIONABLE": "QUESTIONABLE",
    "DOUBTFUL": "DOUBTFUL",
    "PROBABLE": "PROBABLE",
    "DAY_TO_DAY": "DAY_TO_DAY",
    "DTD": "DAY_TO_DAY",
    "UNKNOWN": "UNKNOWN",
}


def parse_status(value: str) -> StatusCode:
    """Reject unsupported request values rather than silently mapping them to UNKNOWN."""
    key = re.sub(r"[\s-]+", "_", value.strip().upper())
    try:
        return ALIASES[key]
    except KeyError:
        raise ValueError("Unknown injury status") from None


def status_code(source_status: str) -> StatusCode:
    try:
        return parse_status(source_status)
    except ValueError:
        return "UNKNOWN"
