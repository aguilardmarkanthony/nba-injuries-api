"""Public, read-only production probe. A nonzero exit triggers a GitHub Actions failure."""

import argparse
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_URL = "https://nba-injuries-api-1.onrender.com/v1/freshness"


class MonitorFailure(RuntimeError):
    pass


def validate_report(report: dict, http_status: int) -> str:
    if not isinstance(report, dict):
        raise MonitorFailure("Response is not a freshness report")
    state = report.get("status")
    if not isinstance(state, str):
        raise MonitorFailure("Missing or invalid freshness state")
    if http_status != 200 or state not in {"fresh", "delayed"}:
        safe_state = state if state in {"stale", "degraded", "unavailable"} else "unexpected"
        raise MonitorFailure(f"Freshness check returned HTTP {http_status} ({safe_state})")
    info = report.get("cache")
    if not isinstance(info, dict) or report.get("source") != "ESPN":
        raise MonitorFailure("Missing or invalid cache/source metadata")
    numbers = [
        info.get("ageSeconds"),
        report.get("alertAfterSeconds"),
        report.get("consecutiveRefreshFailures"),
        report.get("updatedAt"),
    ]
    if any(type(value) is not int for value in numbers):
        raise MonitorFailure("Invalid freshness measurements")
    age, threshold, failures, updated = numbers
    if age < 0 or threshold <= 0 or age >= threshold or not 0 <= failures < 3 or updated <= 0:
        raise MonitorFailure("Snapshot exceeded freshness or refresh-failure limits")
    return f"PASS: {state}; snapshot age {age}s; alert threshold {threshold}s; refresh failures {failures}"


def probe(url: str, timeout: float) -> str:
    request = Request(
        url, headers={"Accept": "application/json", "User-Agent": "nba-api-freshness-monitor/1"}
    )
    try:
        try:
            response = urlopen(request, timeout=timeout)
        except HTTPError as error:
            response = error
        with response:
            body = response.read(65_537)
            if len(body) > 65_536:
                raise MonitorFailure("Freshness response exceeded the size limit")
            report = json.loads(body)
            return validate_report(report, response.status)
    except (URLError, TimeoutError, OSError):
        raise MonitorFailure("Could not reach the freshness endpoint") from None
    except (ValueError, UnicodeError):
        raise MonitorFailure("Freshness endpoint did not return valid JSON") from None


def check(url: str, attempts: int = 3, timeout: float = 75, delay: float = 15) -> int:
    if attempts < 1 or timeout <= 0 or delay < 0:
        raise ValueError("Invalid monitor retry settings")
    for attempt in range(1, attempts + 1):
        try:
            print(probe(url, timeout))
            return 0
        except MonitorFailure as error:
            print(f"Attempt {attempt}/{attempts}: {error}")
            if attempt < attempts:
                time.sleep(delay)
    print("ALERT: production injury data freshness could not be verified.")
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()
    return check(args.url)


if __name__ == "__main__":
    raise SystemExit(main())
