import io
import json
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

import pytest

from examples import injury_watch as example


def test_example_sends_key_in_header_and_encodes_filters(monkeypatch):
    def open_request(request, timeout):
        assert timeout == 75
        assert "private-key" not in request.full_url
        assert request.get_header("X-rapidapi-key") == "private-key"
        assert request.get_header("X-rapidapi-host") == example.HOST
        assert parse_qs(urlsplit(request.full_url).query) == {
            "teams": ["GSW,NOP"],
            "status": ["Out For Season"],
        }
        return io.BytesIO(json.dumps({"teams": {}, "playerEntryCount": 0}).encode())

    monkeypatch.setattr(example, "urlopen", open_request)
    assert (
        example.fetch_injuries("private-key", teams="GSW,NOP", status="Out For Season")["playerEntryCount"]
        == 0
    )


def test_example_builds_player_id_path(monkeypatch):
    def open_request(request, timeout):
        assert request.full_url == f"https://{example.HOST}/v1/player/1966"
        return io.BytesIO(b'{"playerId":"1966","count":0,"results":[]}')

    monkeypatch.setattr(example, "urlopen", open_request)
    assert example.fetch_injuries("private-key", player_id="1966")["count"] == 0


@pytest.mark.parametrize(
    "params",
    [
        {"team": "GS", "teams": "NO"},
        {"player_id": "../health"},
        {"player_id": "0"},
        {"player_id": "1966", "status": "OUT"},
    ],
)
def test_example_rejects_invalid_combinations_before_network(params):
    with pytest.raises(ValueError):
        example.fetch_injuries("private-key", **params)


@pytest.mark.parametrize(
    "error",
    [
        HTTPError("https://example.test", 429, "rate limit", {}, io.BytesIO(b"private-key")),
        URLError("private-key"),
    ],
)
def test_example_does_not_echo_credentials_from_errors(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(example, "urlopen", fail)
    with pytest.raises(example.APIError) as caught:
        example.fetch_injuries("private-key")
    assert "private-key" not in str(caught.value)
