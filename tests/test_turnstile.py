import json
from unittest.mock import MagicMock, patch

import pytest

from backend.turnstile import TurnstileRejected, TurnstileUnavailable, TurnstileVerifier


def _response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(payload).encode()
    return response


def test_skips_verification_when_explicitly_disabled() -> None:
    TurnstileVerifier(secret_key=None, required=False).verify(None)


def test_fails_closed_without_secret() -> None:
    with pytest.raises(TurnstileUnavailable, match="서버 키"):
        TurnstileVerifier(secret_key=None, required=True).verify("token")


def test_rejects_missing_token() -> None:
    with pytest.raises(TurnstileRejected, match="완료"):
        TurnstileVerifier(secret_key="secret", required=True).verify(None)


@patch("backend.turnstile.urlopen")
def test_accepts_expected_hostname_and_action(mock_urlopen: MagicMock) -> None:
    mock_urlopen.return_value = _response(
        {"success": True, "hostname": "paper.woojulab.com", "action": "paper_ask"}
    )
    verifier = TurnstileVerifier(
        secret_key="secret",
        required=True,
        expected_hostnames={"paper.woojulab.com"},
    )
    verifier.verify("valid-token", "203.0.113.10")


@pytest.mark.parametrize(
    "payload",
    [
        {"success": False},
        {"success": True, "hostname": "attacker.example", "action": "paper_ask"},
        {"success": True, "hostname": "paper.woojulab.com", "action": "wrong"},
    ],
)
@patch("backend.turnstile.urlopen")
def test_rejects_invalid_or_mismatched_result(
    mock_urlopen: MagicMock, payload: dict
) -> None:
    mock_urlopen.return_value = _response(payload)
    verifier = TurnstileVerifier(
        secret_key="secret",
        required=True,
        expected_hostnames={"paper.woojulab.com"},
    )
    with pytest.raises(TurnstileRejected):
        verifier.verify("token")
