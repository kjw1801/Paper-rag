import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SITEVERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


class TurnstileRejected(RuntimeError):
    """The visitor did not pass the Turnstile challenge."""


class TurnstileUnavailable(RuntimeError):
    """Turnstile could not be verified because of configuration or upstream failure."""


class TurnstileVerifier:
    def __init__(
        self,
        *,
        secret_key: str | None,
        required: bool,
        expected_hostnames: set[str] | None = None,
        expected_action: str = "paper_ask",
        timeout: float = 5.0,
    ) -> None:
        self.secret_key = secret_key
        self.required = required
        self.expected_hostnames = expected_hostnames or set()
        self.expected_action = expected_action
        self.timeout = timeout

    @classmethod
    def from_environment(cls) -> "TurnstileVerifier":
        hostnames = {
            hostname.strip().lower()
            for hostname in os.getenv("TURNSTILE_EXPECTED_HOSTNAMES", "").split(",")
            if hostname.strip()
        }
        return cls(
            secret_key=os.getenv("TURNSTILE_SECRET_KEY") or None,
            required=os.getenv("TURNSTILE_REQUIRED", "1") == "1",
            expected_hostnames=hostnames,
            expected_action=os.getenv("TURNSTILE_EXPECTED_ACTION", "paper_ask"),
        )

    @property
    def enabled(self) -> bool:
        return self.required or bool(self.secret_key)

    def verify(self, token: str | None, remote_ip: str | None = None) -> None:
        if not self.enabled:
            return
        if not self.secret_key:
            raise TurnstileUnavailable("Turnstile 서버 키가 설정되지 않았습니다.")
        if not token:
            raise TurnstileRejected("보안 확인을 완료한 뒤 다시 질문해 주세요.")

        payload = {"secret": self.secret_key, "response": token}
        if remote_ip:
            payload["remoteip"] = remote_ip
        request = Request(
            SITEVERIFY_URL,
            data=urlencode(payload).encode(),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read())
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            raise TurnstileUnavailable(
                "보안 확인 서버에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요."
            ) from error

        if not result.get("success"):
            raise TurnstileRejected(
                "보안 확인이 만료되었거나 유효하지 않습니다. 다시 확인해 주세요."
            )
        hostname = str(result.get("hostname", "")).lower()
        if self.expected_hostnames and hostname not in self.expected_hostnames:
            raise TurnstileRejected("허용되지 않은 사이트에서 전송된 요청입니다.")
        if self.expected_action and result.get("action") != self.expected_action:
            raise TurnstileRejected("보안 확인 요청이 올바르지 않습니다.")
