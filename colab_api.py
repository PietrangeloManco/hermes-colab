"""Colab runtime assignment API (the ``/tun/m`` endpoints the Colab web app and CLIs use).

Every call goes through Hermes: ``credentialed_provider_request("google", ...)`` attaches the
user's own OAuth grant (declared under ``oauth:`` in plugin.yaml) for colab.research.google.com
only. This module never sees the access token, refresh token or client secret.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlencode

COLAB = "https://colab.research.google.com"
TUN = f"{COLAB}/tun/m"
XSSI_PREFIX = ")]}'\n"
CLIENT_AGENT = "hermes-colab"
ACCELERATORS = ("NONE", "T4", "L4", "A100", "H100", "G4", "V5E1", "V6E1")
_TPU = {"V5E1", "V6E1"}
_HIGH_MEM_ONLY = {"L4", "V5E1", "V6E1"}


class ColabError(RuntimeError):
    """A Colab API call failed. ``status`` is the HTTP status (0 for transport errors)."""

    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


class TooManyRuntimes(ColabError):
    """Colab refused a new runtime because the account already holds its maximum."""


class NotSignedIn(ColabError):
    """Hermes has no Google grant for this plugin (or the grant no longer matches the manifest)."""


@dataclass(frozen=True)
class Runtime:
    endpoint: str
    accelerator: str
    url: str
    proxy_token: str
    token_expires_in: int = 0

    def public(self) -> dict[str, Any]:
        return {"endpoint": self.endpoint, "accelerator": self.accelerator}


# (method, url, headers, timeout) -> (status, headers, body bytes)
Transport = Callable[[str, str, Mapping[str, str], float], tuple[int, Mapping[str, str], bytes]]


def hermes_transport(method: str, url: str, headers: Mapping[str, str], timeout: float):
    """Default transport: the request is made by Hermes core with the plugin's Google grant."""
    from hermes_cli.plugin_provider_requests import ProviderNotSignedIn, credentialed_provider_request

    try:
        from hermes_cli.auth_constants import AuthError
    except ImportError:  # pragma: no cover - older cores
        AuthError = RuntimeError  # type: ignore[assignment,misc]
    try:
        resp = credentialed_provider_request("google", method, url, headers=dict(headers), timeout=timeout)
    except ProviderNotSignedIn as exc:
        raise NotSignedIn(str(exc)) from None
    except AuthError as exc:
        code = getattr(exc, "code", "")
        raise NotSignedIn(f"{code}: {exc}" if code else str(exc)) from None
    except PermissionError as exc:
        raise ColabError(f"Hermes refused the request: {exc}") from None
    except Exception as exc:  # httpx transport errors arrive detached from their request
        raise ColabError(f"{type(exc).__name__}: {exc}") from None
    return resp.status, resp.headers, resp.body


def _nbh() -> str:
    """Random notebook hash in the web-safe base64 shape Colab expects (44 chars)."""
    raw = str(uuid.uuid4())
    return raw.replace("-", "_") + "." * (44 - len(raw))


def _decode(body: bytes) -> Any:
    text = body.decode("utf-8", "replace")
    if text.startswith(XSSI_PREFIX):
        text = text[len(XSSI_PREFIX):]
    text = text.strip()
    return json.loads(text) if text else None


def _runtime(data: Mapping[str, Any], accelerator: str = "") -> Runtime:
    info = data.get("runtimeProxyInfo") or {}
    if not data.get("endpoint") or not info.get("url") or not info.get("token"):
        raise ColabError("Colab returned an assignment without endpoint or runtime proxy")
    return Runtime(
        endpoint=str(data["endpoint"]),
        accelerator=str(data.get("accelerator") or accelerator or "NONE"),
        url=str(info["url"]).rstrip("/"),
        proxy_token=str(info["token"]),
        token_expires_in=int(info.get("tokenExpiresInSeconds") or 0),
    )


class ColabTunnel:
    def __init__(self, transport: Transport = hermes_transport):
        self._transport = transport

    def _call(self, method: str, url: str, *, headers: Mapping[str, str] | None = None,
              timeout: float = 30.0, expect_json: bool = True) -> Any:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}authuser=0"
        merged = {"Accept": "application/json", "X-Colab-Client-Agent": CLIENT_AGENT, **(headers or {})}
        status, _headers, body = self._transport(method, url, merged, timeout)
        if status == 412:
            raise TooManyRuntimes(
                "Colab says this account already holds its maximum number of runtimes; release one "
                "(colab_runtimes / colab_release_runtime) or wait for one to end", status)
        if status in (401, 403):
            raise ColabError(_error_text(status, body) + " — check that the Google account has Colab "
                             "access and that the OAuth client includes the Colaboratory scope", status)
        if status >= 400:
            raise ColabError(_error_text(status, body), status)
        return _decode(body) if expect_json else None

    def list_runtimes(self) -> list[Runtime]:
        data = self._call("GET", f"{TUN}/assignments") or {}
        return [_runtime(item) for item in data.get("assignments") or []]

    def assign(self, accelerator: str = "T4", *, high_mem: bool = False) -> Runtime:
        accelerator = (accelerator or "NONE").upper()
        if accelerator not in ACCELERATORS:
            raise ValueError(f"Unknown accelerator {accelerator!r}; choose one of {', '.join(ACCELERATORS)}")
        params = {"nbh": _nbh()}
        if accelerator != "NONE":
            params["variant"] = "TPU" if accelerator in _TPU else "GPU"
            params["accelerator"] = accelerator
        if high_mem and accelerator not in _HIGH_MEM_ONLY:
            params["shape"] = "hm"
        url = f"{TUN}/assign?{urlencode(params)}"
        first = self._call("GET", url) or {}
        if "endpoint" in first:  # already assigned to this notebook hash
            return _runtime(first, accelerator)
        token = first.get("token")
        if not token:
            raise ColabError("Colab did not return an assignment token")
        data = self._call("POST", url, headers={"X-Goog-Colab-Token": str(token)}, timeout=120.0) or {}
        return _runtime(data, accelerator)

    def keep_alive(self, endpoint: str) -> bool:
        """Refresh the idle timer. False when the runtime is gone (404); a timeout counts as success."""
        url = f"{TUN}/{quote(endpoint, safe='')}/keep-alive/"
        try:
            self._call("GET", url, headers={"X-Colab-Tunnel": "Google"}, timeout=10.0, expect_json=False)
        except ColabError as exc:
            if exc.status == 404:
                return False
            if exc.status == 0 and "timeout" in str(exc).lower():
                return True
            raise
        return True

    def release(self, endpoint: str) -> None:
        url = f"{TUN}/unassign/{quote(endpoint, safe='')}"
        first = self._call("GET", url) or {}
        token = first.get("token")
        if not token:
            raise ColabError("Colab did not return an unassign token")
        self._call("POST", url, headers={"X-Goog-Colab-Token": str(token)}, expect_json=False)


def _error_text(status: int, body: bytes) -> str:
    try:
        data = _decode(body)
    except (ValueError, UnicodeDecodeError):
        data = None
    message = ""
    if isinstance(data, dict):
        err = data.get("error")
        message = err.get("message", "") if isinstance(err, dict) else str(err or data.get("message") or "")
    if not message:
        message = body[:200].decode("utf-8", "replace").strip()
    return f"Colab HTTP {status}: {message}" if message else f"Colab HTTP {status}"
