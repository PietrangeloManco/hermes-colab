"""Talk to an assigned Colab runtime: its Jupyter kernel behind the Colab runtime proxy.

The runtime proxy token comes from the assignment (``colab_api.Runtime``); it is short-lived, kept
in memory only and never persisted. Everything the plugin does on the VM — launching the detached
runner, polling it, moving files — is a small Python snippet executed in the runtime's kernel, so
the plugin depends on one protocol only (Jupyter messaging over a websocket).
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlencode

import httpx

from .colab_api import CLIENT_AGENT, Runtime

REMOTE_ROOT = "/content/hermes_colab"
UPLOAD_CHUNK = 4 * 1024 * 1024    # raw bytes per execute_request (shell channel, no iopub limit)
DOWNLOAD_CHUNK = 768 * 1024       # raw bytes per reply: stays under Jupyter's iopub data-rate window
_SENTINEL = "\x1e@hermes-colab@"


class RuntimeGone(RuntimeError):
    """The runtime proxy answered 404/410 or refused the token: the VM is gone or reassigned."""


class RuntimeUnreachable(RuntimeError):
    """A transient transport failure talking to the runtime."""


class KernelError(RuntimeError):
    """The snippet raised inside the kernel."""


class RuntimeClient:
    def __init__(self, runtime: Runtime, *, http: httpx.Client | None = None,
                 ws_connect: Callable[..., Any] | None = None, root: str = REMOTE_ROOT):
        self.runtime = runtime
        self.root = root.rstrip("/")
        self._http = http or httpx.Client(timeout=30.0, follow_redirects=False)
        self._ws_connect = ws_connect
        self._kernel_id: str | None = None
        self._session = uuid.uuid4().hex

    # ── HTTP ──────────────────────────────────────────────────────────────────────────────────────
    def _headers(self) -> dict[str, str]:
        return {"X-Colab-Runtime-Proxy-Token": self.runtime.proxy_token, "X-Colab-Client-Agent": CLIENT_AGENT}

    def _params(self) -> dict[str, str]:
        return {"authuser": "0", "colab-runtime-proxy-token": self.runtime.proxy_token}

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            resp = self._http.request(method, f"{self.runtime.url}{path}", headers=self._headers(),
                                      params=self._params(), **kwargs)
        except httpx.HTTPError as exc:
            raise RuntimeUnreachable(f"{type(exc).__name__}: {exc}") from None
        if resp.status_code in (401, 403, 404, 410):
            raise RuntimeGone(f"runtime proxy answered HTTP {resp.status_code}")
        if resp.status_code >= 500:
            raise RuntimeUnreachable(f"runtime proxy answered HTTP {resp.status_code}")
        if resp.status_code >= 400:
            raise RuntimeUnreachable(f"runtime proxy answered HTTP {resp.status_code}: {resp.text[:200]}")
        return resp

    def kernel_id(self) -> str:
        if self._kernel_id is None:
            kernels = self._request("GET", "/api/kernels").json() or []
            if kernels:
                self._kernel_id = str(kernels[0]["id"])
            else:
                self._kernel_id = str(self._request("POST", "/api/kernels", json={"name": "python3"}).json()["id"])
        return self._kernel_id

    # ── Kernel execution ──────────────────────────────────────────────────────────────────────────
    def _connect(self, url: str):
        if self._ws_connect is not None:
            return self._ws_connect(url, self._headers())
        from websockets.sync.client import connect

        return connect(url, additional_headers=self._headers(), open_timeout=30, max_size=None,
                       ping_interval=20, ping_timeout=20)

    def execute(self, code: str, *, timeout: float = 120.0) -> str:
        """Run *code* in the runtime's kernel; return its stdout. Raises KernelError on an exception."""
        kernel = self.kernel_id()
        base = self.runtime.url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        url = f"{base}/api/kernels/{kernel}/channels?" + urlencode({"session_id": self._session, **self._params()})
        msg_id = uuid.uuid4().hex
        request = {
            "header": {"msg_id": msg_id, "username": "hermes", "session": self._session,
                       "msg_type": "execute_request", "version": "5.3",
                       "date": datetime.now(UTC).isoformat()},
            "parent_header": {}, "metadata": {}, "channel": "shell", "buffers": [],
            "content": {"code": code, "silent": False, "store_history": False, "user_expressions": {},
                        "allow_stdin": False, "stop_on_error": True},
        }
        out: list[str] = []
        error: str | None = None
        deadline = time.monotonic() + timeout
        try:
            ws = self._connect(url)
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", 0)
            if status in (401, 403, 404, 410):
                self._kernel_id = None
                raise RuntimeGone(f"kernel websocket refused with HTTP {status}") from None
            raise RuntimeUnreachable(f"kernel websocket: {type(exc).__name__}: {exc}") from None
        try:
            ws.send(json.dumps(request))
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeUnreachable(f"kernel did not finish within {int(timeout)}s")
                try:
                    raw = ws.recv(timeout=remaining)
                except TimeoutError:
                    raise RuntimeUnreachable(f"kernel did not finish within {int(timeout)}s") from None
                except Exception as exc:
                    raise RuntimeUnreachable(f"kernel websocket closed: {type(exc).__name__}") from None
                msg = json.loads(raw)
                if (msg.get("parent_header") or {}).get("msg_id") != msg_id:
                    continue
                kind = (msg.get("header") or {}).get("msg_type") or msg.get("msg_type")
                content = msg.get("content") or {}
                if kind == "stream" and content.get("name") == "stdout":
                    out.append(content.get("text", ""))
                elif kind == "error":
                    error = f"{content.get('ename')}: {content.get('evalue')}"
                elif kind == "status" and content.get("execution_state") == "idle":
                    break
        finally:
            try:
                ws.close()
            except Exception:
                pass
        if error:
            raise KernelError(error)
        return "".join(out)

    def call(self, code: str, *, timeout: float = 120.0) -> Any:
        """Execute *code* that ends by printing ``_SENTINEL + json``; return the decoded value."""
        text = self.execute(code, timeout=timeout)
        _, sep, payload = text.rpartition(_SENTINEL)
        if not sep:
            raise KernelError("snippet returned no result")
        return json.loads(payload.strip())

    # ── Operations used by the supervisor ─────────────────────────────────────────────────────────
    def run_dir(self, run_id: str) -> str:
        return f"{self.root}/{run_id}"

    def prepare(self, run_id: str) -> str:
        run_dir = self.run_dir(run_id)
        self.call(_snippet(f"""
import os, shutil
d = {run_dir!r}
for sub in ("work", "outputs", ".incoming"):
    os.makedirs(os.path.join(d, sub), exist_ok=True)
emit({{"ok": True}})
"""))
        return run_dir

    def upload(self, remote_path: str, data: bytes) -> None:
        """Write *data* to *remote_path* in chunks, then move it into place atomically."""
        part = remote_path + ".part"
        for offset in range(0, max(len(data), 1), UPLOAD_CHUNK):
            chunk = base64.b64encode(data[offset:offset + UPLOAD_CHUNK]).decode("ascii")
            mode = "wb" if offset == 0 else "ab"
            self.call(_snippet(f"""
import base64, os
os.makedirs(os.path.dirname({part!r}), exist_ok=True)
with open({part!r}, {mode!r}) as fh:
    fh.write(base64.b64decode({chunk!r}))
emit({{"ok": True}})
"""), timeout=300)
        self.call(_snippet(f"""
import os
os.replace({part!r}, {remote_path!r})
emit({{"size": os.path.getsize({remote_path!r})}})
"""))

    def unzip(self, archive: str, dest: str) -> int:
        return int(self.call(_snippet(f"""
import os, zipfile
with zipfile.ZipFile({archive!r}) as zf:
    names = zf.namelist()
    root = os.path.realpath({dest!r})
    for n in names:
        target = os.path.realpath(os.path.join(root, n))
        if not (target == root or target.startswith(root + os.sep)):
            raise ValueError("archive entry escapes destination: " + n)
    zf.extractall(root)
os.remove({archive!r})
emit(len(names))
"""), timeout=600))

    def download(self, remote_path: str, size: int) -> bytes:
        parts: list[bytes] = []
        for offset in range(0, max(size, 1), DOWNLOAD_CHUNK):
            b64 = self.call(_snippet(f"""
import base64
with open({remote_path!r}, "rb") as fh:
    fh.seek({offset})
    emit(base64.b64encode(fh.read({DOWNLOAD_CHUNK})).decode("ascii"))
"""), timeout=300)
            parts.append(base64.b64decode(b64))
        return b"".join(parts)

    def launch(self, run_dir: str) -> int:
        return int(self.call(_snippet(f"""
import os, subprocess, sys
err = open(os.path.join({run_dir!r}, "runner.err"), "ab")
p = subprocess.Popen([sys.executable, os.path.join({run_dir!r}, "runner.py"), {run_dir!r}],
                     cwd={run_dir!r}, stdin=subprocess.DEVNULL, stdout=err, stderr=err, start_new_session=True)
emit(p.pid)
""")))

    def poll(self, run_dir: str, log_offset: int, max_log: int = 64 * 1024) -> dict[str, Any]:
        return self.call(_snippet(f"""
import json, os, time
d = {run_dir!r}
res = {{"status": None, "log": "", "log_offset": {log_offset}, "outputs": [], "runner_err": "", "now": time.time()}}
try:
    res["status"] = json.load(open(os.path.join(d, "status.json")))
except (OSError, ValueError):
    pass
try:
    with open(os.path.join(d, "run.log"), "rb") as fh:
        fh.seek({log_offset})
        chunk = fh.read({max_log})
    res["log"] = chunk.decode("utf-8", "replace")
    res["log_offset"] = {log_offset} + len(chunk)
except OSError:
    pass
try:
    res["runner_err"] = open(os.path.join(d, "runner.err"), "rb").read()[-2000:].decode("utf-8", "replace")
except OSError:
    pass
out = os.path.join(d, "outputs")
for base, dirs, files in os.walk(out):
    dirs[:] = [x for x in dirs if not x.startswith(".")]
    for f in files:
        p = os.path.join(base, f)
        try:
            st = os.stat(p)
        except OSError:
            continue
        res["outputs"].append([os.path.relpath(p, out), st.st_size, st.st_mtime])
        if len(res["outputs"]) >= 2000:
            break
emit(res)
"""))

    def signal_runner(self, run_dir: str) -> bool:
        return bool(self.call(_snippet(f"""
import json, os, signal
try:
    pid = json.load(open(os.path.join({run_dir!r}, "status.json")))["pid"]
    os.kill(pid, signal.SIGTERM)
    emit(True)
except (OSError, ValueError, KeyError):
    emit(False)
""")))

    def close(self) -> None:
        self._http.close()


def _snippet(body: str) -> str:
    """Wrap *body* so it can ``emit(value)`` one JSON result after the sentinel."""
    return (
        "def emit(value):\n"
        "    import json as _j\n"
        f"    print({_SENTINEL!r} + _j.dumps(value))\n"
        + body.strip("\n") + "\n"
    )
