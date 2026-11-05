"""End-to-end tests against a real local Jupyter server standing in for a Colab runtime.

The Colab tunnel (assign / keep-alive / release) is faked; everything that talks to the runtime —
kernel websocket protocol, chunked upload, unzip, detached runner, polling, output sync, resume
with restored checkpoints on a fresh VM — runs for real.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import httpx
import pytest

from hermes_colab.colab_api import Runtime
from hermes_colab.runtime import RuntimeClient
from hermes_colab.store import Store
from hermes_colab.supervisor import Supervisor
from hermes_colab.tools import Tools


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def jupyter(tmp_path_factory):
    root = tmp_path_factory.mktemp("vm")
    port = _free_port()
    env = {**os.environ, "JUPYTER_RUNTIME_DIR": str(root / ".rt"), "JUPYTER_DATA_DIR": str(root / ".data")}
    proc = subprocess.Popen(
        [sys.executable, "-m", "jupyter_server", "--no-browser", f"--port={port}", "--ip=127.0.0.1",
         "--IdentityProvider.token=", "--ServerApp.password=", "--ServerApp.disable_check_xsrf=True",
         f"--ServerApp.root_dir={root}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(f"{url}/api/status", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.2)
    else:
        proc.kill()
        pytest.skip("local jupyter_server did not start")
    yield url, root
    proc.terminate()
    proc.wait(timeout=10)


def test_kernel_roundtrip(jupyter, tmp_path):
    url, root = jupyter
    client = RuntimeClient(Runtime("vm-x", "NONE", url, "tok"), root=str(root / "vm-x"))
    assert client.execute("print(6 * 7)").strip() == "42"
    run_dir = client.prepare("r1")
    blob = os.urandom(3 * 1024 * 1024 + 17)
    client.upload(f"{run_dir}/blob.bin", blob)
    assert client.download(f"{run_dir}/blob.bin", len(blob)) == blob


class FakeTunnel:
    """Hands out fresh 'VMs' (separate roots on the same Jupyter server); can lose one on demand."""

    def __init__(self, url):
        self.url = url
        self.count = 0
        self.alive: set[str] = set()
        self.released: list[str] = []
        self.lose_after_keepalives: dict[str, int] = {}

    def assign(self, accelerator, *, high_mem=False):
        self.count += 1
        endpoint = f"vm{self.count}"
        self.alive.add(endpoint)
        return Runtime(endpoint, accelerator, self.url, "tok")

    def keep_alive(self, endpoint):
        left = self.lose_after_keepalives.get(endpoint)
        if left is not None:
            if left <= 0:
                self.alive.discard(endpoint)
            self.lose_after_keepalives[endpoint] = left - 1
        return endpoint in self.alive

    def list_runtimes(self):
        return [Runtime(e, "T4", self.url, "tok") for e in sorted(self.alive)]

    def release(self, endpoint):
        self.alive.discard(endpoint)
        self.released.append(endpoint)


TRAIN = textwrap.dedent('''
    import json, os, sys, time
    out = os.environ["HERMES_COLAB_OUTPUT_DIR"]
    ckpt = os.path.join(out, "ckpt.json")
    start = 0
    if os.environ.get("HERMES_COLAB_RESUME") == "1" and os.path.exists(ckpt):
        start = json.load(open(ckpt))["epoch"] + 1
        print("resumed at epoch", start)
    for epoch in range(start, 6):
        time.sleep(1.2)
        json.dump({"epoch": epoch}, open(ckpt + ".tmp", "w")); os.replace(ckpt + ".tmp", ckpt)
        print("HERMES_PROGRESS " + json.dumps({"step": epoch + 1, "total": 6, "label": f"epoch {epoch + 1}/6"}))
    open(os.path.join(out, "done.txt"), "w").write("epochs run on attempt %s" % os.environ["HERMES_COLAB_ATTEMPT"])
''')


def _drive(supervisor, store, run_id, *, until, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        supervisor.tick()
        run = store.get(run_id)
        if until(run):
            return run
        time.sleep(0.3)
    raise AssertionError(f"run stuck in {store.get(run_id)['state']}: {store.get(run_id)['events']}")


def _make(tmp_path, jupyter, *, settings=None):
    url, root = jupyter
    store = Store(tmp_path / "runs.db")
    tunnel = FakeTunnel(url)
    finals = []
    supervisor = Supervisor(
        store, settings=lambda: {"auto_resume": True, "max_resumes": 2, "max_sync_mb": 64, **(settings or {})},
        tunnel=tunnel, runtime_factory=lambda rt: RuntimeClient(rt, root=str(root / f"{tmp_path.name}-{rt.endpoint}")),
        on_final=finals.append)
    tools = Tools(store=store, supervisor=supervisor, ensure_supervisor=lambda: None,
                  settings=lambda: {"default_accelerator": "T4", "max_upload_mb": 16,
                                    "compute_unit_rates": {"T4": 2.0}},
                  cwd=lambda: str(tmp_path), env=lambda name: "x")
    return store, tunnel, supervisor, tools, finals


def test_run_resumes_on_fresh_runtime_after_loss(jupyter, tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "train.py").write_text(TRAIN)
    (project / ".env").write_text("SECRET=1")
    store, tunnel, supervisor, tools, finals = _make(tmp_path, jupyter)

    started = json.loads(tools.colab_run({"path": "proj", "entry": "train.py", "name": "demo"}, session_id="s1"))
    assert started["ok"], started
    run_id = started["run"]["id"]
    assert started["files"] == 1  # .env never leaves the machine

    run = _drive(supervisor, store, run_id, until=lambda r: (r.get("progress") or {}).get("step", 0) >= 3
                 and "ckpt.json" in (r.get("synced") or {}))
    assert run["endpoint"] == "vm1"
    synced_epoch = json.loads((Path(run["local_dir"]) / "outputs" / "ckpt.json").read_text())["epoch"]
    supervisor._keepalive[run_id] = 0  # force a keep-alive on the next tick
    tunnel.lose_after_keepalives["vm1"] = 0  # Colab reclaims the VM

    run = _drive(supervisor, store, run_id, until=lambda r: r["state"] in ("succeeded", "failed"))
    assert run["state"] == "succeeded", run["events"]
    assert run["attempt"] == 2 and run["resumes"] == 1
    out = Path(run["local_dir"]) / "outputs"
    assert json.loads((out / "ckpt.json").read_text())["epoch"] == 5
    assert (out / "done.txt").read_text() == "epochs run on attempt 2"
    log = (Path(run["local_dir"]) / "run.log").read_text()
    assert "resumed at epoch" in log
    resumed_at = int(log.split("resumed at epoch")[1].split()[0])
    assert resumed_at >= synced_epoch + 1 >= 1  # resumed from the synced checkpoint, not from scratch
    assert "vm2" in tunnel.released  # the runtime is given back
    assert finals and finals[0]["id"] == run_id
    assert any("Runtime lost" in e[1] for e in run["events"])

    status = json.loads(tools.colab_status({"run": run_id}))["run"]
    assert status["state"] == "succeeded" and "ckpt.json" in status["outputs"]
    assert tools.pending_notices("s1") and not tools.pending_notices("s1")


def test_failing_script_reports_exit_code(jupyter, tmp_path):
    (tmp_path / "boom.py").write_text("import sys\nprint('about to fail')\nsys.exit(3)\n")
    store, tunnel, supervisor, tools, finals = _make(tmp_path, jupyter)
    run_id = json.loads(tools.colab_run({"path": "boom.py"}))["run"]["id"]
    run = _drive(supervisor, store, run_id, until=lambda r: r["state"] in ("succeeded", "failed"))
    assert run["state"] == "failed" and run["exit_code"] == 3
    assert "about to fail" in json.loads(tools.colab_logs({"run": run_id}))["lines"][-1] or \
        "about to fail" in "\n".join(json.loads(tools.colab_logs({"run": run_id}))["lines"])


def test_cancel_running_job(jupyter, tmp_path):
    (tmp_path / "slow.py").write_text("import time\nfor i in range(600):\n    print(i, flush=True); time.sleep(0.1)\n")
    store, tunnel, supervisor, tools, finals = _make(tmp_path, jupyter)
    run_id = json.loads(tools.colab_run({"path": "slow.py"}))["run"]["id"]
    _drive(supervisor, store, run_id, until=lambda r: r["state"] == "running" and r["log_offset"] > 0)
    assert json.loads(tools.colab_cancel({"run": run_id}))["ok"]
    run = _drive(supervisor, store, run_id, until=lambda r: r["state"] == "cancelled")
    assert run["endpoint"] is None and tunnel.released
