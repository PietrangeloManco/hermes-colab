"""hermes-colab runner: executed ON the Colab runtime, never inside Hermes.

Started detached by the plugin (``python runner.py <run dir>``) so the job outlives the kernel
connection. It installs the run's requirements, runs the entry point (a Python script, or a
notebook through papermill / nbconvert), and keeps ``status.json`` current for the plugin to poll:

    state      installing | running | finished
    exit_code  the entry point's exit code once finished
    progress   last ``HERMES_PROGRESS {...}`` line or papermill cell counter
    gpu        nvidia-smi sample (name, util %, memory)

Environment given to the job: HERMES_COLAB_RUN_ID, HERMES_COLAB_ATTEMPT, HERMES_COLAB_OUTPUT_DIR
(synced home while the job runs and restored on a fresh runtime) and HERMES_COLAB_RESUME=1 from the
second attempt on, so a training script can continue from its last checkpoint.
Standard library only: the runtime's Python is whatever Colab ships.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

RUN_DIR = os.path.abspath(sys.argv[1])
SPEC = json.load(open(os.path.join(RUN_DIR, "spec.json"), encoding="utf-8"))
STATUS = os.path.join(RUN_DIR, "status.json")
LOG = os.path.join(RUN_DIR, "run.log")
WORK = os.path.join(RUN_DIR, "work")
OUTPUTS = os.path.join(RUN_DIR, "outputs")
_PROGRESS = re.compile(r"HERMES_PROGRESS\s+(\{.*\})")
_CELL = re.compile(r"Executing Cell (\d+)-+")
_lock = threading.Lock()
status: dict = {"state": "installing", "attempt": SPEC.get("attempt", 1), "started_at": time.time(),
                "pid": os.getpid(), "heartbeat": time.time()}


def write_status(**changes) -> None:
    with _lock:
        status.update(changes, heartbeat=time.time())
        tmp = STATUS + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(status, fh)
        os.replace(tmp, STATUS)


def gpu_sample() -> dict | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    name, util, used, total = [p.strip() for p in out.splitlines()[0].split(",")][:4]
    return {"name": name, "util": int(float(util)), "mem_used_mb": int(float(used)),
            "mem_total_mb": int(float(total))}


def heartbeat() -> None:
    while True:
        write_status(gpu=gpu_sample())
        time.sleep(15)


def run_logged(cmd: list[str], env: dict, cells: int = 0) -> int:
    with open(LOG, "ab", buffering=0) as log:
        log.write(("$ " + " ".join(cmd) + "\n").encode())
        proc = subprocess.Popen(cmd, cwd=WORK, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True)
        write_status(child_pid=proc.pid)
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            log.write(raw)
            line = raw.decode("utf-8", "replace")
            match = _PROGRESS.search(line)
            if match:
                try:
                    write_status(progress=json.loads(match.group(1)))
                except ValueError:
                    pass
            elif cells and (cell := _CELL.search(line)):
                done = int(cell.group(1))
                write_status(progress={"step": done, "total": cells, "fraction": round(done / cells, 3),
                                       "label": f"cell {done}/{cells}"})
        return proc.wait()


def main() -> int:
    os.makedirs(OUTPUTS, exist_ok=True)
    threading.Thread(target=heartbeat, daemon=True).start()
    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in (SPEC.get("env") or {}).items()})
    env.update({"HERMES_COLAB_RUN_ID": SPEC["run_id"], "HERMES_COLAB_ATTEMPT": str(SPEC.get("attempt", 1)),
                "HERMES_COLAB_OUTPUT_DIR": OUTPUTS, "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"})
    if SPEC.get("attempt", 1) > 1:
        env["HERMES_COLAB_RESUME"] = "1"
    py = sys.executable
    reqs = list(SPEC.get("requirements") or [])
    if os.path.isfile(os.path.join(WORK, "requirements.txt")):
        reqs += ["-r", "requirements.txt"]
    entry = SPEC["entry"]
    notebook = entry.endswith(".ipynb")
    if notebook:
        reqs.append("papermill>=2.6,<3")
    if reqs:
        code = run_logged([py, "-m", "pip", "install", "-q", "--disable-pip-version-check", *reqs], env)
        if code != 0:
            write_status(state="finished", exit_code=code, error="pip install failed", finished_at=time.time())
            return code
    write_status(state="running", running_at=time.time())
    if notebook:
        cells = sum(1 for c in json.load(open(os.path.join(WORK, entry), encoding="utf-8")).get("cells", [])
                    if c.get("cell_type") == "code")
        executed = os.path.join(OUTPUTS, os.path.basename(entry).replace(".ipynb", ".executed.ipynb"))
        cmd = [py, "-m", "papermill", entry, executed, "--log-output", "--no-progress-bar", "--cwd", WORK]
        code = run_logged(cmd, env, cells=cells)
    else:
        code = run_logged([py, "-u", entry, *[str(a) for a in SPEC.get("args") or []]], env)
    write_status(state="finished", exit_code=code, finished_at=time.time())
    return code


def _terminate(signum, _frame):
    child = status.get("child_pid")
    if child:
        try:
            os.killpg(child, signal.SIGTERM)
        except OSError:
            pass
    write_status(state="finished", exit_code=128 + signum, error="cancelled", finished_at=time.time())
    sys.exit(128 + signum)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _terminate)
    try:
        sys.exit(main())
    except Exception as exc:  # report, never leave the plugin polling a silent runner
        write_status(state="finished", exit_code=1, error=f"runner: {type(exc).__name__}: {exc}",
                     finished_at=time.time())
        raise
