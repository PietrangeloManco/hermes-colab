"""Run supervisor: drives every active run of a profile through its states.

    queued → assigning → uploading → starting → running → succeeded | failed
                 ↑                                  │
                 └──────── resuming ←── runtime lost (Colab reclaimed the VM)
    waiting_gpu: Colab has no accelerator for us right now; retried with backoff.
    cancelling → cancelled

The supervisor is a daemon thread; one Hermes process per profile holds the lease in the run store
and drives the runs, so a CLI, the gateway and the Desktop backend never act on the same run twice,
and a run keeps being supervised after the process that started it exits.
"""

from __future__ import annotations

import atexit
import logging
import os
import re
import socket
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .bundle import zip_dir
from .colab_api import ColabError, ColabTunnel, NotSignedIn, Runtime, TooManyRuntimes
from .runtime import KernelError, RuntimeClient, RuntimeGone, RuntimeUnreachable
from .store import ACTIVE, Store

logger = logging.getLogger(__name__)

RUNNER = Path(__file__).with_name("remote") / "runner.py"
KEEPALIVE_EVERY = 60.0
LIVENESS_EVERY = 15.0  # Colab drops a reclaimed runtime from the assignment list at once
UNREACHABLE_LIMIT = 4
QUIET_SECONDS = 2.0
GPU_RETRY = (30, 60, 120, 300)


class Supervisor:
    def __init__(self, store: Store, *, settings: Callable[[], Mapping[str, Any]],
                 tunnel: ColabTunnel | None = None,
                 runtime_factory: Callable[[Runtime], RuntimeClient] = RuntimeClient,
                 on_final: Callable[[dict], None] | None = None,
                 clock: Callable[[], float] = time.time):
        self.store = store
        self.settings = settings
        self.tunnel = tunnel or ColabTunnel()
        self.runtime_factory = runtime_factory
        self.on_final = on_final
        self.clock = clock
        self._clients: dict[str, RuntimeClient] = {}
        self._keepalive: dict[str, float] = {}
        self._liveness: dict[str, float] = {}
        self._unreachable: dict[str, int] = {}
        self._next_try: dict[str, float] = {}
        self._gpu_tries: dict[str, int] = {}
        self._wait_since: dict[str, float] = {}
        self._last_tick: dict[str, float] = {}
        self._seen: dict[str, dict[str, list]] = {}

    # ── Driver ────────────────────────────────────────────────────────────────────────────────────
    def tick(self) -> int:
        runs = self.store.list(states=ACTIVE, limit=100)
        for run in runs:
            if self._next_try.get(run["id"], 0) > self.clock():
                continue
            self.step(run)
        return len(runs)

    def step(self, run: dict) -> None:
        run_id = run["id"]
        try:
            self._account(run)
            handler = getattr(self, f"_on_{run['state']}")
            handler(run)
            self._unreachable.pop(run_id, None)
        except NotSignedIn as exc:
            reason = str(exc).rstrip(". ")
            hint = "" if "hermes auth add" in reason else " Run `hermes auth add hermes-colab:google` in a terminal."
            self._finish(run, "failed", error=f"Not signed in to Google through Hermes: {reason}.{hint} "
                         "Then resume the run with colab_resume.")
        except TooManyRuntimes as exc:
            self._wait_gpu(run, str(exc))
        except RuntimeGone as exc:
            self._lost(run, str(exc))
        except (RuntimeUnreachable, KernelError) as exc:
            count = self._unreachable.get(run_id, 0) + 1
            self._unreachable[run_id] = count
            logger.info("hermes-colab run %s: transient failure %d/%d: %s", run_id, count, UNREACHABLE_LIMIT, exc)
            if isinstance(exc, KernelError) and run["state"] in ("uploading", "starting"):
                self._finish(run, "failed", error=f"Could not prepare the runtime: {exc}")
            elif count >= UNREACHABLE_LIMIT:
                if run.get("endpoint") and not self._still_assigned(run["endpoint"]):
                    self._lost(run, f"runtime unreachable ({exc}) and no longer assigned")
                else:
                    self.store.event(run_id, f"Runtime not answering yet ({exc}); retrying")
                    self._unreachable[run_id] = 0
        except ColabError as exc:
            if run["state"] in ("queued", "waiting_gpu", "resuming") and exc.status in (0, 429, 500, 503, 400):
                self._wait_gpu(run, str(exc))
            else:
                self._finish(run, "failed", error=str(exc))
        except Exception as exc:  # a bug must not kill the supervisor loop or other runs
            logger.exception("hermes-colab run %s: unexpected error", run_id)
            self._finish(run, "failed", error=f"internal error: {type(exc).__name__}: {exc}")

    # ── States ────────────────────────────────────────────────────────────────────────────────────
    def _on_queued(self, run: dict) -> None:
        self._assign(run)

    _on_waiting_gpu = _on_queued
    _on_resuming = _on_queued
    _on_assigning = _on_queued

    def _assign(self, run: dict) -> None:
        self.store.update(run["id"], state="assigning")
        runtime = self.tunnel.assign(run["accelerator"], high_mem=bool(run["spec"].get("high_mem")))
        self._clients[run["id"]] = self.runtime_factory(runtime)
        self._keepalive[run["id"]] = self.clock()
        self._gpu_tries.pop(run["id"], None)
        self._wait_since.pop(run["id"], None)
        attempt = run["attempt"] + 1
        self.store.update(run["id"], state="uploading", endpoint=runtime.endpoint, attempt=attempt, log_offset=0)
        self.store.event(run["id"], f"Got a {runtime.accelerator} runtime (attempt {attempt}); uploading")

    def _on_uploading(self, run: dict) -> None:
        client = self._client(run)
        local = Path(run["local_dir"])
        run_dir = client.prepare(run["id"])
        client.upload(f"{run_dir}/.incoming/bundle.zip", (local / "bundle.zip").read_bytes())
        client.unzip(f"{run_dir}/.incoming/bundle.zip", f"{run_dir}/work")
        outputs = local / "outputs"
        if run["attempt"] > 1 and outputs.is_dir() and any(outputs.rglob("*")):
            data = zip_dir(outputs)
            client.upload(f"{run_dir}/.incoming/outputs.zip", data)
            restored = client.unzip(f"{run_dir}/.incoming/outputs.zip", f"{run_dir}/outputs")
            self.store.event(run["id"], f"Restored {restored} synced output files for resume")
        import json

        spec = {**run["spec"], "run_id": run["id"], "attempt": run["attempt"]}
        client.upload(f"{run_dir}/spec.json", json.dumps(spec).encode())
        client.upload(f"{run_dir}/runner.py", RUNNER.read_bytes())
        self.store.update(run["id"], state="starting")

    def _on_starting(self, run: dict) -> None:
        client = self._client(run)
        client.launch(client.run_dir(run["id"]))
        changes: dict[str, Any] = {"state": "running"}
        if not run.get("started_at"):
            changes["started_at"] = self.clock()
        self.store.update(run["id"], **changes)
        self.store.event(run["id"], "Running" if run["attempt"] == 1 else f"Resumed (attempt {run['attempt']})")

    def _on_running(self, run: dict) -> None:
        client = self._client(run)
        now = self.clock()
        if now - self._keepalive.get(run["id"], 0) >= KEEPALIVE_EVERY:
            if not self.tunnel.keep_alive(run["endpoint"]):
                raise RuntimeGone("Colab reports the runtime is gone")
            self._keepalive[run["id"]] = now
        if now - self._liveness.get(run["id"], 0) >= LIVENESS_EVERY:
            self._liveness[run["id"]] = now
            try:
                listed = {r.endpoint for r in self.tunnel.list_runtimes()}
            except ColabError:
                listed = None  # cannot tell; the keep-alive and the kernel calls still catch a dead VM
            if listed is not None and run["endpoint"] not in listed:
                raise RuntimeGone("Colab no longer lists the runtime as assigned")
        limit = float(run["spec"].get("timeout_minutes") or 0) * 60
        if limit and run.get("started_at") and now - run["started_at"] > limit:
            self.store.event(run["id"], f"Time limit of {run['spec']['timeout_minutes']} min reached; cancelling")
            self.store.update(run["id"], state="cancelling", error="time limit reached")
            return
        state = client.poll(client.run_dir(run["id"]), run["log_offset"])
        self._append_log(run, state)
        status = state.get("status") or {}
        changes: dict[str, Any] = {"log_offset": state["log_offset"]}
        if status.get("progress") is not None:
            changes["progress"] = status["progress"]
        if status.get("gpu") is not None:
            changes["gpu"] = status["gpu"]
        self.store.update(run["id"], **changes)
        finished = status.get("state") == "finished"
        self._sync(run, client, state.get("outputs") or [], final=finished, now=state.get("now"))
        if finished:
            code = status.get("exit_code")
            ok = code == 0
            error = None
            if not ok:
                error = status.get("error") or f"exit code {code}"
                cause = _failure_line(Path(run["local_dir"]) / "run.log")
                if cause:
                    error = f"{error}: {cause}"
            self._finish(run, "succeeded" if ok else "failed", exit_code=code, error=error)
        elif not status and state.get("runner_err"):
            self._finish(run, "failed", error="runner did not start: " + state["runner_err"][-500:])

    def _on_cancelling(self, run: dict) -> None:
        if run.get("endpoint"):
            try:
                client = self._client(run)
                client.signal_runner(client.run_dir(run["id"]))
                state = client.poll(client.run_dir(run["id"]), run["log_offset"])
                self._append_log(run, state)
                self._sync(run, client, state.get("outputs") or [], final=True)
            except (RuntimeGone, RuntimeUnreachable, KernelError) as exc:
                self.store.event(run["id"], f"Could not reach the runtime while cancelling: {exc}")
        self._finish(run, "cancelled", error=run.get("error") or "cancelled")

    # ── Helpers ───────────────────────────────────────────────────────────────────────────────────
    def _client(self, run: dict) -> RuntimeClient:
        client = self._clients.get(run["id"])
        if client is not None:
            return client
        runtime = next((r for r in self.tunnel.list_runtimes() if r.endpoint == run["endpoint"]), None)
        if runtime is None:
            raise RuntimeGone("the run's runtime is no longer assigned to this account")
        client = self._clients[run["id"]] = self.runtime_factory(runtime)
        return client

    def _still_assigned(self, endpoint: str) -> bool:
        try:
            return any(r.endpoint == endpoint for r in self.tunnel.list_runtimes())
        except ColabError:
            return True  # cannot tell: do not declare the runtime lost on a Colab hiccup

    def _account(self, run: dict) -> None:
        now = self.clock()
        last = self._last_tick.get(run["id"])
        self._last_tick[run["id"]] = now
        if last is not None and run.get("endpoint") and run["state"] not in ("queued", "waiting_gpu", "resuming"):
            run["runtime_seconds"] = run["runtime_seconds"] + (now - last)
            self.store.update(run["id"], runtime_seconds=run["runtime_seconds"])

    def _append_log(self, run: dict, state: Mapping[str, Any]) -> None:
        text = state.get("log") or ""
        if text:
            with open(Path(run["local_dir"]) / "run.log", "a", encoding="utf-8") as fh:
                fh.write(text)

    def _sync(self, run: dict, client: RuntimeClient, listing: list, *, final: bool,
              now: float | None = None) -> None:
        """Copy new or changed output files home. A file is copied once it has not changed for
        QUIET_SECONDS by the runtime's own clock (so a checkpoint is not read mid-write, yet a file
        rewritten every few seconds still comes home), once its size and mtime held still across two
        polls, or at once when the run finished."""
        synced: dict[str, list] = dict(run.get("synced") or {})
        seen = self._seen.setdefault(run["id"], {})
        max_bytes = int(self.settings().get("max_sync_mb", 1024)) * 1024 * 1024
        out_dir = Path(run["local_dir"]) / "outputs"
        remote = client.run_dir(run["id"]) + "/outputs"
        changed = False
        # Oldest first: a job that writes its weights and then its progress file never has the
        # progress file arrive home ahead of the weights it describes.
        for rel, size, mtime in sorted(listing, key=lambda item: float(item[2])):
            rel_path = PurePosixPath(rel)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                continue
            sig = [size, mtime]
            if synced.get(rel) == sig:
                continue
            quiet = now is not None and now - float(mtime) >= QUIET_SECONDS
            stable = quiet or seen.get(rel) == sig
            seen[rel] = sig
            if not (stable or final):
                continue
            if size > max_bytes:
                if synced.get(rel) != ["skipped", size]:
                    synced[rel] = ["skipped", size]
                    changed = True
                    self.store.event(run["id"], f"Skipped {rel} ({size // 2**20} MB > max_sync_mb)")
                continue
            data = client.download(f"{remote}/{rel}", size)
            target = out_dir.joinpath(*rel_path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            tmp.write_bytes(data)
            os.replace(tmp, target)
            synced[rel] = sig
            changed = True
        if changed:
            self.store.update(run["id"], synced=synced)
            run["synced"] = synced

    def _wait_gpu(self, run: dict, reason: str) -> None:
        tries = self._gpu_tries.get(run["id"], 0)
        since = self._wait_since.setdefault(run["id"], self.clock())
        limit = float(self.settings().get("wait_for_gpu_minutes", 30)) * 60
        if self.clock() - since > limit:
            self._finish(run, "failed", error=f"No {run['accelerator']} runtime within "
                         f"{int(limit // 60)} min: {reason}")
            return
        delay = GPU_RETRY[min(tries, len(GPU_RETRY) - 1)]
        self._gpu_tries[run["id"]] = tries + 1
        self._next_try[run["id"]] = self.clock() + delay
        self.store.update(run["id"], state="waiting_gpu")
        self.store.event(run["id"], f"Waiting for a {run['accelerator']} runtime ({reason}); retry in {delay}s")

    def _salvage(self, run: dict) -> dict:
        """A reclaimed VM often keeps answering for a minute: copy what it wrote since the last poll
        before resuming elsewhere, so the fresh runtime starts from the newest checkpoint."""
        client = self._clients.get(run["id"])
        if client is None or run["state"] != "running":
            return {}
        try:
            state = client.poll(client.run_dir(run["id"]), run["log_offset"], timeout=20)
            self._append_log(run, state)
            self.store.update(run["id"], log_offset=state["log_offset"])
            self._sync(run, client, state.get("outputs") or [], final=True, now=state.get("now"))
        except (RuntimeGone, RuntimeUnreachable, KernelError, OSError):
            return {}
        self.store.event(run["id"], "Copied the latest outputs from the reclaimed runtime")
        return state.get("status") or {}

    def _lost(self, run: dict, reason: str) -> None:
        status = self._salvage(run)
        self._drop(run["id"])
        settings = self.settings()
        if run["state"] == "cancelling":
            self._finish(run, "cancelled", error="cancelled")
            return
        if status.get("state") == "finished" and status.get("exit_code") == 0:
            self.store.event(run["id"], "The job had already finished on the reclaimed runtime")
            self._finish(run, "succeeded", exit_code=0)
            return
        if settings.get("auto_resume", True) and run["resumes"] < int(settings.get("max_resumes", 3)):
            self.store.update(run["id"], state="resuming", endpoint=None, resumes=run["resumes"] + 1)
            self.store.event(run["id"], f"Runtime lost ({reason}); resuming on a fresh runtime from the "
                             f"last synced outputs ({run['resumes'] + 1}/{settings.get('max_resumes', 3)})")
        else:
            self._finish(run, "failed", error=f"Runtime lost ({reason}); no resume attempts left")

    def _drop(self, run_id: str) -> None:
        client = self._clients.pop(run_id, None)
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        self._keepalive.pop(run_id, None)
        self._liveness.pop(run_id, None)
        self._seen.pop(run_id, None)

    def _finish(self, run: dict, state: str, *, exit_code: int | None = None, error: str | None = None) -> None:
        endpoint = run.get("endpoint")
        self._drop(run["id"])
        if endpoint and not run["spec"].get("keep_runtime"):
            try:
                self.tunnel.release(endpoint)
            except ColabError as exc:
                logger.info("hermes-colab: could not release runtime %s: %s", endpoint, exc)
        final = self.store.update(run["id"], state=state, exit_code=exit_code, error=error,
                                  finished_at=self.clock(),
                                  endpoint=endpoint if run["spec"].get("keep_runtime") else None)
        verb = {"succeeded": "Finished", "failed": "Failed", "cancelled": "Cancelled"}[state]
        self.store.event(run["id"], f"{verb}" + (f": {error}" if error and state != "succeeded" else ""))
        if self.on_final is not None:
            try:
                self.on_final(self.store.get(run["id"]) or final)
            except Exception:
                logger.exception("hermes-colab: completion notification failed")


_ERROR_LINE = re.compile(r"^\s*(\w+(Error|Exception|Interrupt)|AssertionError|Killed|CUDA out of memory)\b")


def _failure_line(log: Path) -> str:
    """The line that explains a failed job (last exception line of the log), trimmed for chat."""
    try:
        with open(log, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 32 * 1024))
            lines = [ln.strip() for ln in fh.read().decode("utf-8", "replace").splitlines() if ln.strip()]
    except OSError:
        return ""
    hits = [ln for ln in lines if _ERROR_LINE.match(ln) or "out of memory" in ln]
    line = hits[-1] if hits else (lines[-1] if lines else "")
    return line[:300]


class SupervisorThread:
    """Daemon loop with a store lease; started on demand, exits when no run is active."""

    def __init__(self, supervisor: Supervisor, *, home_scope: Callable[[], Any], interval: float = 10.0):
        self.supervisor = supervisor
        self.home_scope = home_scope
        self.interval = interval
        self.owner = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._atexit = False

    def _release_on_exit(self) -> None:
        try:
            with self.home_scope():
                self.supervisor.store.release_lease(self.owner)
        except Exception:
            pass

    def ensure_running(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="hermes-colab-supervisor", daemon=True)
            self._thread.start()
            if not self._atexit:
                atexit.register(self._release_on_exit)  # a restarted Hermes takes over at once
                self._atexit = True

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5)
        try:
            self.supervisor.store.release_lease(self.owner)
        except Exception:
            pass

    def _run(self) -> None:
        with self.home_scope():
            idle = 0
            while not self._stop.is_set():
                try:
                    if self.supervisor.store.acquire_lease(self.owner):
                        active = self.supervisor.tick()
                        idle = 0 if active else idle + 1
                    else:
                        idle = 0
                except Exception:
                    logger.exception("hermes-colab supervisor tick failed")
                if idle >= 3:
                    self.supervisor.store.release_lease(self.owner)
                    return
                self._stop.wait(self.interval)
