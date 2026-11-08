"""Tool schemas and handlers. Every handler returns a JSON string and never raises."""

from __future__ import annotations

import contextvars
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .bundle import BundleError, build_bundle
from .colab_api import ACCELERATORS, ColabError, NotSignedIn
from .paths import PathPolicy, child, excluded, folder, read_bytes, write_new
from .store import ACTIVE, FINAL, Store, new_run_id

# Clients whose chats Hermes addresses by session key (the TUI/Desktop host of ``inject_message``).
LOCAL_SURFACES = ("desktop", "tui", "cli")
PROGRESS_PREFIX = "HERMES_PROGRESS "
STATE_LABEL = {
    "queued": "queued", "waiting_gpu": "waiting for a GPU", "assigning": "requesting a runtime",
    "uploading": "uploading code", "starting": "starting", "running": "running", "resuming": "resuming",
    "cancelling": "cancelling", "succeeded": "succeeded", "failed": "failed", "cancelled": "cancelled",
}


def display_path(path: Path) -> str:
    """A folder as the user knows it: under the home directory as ``~/...`` (for chat and UI text)."""
    try:
        return "~/" + path.relative_to(Path.home()).as_posix()
    except ValueError:
        return str(path)


def _ok(**data: Any) -> str:
    return json.dumps({"ok": True, **data}, default=str)


def _err(message: str, **data: Any) -> str:
    return json.dumps({"ok": False, "error": message, **data}, default=str)


_RUN_REF = {"type": "string", "description": "Run id (or unique id prefix, or run name). Omit for the latest run."}

SCHEMAS: dict[str, dict] = {
    "colab_doctor": {
        "name": "colab_doctor",
        "description": "Check that Google Colab runs can work: OAuth client variables, the Hermes Google "
                       "sign-in, Colab access, assigned runtimes. Returns each check with the exact next "
                       "step. Call it first when a Colab tool reports a setup problem.",
        "parameters": {"type": "object", "properties": {}},
    },
    "colab_run": {
        "name": "colab_run",
        "description": "Run a Python script or a Jupyter notebook on a Google Colab runtime (GPU or TPU) in "
                       "the background and bring its outputs home. Uploads the file (or the directory with "
                       "its entry point), installs requirements, keeps the runtime alive, syncs files the job "
                       "writes to $HERMES_COLAB_OUTPUT_DIR back to the local run folder while it runs, and "
                       "if Colab reclaims the runtime, continues on a fresh one with HERMES_COLAB_RESUME=1 and "
                       "the synced outputs restored (so a script that saves checkpoints there resumes). "
                       "Returns at once with a run id; follow it with colab_status / colab_logs.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "A .py or .ipynb file, or a project directory "
                         "(then set entry). Relative paths resolve against the session's working directory."},
                "entry": {"type": "string", "description": "Entry point inside the directory, e.g. train.py "
                          "or notebooks/eval.ipynb. Not needed when path is a file."},
                "accelerator": {"type": "string", "enum": list(ACCELERATORS),
                                "description": "Runtime accelerator; default from settings (T4: free tier)."},
                "args": {"type": "array", "items": {"type": "string"},
                         "description": "Command-line arguments for a script entry point."},
                "requirements": {"type": "array", "items": {"type": "string"},
                                 "description": "Extra pip requirements (a requirements.txt next to the entry "
                                                "point is installed automatically)."},
                "env": {"type": "object", "additionalProperties": {"type": "string"},
                        "description": "Non-secret environment variables for the job."},
                "name": {"type": "string", "description": "Short human name for the run."},
                "timeout_minutes": {"type": "integer", "minimum": 1,
                                    "description": "Cancel the job after this many minutes of running."},
                "keep_runtime": {"type": "boolean", "description": "Keep the runtime assigned after the job "
                                 "ends (it keeps using compute units). Default false."},
                "high_mem": {"type": "boolean", "description": "Ask for the high-RAM shape when available."},
                "output_dir": {"type": "string", "description": "Local folder for outputs and the log; "
                               "default ./colab_runs/<run id> next to the code. Must be inside the chat's "
                               "working directory or the folder that holds the code."},
            },
            "required": ["path"],
        },
    },
    "colab_status": {
        "name": "colab_status",
        "description": "Show one Colab run (state, progress, GPU use, elapsed time, estimated compute units, "
                       "synced outputs, recent events) or, with all=true, the recent runs.",
        "parameters": {"type": "object", "properties": {
            "run": _RUN_REF,
            "all": {"type": "boolean", "description": "List recent runs instead of one."}}},
    },
    "colab_logs": {
        "name": "colab_logs",
        "description": "Return the last lines of a Colab run's combined stdout/stderr (synced locally). "
                       "HERMES_PROGRESS lines are left out: colab_status reports progress.",
        "parameters": {"type": "object", "properties": {
            "run": _RUN_REF,
            "lines": {"type": "integer", "minimum": 1, "maximum": 400, "description": "Default 60."}}},
    },
    "colab_cancel": {
        "name": "colab_cancel",
        "description": "Stop a Colab run: terminate the job, sync its outputs and release the runtime.",
        "parameters": {"type": "object", "properties": {"run": _RUN_REF}, "required": ["run"]},
    },
    "colab_resume": {
        "name": "colab_resume",
        "description": "Restart a failed or cancelled Colab run on a fresh runtime, restoring its synced "
                       "outputs and setting HERMES_COLAB_RESUME=1.",
        "parameters": {"type": "object", "properties": {"run": _RUN_REF}, "required": ["run"]},
    },
    "colab_runtimes": {
        "name": "colab_runtimes",
        "description": "List the Colab runtimes currently assigned to the account and which run uses each.",
        "parameters": {"type": "object", "properties": {}},
    },
    "colab_release_runtime": {
        "name": "colab_release_runtime",
        "description": "Release (unassign) one Colab runtime by endpoint, freeing its compute. Refuses a "
                       "runtime an active run is using unless that run is cancelled first.",
        "parameters": {"type": "object", "properties": {"endpoint": {"type": "string"}}, "required": ["endpoint"]},
    },
}


class Tools:
    def __init__(self, *, store: Store, supervisor, ensure_supervisor: Callable[[], None],
                 settings: Callable[[], dict], cwd: Callable[[], str], env: Callable[[str], str | None],
                 surface: Callable[[], str] = lambda: "", paths: PathPolicy | None = None):
        self.store = store
        self.supervisor = supervisor
        self.ensure_supervisor = ensure_supervisor
        self.settings = settings
        self.cwd = cwd
        self.env = env
        self.paths = paths or PathPolicy()
        # Client of the calling chat (``ctx.session_surface()``): Desktop and TUI chats get the finish
        # notice by their session key, messaging chats through the keyless route below.
        self.surface = surface
        # Calling chat's context per run, so the finish notice of a messaging chat is injected for
        # that chat through the public keyless ``ctx.inject_message`` route (in memory only; after a
        # restart the pre_llm_call fallback delivers the summary instead).
        self.contexts: dict[str, contextvars.Context] = {}

    # ── helpers ───────────────────────────────────────────────────────────────────────────────────
    def _resolve(self, raw: str) -> Path:
        path = Path(os.path.expanduser(raw))
        if not path.is_absolute():
            path = Path(self.cwd()) / path
        path = self.paths.check_source(path)
        if not path.exists():
            raise BundleError(f"{path} does not exist")
        return path

    def _run(self, ref: str | None) -> dict | None:
        if ref:
            return self.store.find(ref)
        runs = self.store.list(limit=1)
        return runs[0] if runs else None

    def _view(self, run: dict, *, detail: bool = True) -> dict[str, Any]:
        rates = self.settings().get("compute_unit_rates") or {}
        hours = (run.get("runtime_seconds") or 0) / 3600
        end = run.get("finished_at") or time.time()
        view: dict[str, Any] = {
            "id": run["id"], "name": run["name"], "state": run["state"],
            "state_label": STATE_LABEL.get(run["state"], run["state"]),
            "accelerator": run["accelerator"], "attempt": run["attempt"], "resumes": run["resumes"],
            "progress": run.get("progress"), "message": run.get("message"),
            "elapsed_minutes": round((end - (run.get("started_at") or run["created_at"])) / 60, 1),
            "runtime_hours": round(hours, 3),
            "compute_units_estimate": round(hours * float(rates.get(run["accelerator"], 0) or 0), 2),
            "local_dir": run["local_dir"], "exit_code": run.get("exit_code"), "error": run.get("error"),
        }
        if detail:
            synced = run.get("synced") or {}
            view["gpu"] = run.get("gpu")
            view["outputs"] = sorted(k for k, v in synced.items() if v and v[0] != "skipped")[:50]
            view["events"] = [e[1] for e in (run.get("events") or [])[-8:]]
        return view

    # ── handlers ──────────────────────────────────────────────────────────────────────────────────
    def colab_doctor(self, args: dict, **_: Any) -> str:
        checks: list[dict[str, Any]] = []
        missing = [n for n in ("HERMES_COLAB_CLIENT_ID", "HERMES_COLAB_CLIENT_SECRET") if not self.env(n)]
        checks.append({"check": "OAuth client variables", "ok": not missing,
                       "detail": "set" if not missing else f"missing: {', '.join(missing)}",
                       "next_step": None if not missing else
                       "Create a Google Cloud OAuth client of type 'Desktop app' (README step 1) and set "
                       "HERMES_COLAB_CLIENT_ID / HERMES_COLAB_CLIENT_SECRET in Settings → Plugins → hermes-colab "
                       "or with `hermes config set HERMES_COLAB_CLIENT_ID ...`."})
        runtimes = None
        if not missing:
            try:
                runtimes = self.supervisor.tunnel.list_runtimes()
                checks.append({"check": "Google sign-in and Colab access", "ok": True,
                               "detail": f"{len(runtimes)} runtime(s) assigned"})
            except NotSignedIn as exc:
                checks.append({"check": "Google sign-in and Colab access", "ok": False, "detail": str(exc),
                               "next_step": "Run `hermes auth add hermes-colab:google` in your own terminal "
                                            "and sign in with the Google account you use for Colab."})
            except ColabError as exc:
                checks.append({"check": "Google sign-in and Colab access", "ok": False, "detail": str(exc),
                               "next_step": "Open colab.research.google.com once with that account and accept "
                                            "the terms; check the OAuth consent screen lists your account as a "
                                            "test user."})
        active = self.store.list(states=ACTIVE)
        checks.append({"check": "Active runs", "ok": True, "detail": f"{len(active)} active"})
        if active:
            self.ensure_supervisor()
        ready = all(c["ok"] for c in checks)
        return _ok(ready=ready, checks=checks,
                   runtimes=[r.public() for r in runtimes] if runtimes is not None else None)

    def colab_run(self, args: dict, session_id: str | None = None, **_: Any) -> str:
        try:
            settings = self.settings()
            source = self._resolve(str(args.get("path") or ""))
            entry = args.get("entry") or (source.name if source.is_file() else "")
            if not entry:
                for candidate in ("main.py", "train.py", "run.py"):
                    if (source / candidate).is_file():
                        entry = candidate
                        break
            if not entry:
                return _err("path is a directory: pass entry (the .py or .ipynb to run)")
            if not entry.endswith((".py", ".ipynb")):
                return _err("entry must be a .py script or a .ipynb notebook")
            accelerator = str(args.get("accelerator") or settings.get("default_accelerator") or "T4").upper()
            if accelerator not in ACCELERATORS:
                return _err(f"unknown accelerator {accelerator}")
            max_bytes = int(settings.get("max_upload_mb", 512)) * 2**20
            data, files = build_bundle(source, entry, max_bytes=max_bytes,
                                       skip=excluded(self.paths.private_roots()))
            run_id = new_run_id()
            code_dir = source.parent if source.is_file() else source
            local = self.paths.output_dir(args.get("output_dir"), code_dir=code_dir, cwd=Path(self.cwd()),
                                          run_id=run_id)
            folder(local, "outputs")
            write_new(child(local, "bundle.zip"), data)
            spec = {
                "entry": entry, "args": [str(a) for a in args.get("args") or []],
                "requirements": [str(r) for r in args.get("requirements") or []],
                "env": {str(k): str(v) for k, v in (args.get("env") or {}).items()},
                "timeout_minutes": args.get("timeout_minutes"), "keep_runtime": bool(args.get("keep_runtime")),
                "high_mem": bool(args.get("high_mem")), "source": str(source), "surface": self._surface(),
            }
            name = str(args.get("name") or Path(entry).stem)[:60]
            run = self.store.create({"id": run_id, "name": name, "spec": spec, "local_dir": str(local),
                                     "accelerator": accelerator, "session_id": session_id})
            self.store.event(run_id, f"Queued {entry} ({len(files)} files, {len(data) / 2**20:.1f} MB zipped)")
            self.contexts[run_id] = contextvars.copy_context()
            self.ensure_supervisor()
            return _ok(run=self._view(self.store.get(run_id) or run, detail=False), files=len(files),
                       hint="Colab runs in the background. Check with colab_status; outputs land in local_dir.")
        except BundleError as exc:
            return _err(str(exc))
        except Exception as exc:
            return _err(f"{type(exc).__name__}: {exc}")

    def colab_status(self, args: dict, **_: Any) -> str:
        if args.get("all"):
            runs = self.store.list(limit=20)
            return _ok(runs=[self._view(r, detail=False) for r in runs])
        run = self._run(args.get("run"))
        if run is None:
            return _err("no such run" if args.get("run") else "no runs yet")
        if run["state"] in ACTIVE:
            self.ensure_supervisor()
        return _ok(run=self._view(run))

    def colab_logs(self, args: dict, **_: Any) -> str:
        run = self._run(args.get("run"))
        if run is None:
            return _err("no such run")
        lines = max(1, min(int(args.get("lines") or 60), 400))
        try:
            log = self.paths.check_recorded(Path(run["local_dir"])) / "run.log"
        except BundleError as exc:
            return _err(str(exc))
        if log.is_symlink() or not log.is_file():
            return _ok(id=run["id"], state=run["state"], lines=[], note="no output yet")
        try:
            data = read_bytes(log, limit=256 * 1024)
        except BundleError as exc:
            return _err(str(exc))
        tail = [ln for ln in data.decode("utf-8", "replace").splitlines()
                if not ln.startswith(PROGRESS_PREFIX)][-lines:]  # progress is in colab_status
        return _ok(id=run["id"], state=run["state"], lines=tail)

    def colab_cancel(self, args: dict, **_: Any) -> str:
        run = self._run(args.get("run"))
        if run is None:
            return _err("no such run")
        if run["state"] in FINAL:
            return _err(f"run already {run['state']}")
        if run["state"] in ("queued", "waiting_gpu", "resuming") or not run.get("endpoint"):
            self.store.update(run["id"], state="cancelled", finished_at=time.time(), error="cancelled")
            self.store.event(run["id"], "Cancelled before a runtime was assigned")
        else:
            self.store.update(run["id"], state="cancelling")
            self.store.event(run["id"], "Cancellation requested")
            self.ensure_supervisor()
        return _ok(run=self._view(self.store.get(run["id"]) or run, detail=False))

    def colab_resume(self, args: dict, session_id: str | None = None, **_: Any) -> str:
        run = self._run(args.get("run"))
        if run is None:
            return _err("no such run")
        if run["state"] not in ("failed", "cancelled"):
            return _err(f"only failed or cancelled runs can be resumed (this one is {run['state']})")
        try:
            self.paths.check_recorded(Path(run["local_dir"]))
        except BundleError as exc:
            return _err(str(exc))
        self.store.update(run["id"], state="resuming", error=None, exit_code=None, finished_at=None,
                          endpoint=None, notified=0, session_id=session_id or run.get("session_id"),
                          spec={**(run.get("spec") or {}), "surface": self._surface()})
        self.store.event(run["id"], "Manual resume requested")
        self.contexts[run["id"]] = contextvars.copy_context()
        self.ensure_supervisor()
        return _ok(run=self._view(self.store.get(run["id"]) or run, detail=False))

    def colab_runtimes(self, args: dict, **_: Any) -> str:
        try:
            runtimes = self.supervisor.tunnel.list_runtimes()
        except ColabError as exc:
            return _err(str(exc), next_step="Run colab_doctor.")
        users = {r["endpoint"]: r["id"] for r in self.store.list(limit=100) if r.get("endpoint")}
        return _ok(runtimes=[{**r.public(), "run": users.get(r.endpoint)} for r in runtimes])

    def colab_release_runtime(self, args: dict, **_: Any) -> str:
        endpoint = str(args.get("endpoint") or "")
        busy = [r for r in self.store.list(states=ACTIVE, limit=100) if r.get("endpoint") == endpoint]
        if busy:
            return _err(f"run {busy[0]['id']} is using this runtime; colab_cancel it first")
        try:
            self.supervisor.tunnel.release(endpoint)
        except ColabError as exc:
            return _err(str(exc))
        for run in self.store.list(limit=100):
            if run.get("endpoint") == endpoint:
                self.store.update(run["id"], endpoint=None)
        return _ok(released=endpoint)

    # ── completion notices ────────────────────────────────────────────────────────────────────────
    def summary(self, run: dict) -> str:
        view = self._view(run)
        verbs = {"succeeded": "finished", "failed": "failed", "cancelled": "was cancelled"}
        verb = verbs.get(run["state"], run["state"])
        text = (f"[hermes-colab] Run {run['name']} ({run['id']}) {verb} on {run['accelerator']} after "
                f"{view['elapsed_minutes']} min (~{view['compute_units_estimate']} compute units).")
        if run.get("error") and run["state"] != "succeeded":
            text += f" Error: {run['error']}."
        if view["outputs"]:
            outputs = display_path(Path(run["local_dir"]) / "outputs")
            text += f" Outputs in {outputs}: {', '.join(view['outputs'][:8])}."
        return text

    def pending_notices(self, session_id: str | None) -> list[str]:
        if not session_id:
            return []
        notes = []
        for run in self.store.list(states=FINAL, limit=20):
            if run.get("session_id") == session_id and not run.get("notified"):
                notes.append(self.summary(run))
                self.store.update(run["id"], notified=1)
        return notes

    def push_notice(self, run: dict, inject: Callable[..., bool]) -> bool:
        """Best effort: queue the summary on the chat that started the run.

        A Desktop or TUI chat is addressed by its session key (the stored session id), which needs
        no saved context and so also works after a restart. Any other chat gets a keyless
        ``inject_message`` run inside its copied context, where Hermes routes it to the calling
        session. Needs ``plugins.entries.hermes-colab.allow_gateway_injection: true``; without it,
        or when the chat is gone, the pre_llm_call hook adds the summary to that session's next turn.
        """
        context = self.contexts.pop(run["id"], None)
        if not self.settings().get("notify_on_finish", True):
            return False
        text = self.summary(run) + " (automatic notice from the hermes-colab plugin)"
        session = run.get("session_id")
        surface = str((run.get("spec") or {}).get("surface") or "")
        # A messaging chat names its platform; Desktop/TUI chats report a local surface or none at
        # all (some cores leave it empty there), and their session id is their session key.
        local = bool(session) and (surface in LOCAL_SURFACES or not surface)
        delivered = False
        try:
            if local:
                delivered = bool(inject(text, session_key=session))
            if not delivered and context is not None:
                delivered = bool(context.run(inject, text))
        except Exception:
            return False
        if delivered:
            self.store.update(run["id"], notified=1)
        return delivered

    def _surface(self) -> str:
        try:
            return str(self.surface() or "")
        except Exception:  # outside a session (tests, probes): no surface
            return ""
