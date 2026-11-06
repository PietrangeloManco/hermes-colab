"""hermes-colab: run scripts and notebooks on Google Colab runtimes from Hermes."""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from collections.abc import Callable, Iterator
from typing import Any

from .store import ACTIVE, Store
from .supervisor import Supervisor, SupervisorThread
from .tools import SCHEMAS, Tools

logger = logging.getLogger(__name__)

PLUGIN_ID = "hermes-colab"
TOOLSET = "colab"
EMOJI = {"colab_doctor": "🩺", "colab_run": "🚀", "colab_status": "📊", "colab_logs": "📜",
         "colab_cancel": "🛑", "colab_resume": "🔁", "colab_runtimes": "🖥️", "colab_release_runtime": "♻️"}
DEFAULTS = {"default_accelerator": "T4", "auto_resume": True, "max_resumes": 3, "max_upload_mb": 512,
            "max_sync_mb": 1024, "wait_for_gpu_minutes": 30, "notify_on_finish": True,
            "compute_unit_rates": {"T4": 1.96, "L4": 4.82, "A100": 11.77, "H100": 18.0, "NONE": 0.07}}


def _secret(name: str):
    try:
        from agent.secret_scope import get_secret
    except ImportError:  # pragma: no cover
        return os.environ.get(name)
    try:
        return get_secret(name)
    except Exception:
        return None


def _settings() -> dict[str, Any]:
    """``plugins.entries.hermes-colab.settings`` over the defaults, read fresh so edits apply live."""
    values: dict[str, Any] = {}
    with contextlib.suppress(Exception):
        from hermes_cli.config import load_config_readonly

        entry = (((load_config_readonly() or {}).get("plugins") or {}).get("entries") or {}).get(PLUGIN_ID) or {}
        values = dict(entry.get("settings") or {})
    return {key: values.get(key, default) for key, default in DEFAULTS.items()}


class Service:
    """One per Hermes process and profile: run store, supervisor thread and tool handlers.

    Shared by the agent tools (``register``) and the Desktop page backend (``dashboard/plugin_api.py``).
    """

    def __init__(self) -> None:
        from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
        from plugins.plugin_storage import plugin_data_dir

        home = get_hermes_home()

        @contextlib.contextmanager
        def home_scope() -> Iterator[None]:
            token = set_hermes_home_override(home)
            try:
                yield
            finally:
                reset_hermes_home_override(token)

        self.home = home
        self.store = Store(plugin_data_dir(PLUGIN_ID) / "runs.db")
        self.inject: Callable[..., bool] | None = None
        self.cwd: Callable[[], str] = os.getcwd
        self.surface: Callable[[], str] = lambda: ""
        self.supervisor = Supervisor(self.store, settings=_settings, on_final=self._on_final)
        self.thread = SupervisorThread(self.supervisor, home_scope=home_scope)
        self.tools = Tools(store=self.store, supervisor=self.supervisor, ensure_supervisor=self.thread.ensure_running,
                           settings=_settings, cwd=lambda: self.cwd(), env=_secret, surface=lambda: self.surface())

    def _on_final(self, run: dict) -> None:
        if self.inject is not None:
            self.tools.push_notice(run, self.inject)
        with contextlib.suppress(Exception):
            from hermes_cli.plugin_events import broadcast_plugin_event

            broadcast_plugin_event(PLUGIN_ID, "run.finished", {"id": run["id"], "state": run["state"]})


_services: dict[str, Service] = {}
_lock = threading.Lock()


def service() -> Service:
    from hermes_constants import get_hermes_home

    key = str(get_hermes_home())
    with _lock:
        if key not in _services:
            _services[key] = Service()
        return _services[key]


def register(ctx) -> None:
    svc = service()

    def cwd() -> str:
        with contextlib.suppress(Exception):
            value = ctx.session_cwd()
            if value:
                return value
        return os.getcwd()

    svc.cwd = cwd
    svc.surface = getattr(ctx, "session_surface", None) or svc.surface
    svc.inject = ctx.inject_message
    tools = svc.tools

    for name, schema in SCHEMAS.items():
        ctx.register_tool(name=name, toolset=TOOLSET, schema=schema, handler=getattr(tools, name),
                          description=schema["description"], emoji=EMOJI.get(name, ""))

    def pre_llm_call(session_id: str = "", **_: Any):
        notes = tools.pending_notices(session_id)
        return {"context": "\n".join(notes)} if notes else None

    ctx.register_hook("pre_llm_call", pre_llm_call)
    ctx.on_unload(svc.thread.stop)

    # Runs survive a Hermes restart: pick supervision back up if any run is still active.
    if svc.store.list(states=ACTIVE, limit=1):
        svc.thread.ensure_running()
