"""hermes-colab: run scripts and notebooks on Google Colab runtimes from Hermes."""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator
from typing import Any

from .store import ACTIVE, Store
from .supervisor import Supervisor, SupervisorThread
from .tools import SCHEMAS, Tools

logger = logging.getLogger(__name__)

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


def register(ctx) -> None:
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
    from plugins.plugin_storage import plugin_data_dir

    home = get_hermes_home()
    store = Store(plugin_data_dir("hermes-colab") / "runs.db")

    def settings() -> dict[str, Any]:
        return {key: ctx.get_config(key, default=value) for key, value in DEFAULTS.items()}

    @contextlib.contextmanager
    def home_scope() -> Iterator[None]:
        token = set_hermes_home_override(home)
        try:
            yield
        finally:
            reset_hermes_home_override(token)

    def cwd() -> str:
        with contextlib.suppress(Exception):
            value = ctx.session_cwd()
            if value:
                return value
        return os.getcwd()

    tools: Tools

    def on_final(run: dict) -> None:
        tools.push_notice(run, ctx.inject_message)

    supervisor = Supervisor(store, settings=settings, on_final=on_final)
    thread = SupervisorThread(supervisor, home_scope=home_scope)
    tools = Tools(store=store, supervisor=supervisor, ensure_supervisor=thread.ensure_running,
                  settings=settings, cwd=cwd, env=_secret)

    for name, schema in SCHEMAS.items():
        ctx.register_tool(name=name, toolset=TOOLSET, schema=schema, handler=getattr(tools, name),
                          description=schema["description"], emoji=EMOJI.get(name, ""))

    def pre_llm_call(session_id: str = "", **_: Any):
        notes = tools.pending_notices(session_id)
        return {"context": "\n".join(notes)} if notes else None

    ctx.register_hook("pre_llm_call", pre_llm_call)
    ctx.on_unload(thread.stop)

    # Runs survive a Hermes restart: pick supervision back up if any run is still active.
    if store.list(states=ACTIVE, limit=1):
        thread.ensure_running()
