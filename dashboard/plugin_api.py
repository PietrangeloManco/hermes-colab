"""Backend routes for the hermes-colab Desktop page: /api/plugins/hermes-colab/*.

Read-only views of the run store plus cancel / resume, the same operations the agent tools
perform. Nothing here returns credentials: runs hold no token, and the Google grant stays in
Hermes' auth store.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter()
_ROOT = Path(__file__).resolve().parents[1]


def _package():
    """The plugin package as Hermes loaded it (same supervisor), else load it by path."""
    init = (_ROOT / "__init__.py").resolve()
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if file and Path(file).resolve() == init and hasattr(module, "service"):
            return module
    spec = importlib.util.spec_from_file_location("hermes_colab_dashboard", init,
                                                  submodule_search_locations=[str(_ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["hermes_colab_dashboard"] = module
    spec.loader.exec_module(module)
    return module


def _tools():
    return _package().service().tools


def _json(raw: str) -> dict:
    data = json.loads(raw)
    if not data.get("ok"):
        raise HTTPException(status_code=400, detail=data.get("error") or "failed")
    return data


@router.get("/runs")
async def runs():
    tools = _tools()
    return {"runs": [tools._view(r) for r in tools.store.list(limit=30)]}


@router.get("/runs/{run_id}")
async def run(run_id: str):
    tools = _tools()
    data = _json(tools.colab_status({"run": run_id}))
    logs = json.loads(tools.colab_logs({"run": run_id, "lines": 200}))
    data["run"]["log"] = logs.get("lines") or []
    return data


@router.post("/runs/{run_id}/cancel")
async def cancel(run_id: str):
    return _json(_tools().colab_cancel({"run": run_id}))


@router.post("/runs/{run_id}/resume")
async def resume(run_id: str):
    return _json(_tools().colab_resume({"run": run_id}))


@router.get("/doctor")
async def doctor():
    return json.loads(_tools().colab_doctor({}))
