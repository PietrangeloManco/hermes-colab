from __future__ import annotations

import importlib.util
import json

import pytest

from hermes_colab.store import Store
from hermes_colab.supervisor import Supervisor
from hermes_colab.tools import Tools

fastapi = pytest.importorskip("fastapi")
from conftest import ROOT  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


class _Svc:
    def __init__(self, tools):
        self.tools = tools


def _client(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("hermes_colab_dash_test", ROOT / "dashboard" / "plugin_api.py")
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    store = Store(tmp_path / "runs.db")
    sup = Supervisor(store, settings=lambda: {}, tunnel=object())
    tools = Tools(store=store, supervisor=sup, ensure_supervisor=lambda: None,
                  settings=lambda: {"max_upload_mb": 1}, cwd=lambda: str(tmp_path), env=lambda n: None)
    monkeypatch.setattr(api, "_tools", lambda: tools)
    app = fastapi.FastAPI()
    app.include_router(api.router)
    return TestClient(app), tools


def test_runs_detail_cancel_resume(tmp_path, monkeypatch):
    client, tools = _client(tmp_path, monkeypatch)
    (tmp_path / "a.py").write_text("print(1)")
    run_id = json.loads(tools.colab_run({"path": "a.py"}))["run"]["id"]
    listed = client.get("/runs").json()["runs"]
    assert [r["id"] for r in listed] == [run_id]
    detail = client.get(f"/runs/{run_id}").json()["run"]
    assert detail["state"] == "queued" and detail["log"] == []
    assert client.post(f"/runs/{run_id}/cancel").json()["run"]["state"] == "cancelled"
    assert client.post(f"/runs/{run_id}/resume").json()["run"]["state"] == "resuming"
    assert client.get("/runs/missing").status_code == 400
    assert "token" not in json.dumps(listed).lower()


def test_detail_shows_the_outputs_folder_under_home_as_tilde(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    client, tools = _client(tmp_path, monkeypatch)
    (tmp_path / "a.py").write_text("print(1)")
    run_id = json.loads(tools.colab_run({"path": "a.py"}))["run"]["id"]
    detail = client.get(f"/runs/{run_id}").json()["run"]
    assert detail["outputs_dir"] == f"~/colab_runs/{run_id}/outputs"
    assert detail["local_dir"] == str(tmp_path / "colab_runs" / run_id)


def test_dashboard_manifest_reports_the_plugin_version():
    """The backend lists the dashboard half with the version in its manifest (``0.0.0`` when absent)."""
    import re

    manifest = json.loads((ROOT / "dashboard" / "manifest.json").read_text())
    version = re.search(r"^version:\s*(\S+)", (ROOT / "plugin.yaml").read_text(), re.M).group(1)
    assert manifest.get("version") == version
