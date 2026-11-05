from __future__ import annotations

import json

from hermes_colab.colab_api import NotSignedIn
from hermes_colab.store import Store
from hermes_colab.supervisor import Supervisor
from hermes_colab.tools import Tools


class NoTunnel:
    def __init__(self, exc):
        self.exc = exc

    def list_runtimes(self):
        raise self.exc

    def assign(self, *a, **k):
        raise self.exc


def _tools(tmp_path, *, env=None, tunnel=None, cwd=None):
    store = Store(tmp_path / "runs.db")
    sup = Supervisor(store, settings=lambda: {}, tunnel=tunnel or NoTunnel(NotSignedIn("not signed in")))
    return store, sup, Tools(store=store, supervisor=sup, ensure_supervisor=lambda: None,
                             settings=lambda: {"default_accelerator": "T4", "max_upload_mb": 1},
                             cwd=lambda: str(cwd or tmp_path), env=lambda n: (env or {}).get(n))


def test_doctor_reports_missing_client_then_sign_in(tmp_path):
    _, _, tools = _tools(tmp_path)
    first = json.loads(tools.colab_doctor({}))
    assert not first["ready"] and "README step 1" in first["checks"][0]["next_step"]
    _, _, tools = _tools(tmp_path, env={"HERMES_COLAB_CLIENT_ID": "x", "HERMES_COLAB_CLIENT_SECRET": "y"})
    second = json.loads(tools.colab_doctor({}))
    assert "hermes auth add hermes-colab:google" in second["checks"][1]["next_step"]


def test_not_signed_in_fails_run_with_next_step(tmp_path):
    (tmp_path / "a.py").write_text("print(1)")
    store, sup, tools = _tools(tmp_path)
    run_id = json.loads(tools.colab_run({"path": "a.py"}))["run"]["id"]
    sup.tick()
    run = store.get(run_id)
    assert run["state"] == "failed" and "hermes auth add hermes-colab:google" in run["error"]
    resumed = json.loads(tools.colab_resume({"run": run_id}))
    assert resumed["ok"] and store.get(run_id)["state"] == "resuming"


def test_refuses_protected_directories(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id.py").write_text("x")
    monkeypatch.setenv("HOME", str(home))
    _, _, tools = _tools(tmp_path)
    out = json.loads(tools.colab_run({"path": str(home / ".ssh"), "entry": "id.py"}))
    assert not out["ok"] and "refusing" in out["error"]


def test_rejects_non_python_entry_and_unknown_accelerator(tmp_path):
    (tmp_path / "x.sh").write_text("echo")
    (tmp_path / "a.py").write_text("x")
    _, _, tools = _tools(tmp_path)
    assert "entry must be" in json.loads(tools.colab_run({"path": "x.sh"}))["error"]
    assert "unknown accelerator" in json.loads(tools.colab_run({"path": "a.py", "accelerator": "B200"}))["error"]


def test_cancel_queued_run_needs_no_runtime(tmp_path):
    (tmp_path / "a.py").write_text("x")
    store, _, tools = _tools(tmp_path)
    run_id = json.loads(tools.colab_run({"path": "a.py"}))["run"]["id"]
    assert json.loads(tools.colab_cancel({"run": run_id}))["run"]["state"] == "cancelled"
    assert "already cancelled" in json.loads(tools.colab_cancel({"run": run_id}))["error"]


def test_store_lease_is_exclusive(tmp_path):
    store = Store(tmp_path / "runs.db")
    assert store.acquire_lease("a") and not store.acquire_lease("b") and store.acquire_lease("a")
    store.release_lease("a")
    assert store.acquire_lease("b")


def test_finish_notice_goes_to_the_starting_chat_by_session_key(tmp_path, monkeypatch):
    import hermes_colab.tools as tools_mod

    (tmp_path / "a.py").write_text("x")
    store, _, tools = _tools(tmp_path)
    monkeypatch.setattr(tools_mod, "_session_key", lambda: "ses_abc")
    run_id = json.loads(tools.colab_run({"path": "a.py"}, session_id="s1"))["run"]["id"]
    run = store.update(run_id, state="succeeded", finished_at=store.get(run_id)["created_at"] + 90)
    calls = []
    assert tools.push_notice(run, lambda text, session_key=None: calls.append((text, session_key)) or True)
    assert calls[0][1] == "ses_abc" and "finished" in calls[0][0]
    assert store.get(run_id)["notified"] == 1
    assert tools.pending_notices("s1") == []  # not repeated on the next turn


def test_refused_notice_falls_back_to_next_turn(tmp_path, monkeypatch):
    import hermes_colab.tools as tools_mod

    (tmp_path / "a.py").write_text("x")
    store, _, tools = _tools(tmp_path)
    monkeypatch.setattr(tools_mod, "_session_key", lambda: "ses_abc")
    run_id = json.loads(tools.colab_run({"path": "a.py"}, session_id="s1"))["run"]["id"]
    run = store.update(run_id, state="failed", error="exit code 1")
    assert not tools.push_notice(run, lambda text, session_key=None: False)  # injection not allowed
    notes = tools.pending_notices("s1")
    assert len(notes) == 1 and "failed" in notes[0] and "exit code 1" in notes[0]


def test_old_store_gets_session_key_column(tmp_path):
    import sqlite3

    db = sqlite3.connect(tmp_path / "runs.db")
    db.execute("CREATE TABLE runs (id TEXT PRIMARY KEY, name TEXT NOT NULL, state TEXT NOT NULL, "
               "created_at REAL NOT NULL, updated_at REAL NOT NULL, spec TEXT NOT NULL, local_dir TEXT NOT NULL, "
               "accelerator TEXT NOT NULL)")
    db.commit()
    db.close()
    Store(tmp_path / "runs.db")
    cols = {r[1] for r in sqlite3.connect(tmp_path / "runs.db").execute("PRAGMA table_info(runs)")}
    assert "session_key" in cols
