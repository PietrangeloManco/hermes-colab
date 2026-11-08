from __future__ import annotations

import json

from hermes_colab.colab_api import NotSignedIn
from hermes_colab.paths import PathPolicy
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


def _tools(tmp_path, *, env=None, tunnel=None, cwd=None, surface="", hermes_home=None):
    store = Store(tmp_path / "runs.db")
    sup = Supervisor(store, settings=lambda: {}, tunnel=tunnel or NoTunnel(NotSignedIn("not signed in")))
    return store, sup, Tools(store=store, supervisor=sup, ensure_supervisor=lambda: None,
                             settings=lambda: {"default_accelerator": "T4", "max_upload_mb": 1},
                             cwd=lambda: str(cwd or tmp_path), env=lambda n: (env or {}).get(n),
                             surface=lambda: surface,
                             paths=PathPolicy(lambda: hermes_home) if hermes_home else None)


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


def test_core_sign_in_hint_is_not_repeated(tmp_path):
    (tmp_path / "a.py").write_text("print(1)")
    core = "Not signed in to google for plugin 'hermes-colab'. Run `hermes auth add hermes-colab:google`."
    store, sup, tools = _tools(tmp_path, tunnel=NoTunnel(NotSignedIn(core)))
    run_id = json.loads(tools.colab_run({"path": "a.py"}))["run"]["id"]
    sup.tick()
    error = store.get(run_id)["error"]
    assert error.count("hermes auth add") == 1 and ".." not in error and "colab_resume" in error


def test_refuses_protected_directories(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id.py").write_text("x")
    monkeypatch.setenv("HOME", str(home))
    _, _, tools = _tools(tmp_path)
    out = json.loads(tools.colab_run({"path": str(home / ".ssh"), "entry": "id.py"}))
    assert not out["ok"] and "refusing" in out["error"]


def test_refuses_to_upload_home_hermes_home_or_a_folder_holding_them(tmp_path, monkeypatch):
    home, hermes_home = tmp_path / "home", tmp_path / "srv" / "hermes"
    for folder in (home, hermes_home):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "main.py").write_text("x")
    _, _, tools = _tools(tmp_path, cwd=home, hermes_home=hermes_home)
    for path in ("~", ".", str(home), str(tmp_path), str(hermes_home), str(hermes_home / "main.py"),
                 str(tmp_path / "srv"), "/"):
        out = json.loads(tools.colab_run({"path": path, "entry": "main.py"}))
        assert not out["ok"] and "refusing" in out["error"], path


def test_output_dir_stays_in_the_session_cwd_or_next_to_the_code(tmp_path, monkeypatch):
    home, hermes_home = tmp_path / "home", tmp_path / "home" / ".hermes" / "profiles" / "p"
    project, other = home / "work" / "proj", home / "other"
    for folder in (hermes_home, home / ".ssh", project, other):
        folder.mkdir(parents=True)
    (project / "a.py").write_text("x")
    (project / "escape").symlink_to(hermes_home)
    store, _, tools = _tools(tmp_path, cwd=project, hermes_home=hermes_home)
    for output_dir in (str(hermes_home / "runs"), "~/.ssh/runs", "~/.hermes/x", "escape/runs", str(other),
                       "../../../other", "~", str(tmp_path / "elsewhere")):
        out = json.loads(tools.colab_run({"path": "a.py", "output_dir": output_dir}))
        assert not out["ok"], output_dir
    assert store.list() == [] and not (hermes_home / "runs").exists() and not (tmp_path / "elsewhere").exists()
    ok = json.loads(tools.colab_run({"path": "a.py", "output_dir": "runs/first"}))
    assert ok["ok"] and ok["run"]["local_dir"] == str(project / "runs" / "first")


def test_output_dir_next_to_the_code_from_another_working_directory(tmp_path):
    home = tmp_path / "home"
    project, chat_dir = home / "work" / "proj", home / "notes"
    for folder in (project, chat_dir):
        folder.mkdir(parents=True)
    (project / "a.py").write_text("x")
    _, _, tools = _tools(tmp_path, cwd=chat_dir)
    ok = json.loads(tools.colab_run({"path": str(project / "a.py"), "output_dir": str(project / "out")}))
    assert ok["ok"] and ok["run"]["local_dir"] == str(project / "out")
    sibling = json.loads(tools.colab_run({"path": str(project / "a.py"), "output_dir": str(home / "work" / "x")}))
    assert not sibling["ok"] and "must be inside" in sibling["error"]


def test_a_working_directory_holding_home_does_not_widen_output_dir(tmp_path):
    home = tmp_path / "home"
    project, other = home / "work" / "proj", home / "work" / "other"
    for folder in (project, other):
        folder.mkdir(parents=True)
    (project / "a.py").write_text("x")
    for cwd in (home, tmp_path, "/"):
        _, _, tools = _tools(tmp_path, cwd=cwd)
        out = json.loads(tools.colab_run({"path": str(project / "a.py"), "output_dir": str(other / "run")}))
        assert not out["ok"] and "must be inside" in out["error"], cwd
    assert not (other / "run").exists()


def test_default_run_folder_symlinked_into_hermes_home_is_refused(tmp_path):
    hermes_home, project = tmp_path / "home" / ".hermes", tmp_path / "proj"
    for folder in (hermes_home, project):
        folder.mkdir(parents=True)
    (project / "a.py").write_text("x")
    (project / "colab_runs").symlink_to(hermes_home)
    store, _, tools = _tools(tmp_path, cwd=project)
    for args in ({"path": "a.py"}, {"path": "a.py", "output_dir": ""}):
        out = json.loads(tools.colab_run(args))
        assert not out["ok"] and "refusing" in out["error"], args
    assert store.list() == [] and list(hermes_home.iterdir()) == []


def test_default_run_folder_symlinked_to_an_ordinary_folder_is_refused(tmp_path):
    project, other = tmp_path / "proj", tmp_path / "other"
    for folder in (project, other):
        folder.mkdir()
    (project / "a.py").write_text("x")
    (project / "colab_runs").symlink_to(other)
    store, _, tools = _tools(tmp_path, cwd=project)
    out = json.loads(tools.colab_run({"path": "a.py"}))
    assert not out["ok"] and "symlink" in out["error"]
    assert store.list() == [] and list(other.iterdir()) == []


def test_a_run_folder_holding_the_target_of_a_symlinked_protected_folder_is_refused(tmp_path):
    home, project = tmp_path / "home", tmp_path / "proj"
    vault = project / "out" / "outputs" / "vault"
    vault.mkdir(parents=True)
    (vault / "authorized_keys").write_text("keep")
    (project / "a.py").write_text("x")
    (home / ".ssh").symlink_to(vault)
    store, sup, tools = _tools(tmp_path, cwd=project)
    out = json.loads(tools.colab_run({"path": "a.py", "output_dir": "out"}))
    assert not out["ok"] and "contains" in out["error"] and store.list() == []
    store.create({"id": "old", "name": "old", "spec": {}, "local_dir": str(project / "out"), "accelerator": "T4"})
    sup.tick()
    assert "Run folder refused" in store.get("old")["error"]
    assert sorted(p.name for p in vault.iterdir()) == ["authorized_keys"]
    assert (vault / "authorized_keys").read_text() == "keep"


def test_a_symlinked_or_hidden_entry_file_is_refused_as_named(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "main.py").write_text("x")
    (project / ".secret.py").symlink_to(project / "main.py")
    (project / "train.py").symlink_to(project / "main.py")
    _, _, tools = _tools(tmp_path, cwd=project)
    for path in (".secret.py", "train.py"):
        out = json.loads(tools.colab_run({"path": path}))
        assert not out["ok"] and "symlinked" in out["error"], path


def test_a_run_folder_retargeted_after_creation_is_refused(tmp_path):
    project, other = tmp_path / "proj", tmp_path / "proj" / "other"
    other.mkdir(parents=True)
    (project / "a.py").write_text("x")
    store, sup, tools = _tools(tmp_path, cwd=project)
    run_id = json.loads(tools.colab_run({"path": "a.py", "output_dir": "out"}))["run"]["id"]
    (project / "out").rename(project / "moved")
    (project / "out").symlink_to(other)
    sup.tick()
    run = store.get(run_id)
    assert run["state"] == "failed" and "now a symlink" in run["error"]
    assert "now a symlink" in json.loads(tools.colab_resume({"run": run_id}))["error"]
    assert "now a symlink" in json.loads(tools.colab_logs({"run": run_id}))["error"]
    assert list(other.iterdir()) == []


def test_logs_refuse_a_hard_linked_run_log(tmp_path):
    import os

    (tmp_path / "a.py").write_text("x")
    private = tmp_path / "home" / ".hermes" / "auth.json"
    private.parent.mkdir(parents=True)
    private.write_text("token")
    _, _, tools = _tools(tmp_path)
    run_id = json.loads(tools.colab_run({"path": "a.py"}))["run"]["id"]
    os.link(private, tmp_path / "colab_runs" / run_id / "run.log")
    out = json.loads(tools.colab_logs({"run": run_id}))
    assert not out["ok"] and "hard link" in out["error"] and "token" not in json.dumps(out)


def test_symlinks_inside_the_run_folder_cannot_redirect_writes(tmp_path):
    hermes_home, project = tmp_path / "home" / ".hermes", tmp_path / "proj"
    for folder in (hermes_home, project / "out", project / "out2"):
        folder.mkdir(parents=True)
    (project / "a.py").write_text("x")
    (project / "out" / "outputs").symlink_to(hermes_home)
    (hermes_home / "config.yaml").write_text("keep")
    (project / "out2" / "bundle.zip").symlink_to(hermes_home / "config.yaml")
    store, _, tools = _tools(tmp_path, cwd=project)
    out = json.loads(tools.colab_run({"path": "a.py", "output_dir": "out"}))
    assert not out["ok"] and "symlink" in out["error"]
    ok = json.loads(tools.colab_run({"path": "a.py", "output_dir": "out2"}))
    assert ok["ok"] and not (project / "out2" / "bundle.zip").is_symlink()
    assert (hermes_home / "config.yaml").read_text() == "keep" and sorted(hermes_home.iterdir()) == [
        hermes_home / "config.yaml"]


def test_protected_folders_count_where_their_symlinks_point(tmp_path):
    home, project = tmp_path / "home", tmp_path / "proj"
    (project / "vault").mkdir(parents=True)
    (project / "vault" / "private.py").write_text("key")
    (project / "main.py").write_text("x")
    (home / ".ssh").symlink_to(project / "vault")
    store, _, tools = _tools(tmp_path, cwd=project)
    assert "refusing" in json.loads(tools.colab_run({"path": "~/.ssh/private.py"}))["error"]
    assert "refusing" in json.loads(tools.colab_run({"path": "main.py", "output_dir": "~/.ssh/run"}))["error"]
    assert "refusing" in json.loads(tools.colab_run({"path": "vault/private.py"}))["error"]
    ok = json.loads(tools.colab_run({"path": ".", "entry": "main.py"}))
    assert ok["ok"] and ok["files"] == 1  # the directory bundle leaves the vault out
    assert not (project / "vault" / "run").exists()


def test_resume_refuses_a_run_folder_recorded_inside_hermes_home(tmp_path):
    hermes_home = tmp_path / "home" / ".hermes"
    hermes_home.mkdir(parents=True)
    store, _, tools = _tools(tmp_path)
    store.create({"id": "old", "name": "old", "spec": {}, "local_dir": str(hermes_home / "run"), "accelerator": "T4"})
    store.update("old", state="failed")
    out = json.loads(tools.colab_resume({"run": "old"}))
    assert not out["ok"] and "refusing" in out["error"] and store.get("old")["state"] == "failed"
    assert not json.loads(tools.colab_logs({"run": "old"}))["ok"]


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


def test_finish_notice_is_injected_in_the_starting_chat_context(tmp_path):
    import contextvars

    chat = contextvars.ContextVar("chat", default="")
    (tmp_path / "a.py").write_text("x")
    store, _, tools = _tools(tmp_path, surface="telegram")  # a messaging chat: keyless route
    token = chat.set("ses_abc")  # the tool call runs inside the chat's context
    run_id = json.loads(tools.colab_run({"path": "a.py"}, session_id="s1"))["run"]["id"]
    chat.reset(token)
    run = store.update(run_id, state="succeeded")
    seen = []
    # Later, on the supervisor thread, the injection must still see the starting chat.
    assert tools.push_notice(run, lambda text, **kw: seen.append((chat.get(), text, kw)) or True)
    assert seen[0][0] == "ses_abc" and "finished" in seen[0][1] and seen[0][2] == {}
    assert store.get(run_id)["notified"] == 1
    assert tools.pending_notices("s1") == []  # not repeated on the next turn


def test_desktop_chat_without_a_reported_surface_uses_its_session_key(tmp_path):
    (tmp_path / "a.py").write_text("x")
    store, _, tools = _tools(tmp_path, surface="")  # what the Desktop backend reports today
    run_id = json.loads(tools.colab_run({"path": "a.py"}, session_id="20261106_101849_202906"))["run"]["id"]
    run = store.update(run_id, state="succeeded")
    seen = []
    assert tools.push_notice(run, lambda text, **kw: seen.append(kw) or True)
    assert seen == [{"session_key": "20261106_101849_202906"}]


def test_desktop_notice_goes_to_the_session_key_even_after_a_restart(tmp_path):
    (tmp_path / "a.py").write_text("x")
    store, _, tools = _tools(tmp_path, surface="desktop")
    run_id = json.loads(tools.colab_run({"path": "a.py"}, session_id="20261106_094708_be892f"))["run"]["id"]
    tools.contexts.clear()  # a restart loses the in-memory contexts; the Desktop route needs none
    run = store.update(run_id, state="succeeded")
    seen = []
    assert tools.push_notice(run, lambda text, **kw: seen.append(kw) or True)
    assert seen == [{"session_key": "20261106_094708_be892f"}]
    assert store.get(run_id)["notified"] == 1


def test_resume_from_another_chat_moves_the_notice_there(tmp_path):
    (tmp_path / "a.py").write_text("x")
    store, _, tools = _tools(tmp_path, surface="telegram")
    run_id = json.loads(tools.colab_run({"path": "a.py"}, session_id="s1"))["run"]["id"]
    store.update(run_id, state="failed")
    tools.surface = lambda: "desktop"
    tools.colab_resume({"run": run_id}, session_id="s2")
    run = store.update(run_id, state="succeeded")
    seen = []
    assert tools.push_notice(run, lambda text, **kw: seen.append(kw) or True)
    assert seen == [{"session_key": "s2"}]


def test_refused_notice_falls_back_to_next_turn(tmp_path):
    (tmp_path / "a.py").write_text("x")
    store, _, tools = _tools(tmp_path)
    run_id = json.loads(tools.colab_run({"path": "a.py"}, session_id="s1"))["run"]["id"]
    run = store.update(run_id, state="failed", error="exit code 1")
    assert not tools.push_notice(run, lambda text, **kw: False)  # injection not allowed
    notes = tools.pending_notices("s1")
    assert len(notes) == 1 and "failed" in notes[0] and "exit code 1" in notes[0]


def test_logs_leave_out_progress_lines(tmp_path):
    (tmp_path / "a.py").write_text("x")
    store, _, tools = _tools(tmp_path)
    run_id = json.loads(tools.colab_run({"path": "a.py"}))["run"]["id"]
    log = [f"line {i}\nHERMES_PROGRESS {{\"step\": {i}, \"total\": 9}}" for i in range(9)]
    (tmp_path / "colab_runs" / run_id / "run.log").write_text("\n".join(log) + "\n")
    lines = json.loads(tools.colab_logs({"run": run_id, "lines": 3}))["lines"]
    assert lines == ["line 6", "line 7", "line 8"]


def test_idle_supervisor_adopts_a_run_left_by_an_exited_process(tmp_path):
    """A long-lived Hermes (the Desktop backend, the gateway) whose supervisor went idle must pick up
    a run that a short-lived process (a cron tick, a one-shot CLI) started and then left behind."""
    import contextlib
    import time as _time

    from hermes_colab.supervisor import SupervisorThread

    store = Store(tmp_path / "runs.db")
    ticks = []

    class Counting:
        def __init__(self):
            self.store = store

        def tick(self):
            active = len(store.list(states=("queued",)))
            ticks.append(active)
            return active

    thread = SupervisorThread(Counting(), home_scope=contextlib.nullcontext, interval=0.02)
    thread.standby_interval = 0.02
    thread.ensure_running()
    try:
        _time.sleep(0.3)  # idle: nothing to drive
        store.create({"id": "r1", "name": "left", "spec": {}, "local_dir": str(tmp_path), "accelerator": "T4"})
        deadline = _time.monotonic() + 2
        while 1 not in ticks and _time.monotonic() < deadline:
            _time.sleep(0.02)
        assert 1 in ticks, "the idle supervisor never adopted the run"
    finally:
        thread.stop()


def test_new_run_wakes_a_standby_supervisor_at_once(tmp_path):
    """colab_run calls ensure_running: a supervisor on standby must not sit out its standby interval."""
    import contextlib
    import time as _time

    from hermes_colab.supervisor import SupervisorThread

    store = Store(tmp_path / "runs.db")
    seen = []

    class Counting:
        def __init__(self):
            self.store = store

        def tick(self):
            seen.append(len(store.list(states=("queued",))))
            return seen[-1]

    thread = SupervisorThread(Counting(), home_scope=contextlib.nullcontext, interval=0.02)
    thread.standby_interval = 30
    thread.ensure_running()
    try:
        _time.sleep(0.2)
        store.create({"id": "r1", "name": "new", "spec": {}, "local_dir": str(tmp_path), "accelerator": "T4"})
        thread.ensure_running()
        deadline = _time.monotonic() + 2
        while 1 not in seen and _time.monotonic() < deadline:
            _time.sleep(0.02)
        assert 1 in seen, "the new run waited for the standby interval"
    finally:
        thread.stop()


def test_finish_notice_shows_the_outputs_folder_under_home_as_tilde(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "a.py").write_text("x")
    store, _, tools = _tools(tmp_path)
    run_id = json.loads(tools.colab_run({"path": "a.py"}))["run"]["id"]
    run = store.update(run_id, state="succeeded", synced={"model.pt": [1, 1.0]})
    text = tools.summary(run)
    assert f"Outputs in ~/colab_runs/{run_id}/outputs: model.pt." in text
    assert str(tmp_path) not in text
