from __future__ import annotations

import io
import os
import zipfile

from hermes_colab.store import Store
from hermes_colab.supervisor import Supervisor


class FakeClient:
    def __init__(self):
        self.downloads = []

    def run_dir(self, run_id):
        return f"/vm/{run_id}"

    def download(self, path, size):
        self.downloads.append(path)
        return b"x" * size


def _setup(tmp_path):
    store = Store(tmp_path / "runs.db")
    sup = Supervisor(store, settings=lambda: {"max_sync_mb": 1}, tunnel=object())
    run = store.create({"id": "r", "name": "r", "spec": {}, "local_dir": str(tmp_path / "local"), "accelerator": "T4"})
    return sup, store, run


def test_checkpoint_rewritten_every_poll_still_syncs(tmp_path):
    """Regression: a file rewritten once per poll never held still across two polls and only came home
    at the end of the run, so a runtime loss restarted training from scratch."""
    sup, store, run = _setup(tmp_path)
    client = FakeClient()
    for poll in range(3):
        mtime = 1000.0 + 10 * poll
        sup._sync(store.get("r"), client, [["state.json", 10 + poll, mtime]], final=False, now=mtime + 2.5)
    assert len(client.downloads) == 3
    assert (tmp_path / "local" / "outputs" / "state.json").read_bytes() == b"x" * 12


def test_file_being_written_is_not_copied(tmp_path):
    sup, store, run = _setup(tmp_path)
    client = FakeClient()
    sup._sync(store.get("r"), client, [["big.bin", 100, 1000.0]], final=False, now=1000.5)
    assert client.downloads == []
    sup._sync(store.get("r"), client, [["big.bin", 200, 1001.0]], final=False, now=1001.4)
    assert client.downloads == []
    sup._sync(store.get("r"), client, [["big.bin", 200, 1001.0]], final=False, now=1001.6)  # held still
    assert client.downloads == ["/vm/r/outputs/big.bin"]


def test_unsafe_paths_and_oversized_files_are_skipped(tmp_path):
    sup, store, run = _setup(tmp_path)
    client = FakeClient()
    listing = [["../escape", 1, 0.0], ["/abs", 1, 0.0], ["huge.bin", 5 * 2**20, 0.0]]
    sup._sync(store.get("r"), client, listing, final=True, now=100.0)
    assert client.downloads == []
    assert store.get("r")["synced"]["huge.bin"][0] == "skipped"


def test_outputs_sync_in_write_order(tmp_path):
    """Weights written before the progress file reach home first, so a resume never pairs a newer
    progress file with older weights."""
    sup, store, run = _setup(tmp_path)
    client = FakeClient()
    listing = [["state.json", 1, 1005.0], ["model.pt", 1, 1001.0], ["metrics.csv", 1, 1003.0]]
    sup._sync(store.get("r"), client, listing, final=False, now=1010.0)
    assert client.downloads == ["/vm/r/outputs/model.pt", "/vm/r/outputs/metrics.csv", "/vm/r/outputs/state.json"]


def test_failure_line_names_the_exception(tmp_path):
    from hermes_colab.supervisor import _failure_line

    log = tmp_path / "run.log"
    log.write_text("epoch 1\nTraceback (most recent call last):\n  File \"t.py\", line 3\n"
                   "ImportError: Found an incompatible version of torchao.\n")
    assert _failure_line(log) == "ImportError: Found an incompatible version of torchao."
    log.write_text("about to fail\n")
    assert _failure_line(log) == "about to fail"
    assert _failure_line(tmp_path / "missing.log") == ""


def test_symlinks_in_the_run_folder_do_not_redirect_synced_outputs_or_the_log(tmp_path):
    import pytest

    from hermes_colab.bundle import BundleError

    sup, store, run = _setup(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "victim").write_text("keep")
    out = tmp_path / "local" / "outputs"
    (out / "sub").mkdir(parents=True)
    (out / "linked").symlink_to(outside)
    (out / "model.pt").symlink_to(outside / "victim")
    (out / "sub" / "w.bin.part").symlink_to(outside / "victim")
    with pytest.raises(BundleError):
        sup._sync(store.get("r"), FakeClient(), [["linked/new/nested/x", 1, 0.0]], final=True, now=100.0)
    assert not (outside / "new").exists()  # refused before creating anything below the link
    sup._sync(store.get("r"), FakeClient(), [["model.pt", 1, 0.0], ["sub/w.bin", 2, 0.0]], final=True, now=100.0)
    assert not (out / "model.pt").is_symlink() and (out / "sub" / "w.bin").read_bytes() == b"xx"
    (tmp_path / "local" / "run.log").symlink_to(outside / "victim")
    with pytest.raises(BundleError):
        sup._append_log(store.get("r"), {"log": "line\n"})
    assert (outside / "victim").read_text() == "keep" and sorted(p.name for p in outside.iterdir()) == ["victim"]


def test_supervisor_fails_a_recorded_run_whose_folder_is_inside_hermes_home(tmp_path):
    hermes_home = tmp_path / "home" / ".hermes"
    store = Store(tmp_path / "runs.db")
    finals = []
    sup = Supervisor(store, settings=lambda: {}, tunnel=object(), on_final=finals.append)
    store.create({"id": "old", "name": "old", "spec": {}, "local_dir": str(hermes_home / "run"), "accelerator": "T4"})
    sup.tick()
    run = store.get("old")
    assert run["state"] == "failed" and "Run folder refused" in run["error"] and finals
    assert not hermes_home.exists()


class RecordingClient(FakeClient):
    """A runtime that records every upload and serves one finished output file."""

    def __init__(self):
        super().__init__()
        self.uploads = {}

    def prepare(self, run_id):
        return self.run_dir(run_id)

    def upload(self, path, data):
        self.uploads[path] = data

    def unzip(self, *args):
        return 1

    def launch(self, run_dir):
        pass

    def poll(self, run_dir, offset, **kw):
        return {"log": "line\n", "log_offset": offset + 5, "outputs": [["ckpt.pt", 2, 0.0]], "now": 100.0,
                "status": {}}


def _live(tmp_path, state, *, attempt=1):
    """A run in *state* on a fake runtime, with its run folder at tmp_path/proj/out."""
    store = Store(tmp_path / "runs.db")
    tunnel = type("T", (), {"keep_alive": lambda self, e: True, "list_runtimes": lambda self: [],
                            "release": lambda self, e: None})()
    sup = Supervisor(store, settings=lambda: {}, tunnel=tunnel)
    local = tmp_path / "proj" / "out"
    (local / "outputs").mkdir(parents=True)
    (local / "bundle.zip").write_bytes(b"zip")
    store.create({"id": "r", "name": "r", "spec": {}, "local_dir": str(local), "accelerator": "T4"})
    store.update("r", state=state, endpoint="e", attempt=attempt, started_at=99.0)
    client = sup._clients["r"] = RecordingClient()
    sup._keepalive["r"] = sup._liveness["r"] = 10**12
    return store, sup, client, local


def test_a_retargeted_run_folder_stops_sync_and_log_before_any_write(tmp_path):
    store, sup, client, local = _live(tmp_path, "running")
    other = tmp_path / "other"
    other.mkdir()
    local.rename(tmp_path / "proj" / "moved")
    local.symlink_to(other)
    sup.tick()
    assert "now a symlink" in store.get("r")["error"] and client.downloads == []
    assert list(other.iterdir()) == []


def test_resume_upload_leaves_out_a_protected_folder_inside_the_run_folder(tmp_path):
    store, sup, client, local = _live(tmp_path, "uploading", attempt=2)
    vault = local / "outputs" / "vault"
    vault.mkdir()
    (vault / "id_ed25519").write_text("key")
    (tmp_path / "home" / ".ssh").symlink_to(vault)
    sup.tick()
    assert "Run folder refused" in store.get("r")["error"] and client.uploads == {}


def test_hard_linked_run_files_are_neither_appended_read_nor_uploaded(tmp_path):
    from hermes_colab.supervisor import _failure_line

    private = tmp_path / "home" / ".hermes" / "auth.json"
    private.parent.mkdir(parents=True)
    private.write_text("token")
    store, sup, client, local = _live(tmp_path, "uploading", attempt=2)
    (local / "outputs" / "ckpt.pt").write_text("w")
    os.link(private, local / "outputs" / "stolen.json")
    sup.tick()
    restored = client.uploads["/vm/r/.incoming/outputs.zip"]
    assert zipfile.ZipFile(io.BytesIO(restored)).namelist() == ["ckpt.pt"]
    os.link(private, local / "run.log")
    store.update("r", state="running")
    sup.tick()
    assert private.read_text() == "token" and "hard link" in store.get("r")["error"]
    assert _failure_line(local / "run.log") == ""
