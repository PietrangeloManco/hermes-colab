from __future__ import annotations

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
