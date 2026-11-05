"""Durable run records (SQLite in the plugin's profile-scoped data directory).

No credential is stored here: the runtime proxy token is re-read from Colab's assignment list when
a supervisor needs it, and the Google grant lives in Hermes' own auth store.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ACTIVE = ("queued", "waiting_gpu", "assigning", "uploading", "starting", "running", "resuming", "cancelling")
FINAL = ("succeeded", "failed", "cancelled")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    spec TEXT NOT NULL,
    local_dir TEXT NOT NULL,
    accelerator TEXT NOT NULL,
    endpoint TEXT,
    attempt INTEGER NOT NULL DEFAULT 0,
    resumes INTEGER NOT NULL DEFAULT 0,
    log_offset INTEGER NOT NULL DEFAULT 0,
    progress TEXT,
    gpu TEXT,
    exit_code INTEGER,
    error TEXT,
    message TEXT,
    started_at REAL,
    finished_at REAL,
    runtime_seconds REAL NOT NULL DEFAULT 0,
    synced TEXT NOT NULL DEFAULT '{}',
    session_id TEXT,
    notified INTEGER NOT NULL DEFAULT 0,
    events TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS lease (
    name TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    expires_at REAL NOT NULL
);
"""
_JSON = ("spec", "progress", "gpu", "synced", "events")


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._local = threading.local()
        with self._db() as db:
            db.executescript(_SCHEMA)

    def _db(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        data = dict(row)
        for key in _JSON:
            if data.get(key) is not None:
                data[key] = json.loads(data[key])
        return data

    def create(self, run: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        row = {"created_at": now, "updated_at": now, "state": "queued", **run}
        for key in _JSON:
            if key in row and not isinstance(row[key], str) and row[key] is not None:
                row[key] = json.dumps(row[key])
        cols = ", ".join(row)
        self._db().execute(f"INSERT INTO runs ({cols}) VALUES ({', '.join('?' * len(row))})", tuple(row.values()))
        return self.get(run["id"])  # type: ignore[return-value]

    def get(self, run_id: str) -> dict[str, Any] | None:
        return self._row(self._db().execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone())

    def find(self, ref: str) -> dict[str, Any] | None:
        """A run by exact id, id prefix, or most recent with that name."""
        db = self._db()
        row = db.execute("SELECT * FROM runs WHERE id = ?", (ref,)).fetchone()
        if row is None:
            rows = db.execute("SELECT * FROM runs WHERE id LIKE ? ORDER BY created_at DESC", (ref + "%",)).fetchall()
            row = rows[0] if len(rows) == 1 else None
        if row is None:
            row = db.execute("SELECT * FROM runs WHERE name = ? ORDER BY created_at DESC LIMIT 1", (ref,)).fetchone()
        return self._row(row)

    def list(self, *, states: Iterable[str] = (), limit: int = 20) -> list[dict[str, Any]]:
        states = tuple(states)
        if states:
            marks = ", ".join("?" * len(states))
            rows = self._db().execute(
                f"SELECT * FROM runs WHERE state IN ({marks}) ORDER BY created_at DESC LIMIT ?", (*states, limit))
        else:
            rows = self._db().execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
        return [self._row(r) for r in rows.fetchall()]  # type: ignore[misc]

    def update(self, run_id: str, **changes: Any) -> dict[str, Any]:
        changes["updated_at"] = time.time()
        for key in _JSON:
            if key in changes and changes[key] is not None and not isinstance(changes[key], str):
                changes[key] = json.dumps(changes[key])
        sets = ", ".join(f"{k} = ?" for k in changes)
        self._db().execute(f"UPDATE runs SET {sets} WHERE id = ?", (*changes.values(), run_id))
        return self.get(run_id)  # type: ignore[return-value]

    def event(self, run_id: str, text: str) -> None:
        run = self.get(run_id)
        if run is None:
            return
        events = (run.get("events") or [])[-49:] + [[round(time.time(), 1), text]]
        self.update(run_id, events=events, message=text)

    # ── Supervisor lease: one Hermes process drives the runs of a profile at a time ───────────────
    def acquire_lease(self, owner: str, ttl: float = 35.0) -> bool:
        now = time.time()
        db = self._db()
        db.execute("BEGIN IMMEDIATE")
        try:
            row = db.execute("SELECT owner, expires_at FROM lease WHERE name = 'supervisor'").fetchone()
            if row is None or row["owner"] == owner or row["expires_at"] < now:
                db.execute("INSERT OR REPLACE INTO lease (name, owner, expires_at) VALUES ('supervisor', ?, ?)",
                           (owner, now + ttl))
                db.execute("COMMIT")
                return True
            db.execute("COMMIT")
            return False
        except Exception:
            db.execute("ROLLBACK")
            raise

    def release_lease(self, owner: str) -> None:
        self._db().execute("DELETE FROM lease WHERE name = 'supervisor' AND owner = ?", (owner,))


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + os.urandom(3).hex()
