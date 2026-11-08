"""Where a run may read its code and keep its folder.

A run never reads from or writes to HERMES_HOME or the credential folders in the home directory,
whether named directly or reached through a symlink, and never takes the home directory (or a folder
holding it or HERMES_HOME) as a whole. Files inside a run folder are written without following
symlinks, so a link planted there cannot redirect a write.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Iterable
from pathlib import Path

from .bundle import BundleError, packable

PROTECTED = (".hermes", ".ssh", ".gnupg", ".aws", ".config", ".kube", ".docker", ".local/share/keyrings")
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def default_hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except Exception:  # outside Hermes (tests, probes)
        return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")


def within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


class PathPolicy:
    def __init__(self, hermes_home: Callable[[], Path] = default_hermes_home):
        self.hermes_home = hermes_home

    def _hermes_home(self) -> Path:
        return Path(self.hermes_home()).expanduser().resolve()

    def private_roots(self) -> list[Path]:
        """HERMES_HOME and the credential folders, as they resolve (a symlinked one counts at its target)."""
        home = Path.home()
        return [self._hermes_home(), *((home / name).resolve() for name in PROTECTED)]

    def broad(self, path: Path) -> bool:
        """True for the home directory, HERMES_HOME or any folder holding either."""
        return any(within(root, path) for root in (Path.home().resolve(), self._hermes_home()))

    def check_private(self, path: Path, action: str) -> None:
        for root in self.private_roots():
            if within(path, root):
                raise BundleError(f"refusing to {action} {root}")

    def check_source(self, path: Path) -> Path:
        """*path* resolved, if it may be uploaded. A file must be packable as named: not a hidden or
        credential file, nor a symlink or hard link (another file travelling under this name)."""
        if path.is_file() and not packable(path):
            raise BundleError(f"refusing to upload {path.name}: hidden, credential, symlinked or hard-linked file")
        path = path.resolve()
        if self.broad(path):
            raise BundleError(f"refusing to upload {path}: it is or contains the home directory or HERMES_HOME; "
                              "pass the project folder")
        self.check_private(path, "upload from")
        return path

    def check_run_dir(self, path: Path) -> Path:
        """*path* resolved, if a run may keep its log, bundle and outputs there."""
        path = Path(path).resolve()
        self.check_private(path, "write run files to")
        if self.broad(path):
            raise BundleError(f"refusing to write run files to {path}: pick a folder for this run")
        for root in self.private_roots():
            if within(root, path):  # a symlinked credential folder may point in here
                raise BundleError(f"refusing to write run files to {path}: it contains {root}")
        return path

    def check_recorded(self, path: Path) -> Path:
        """A run folder recorded at creation (always resolved), if it is still where it was: a folder on
        the way replaced by a symlink would move every later write."""
        path = Path(path)
        if self.check_run_dir(path) != path:
            raise BundleError(f"refusing to write run files to {path}: it or a folder on the way is now a symlink")
        return path

    def output_dir(self, raw: str | None, *, code_dir: Path, cwd: Path, run_id: str) -> Path:
        """The run folder: ``<code folder>/colab_runs/<id>``, or *raw* inside the chat's working directory
        or the code folder (a working directory that is the home directory or holds it does not count)."""
        code_dir = code_dir.resolve()
        if not raw:
            path = code_dir / "colab_runs" / run_id
            if self.check_run_dir(path) != path:
                raise BundleError(f"refusing to write run files to {path}: colab_runs is a symlink")
            return path
        cwd = cwd.resolve()
        path = Path(os.path.expanduser(raw))
        path = self.check_run_dir(path if path.is_absolute() else cwd / path)
        roots = [root for root in (cwd, code_dir) if not self.broad(root)]
        if not any(within(path, root) for root in roots):
            raise BundleError(f"output_dir {path} must be inside the chat's working directory or the folder that "
                              f"holds the code ({code_dir}), not the home directory as a whole")
        return path


def folder(root: Path, *parts: str) -> Path:
    """``root/parts`` created one level at a time, refused before creating anything below a symlink.
    *root* is a resolved run folder."""
    root.mkdir(parents=True, exist_ok=True)
    if root.resolve() != root:
        raise BundleError(f"refusing to write into {root}: a folder on the way is a symlink")
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise BundleError(f"refusing to write into {path}: it is a symlink")
        path.mkdir(exist_ok=True)
    return path


def child(root: Path, *parts: str) -> Path:
    """``root/parts`` with its parent folders created as by :func:`folder`."""
    return folder(root, *parts[:-1]) / parts[-1]


def write_new(path: Path, data: bytes) -> None:
    """Write *path* afresh: an existing entry (a symlink included) is removed, never followed."""
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)


def _no_symlink(path: Path) -> None:
    if path.is_symlink():
        raise BundleError(f"refusing to follow the symlink {path}")


def _single_link(fd: int, path: Path) -> None:
    """A run file with another hard link may be another file of the user's (a credential, a config)."""
    if os.fstat(fd).st_nlink > 1:
        os.close(fd)
        raise BundleError(f"refusing to use {path}: it has another hard link")


def append_text(path: Path, text: str) -> None:
    _no_symlink(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | _NOFOLLOW, 0o644)
    _single_link(fd, path)
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        fh.write(text)


def read_bytes(path: Path, limit: int | None = None) -> bytes:
    """The file's bytes (the last *limit* ones when given), refusing symlinks and hard-linked files."""
    _no_symlink(path)
    fd = os.open(path, os.O_RDONLY | _NOFOLLOW | getattr(os, "O_BINARY", 0))
    _single_link(fd, path)
    with os.fdopen(fd, "rb") as fh:
        if limit is not None:
            fh.seek(max(0, fh.seek(0, os.SEEK_END) - limit))
        return fh.read()


def excluded(roots: Iterable[Path]) -> Callable[[Path], bool]:
    resolved = list(roots)
    return lambda path: any(within(path.resolve(), root) for root in resolved)
