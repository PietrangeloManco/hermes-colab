"""Package the local code a run uploads to its runtime."""

from __future__ import annotations

import contextlib
import io
import os
import zipfile
from collections.abc import Callable
from pathlib import Path

SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "env", "node_modules", ".mypy_cache",
             ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints", "colab_runs", ".hermes"}
SKIP_SUFFIXES = {".pyc", ".pyo"}
SECRET_NAMES = {".env", ".netrc", ".npmrc", ".pypirc", "credentials.json", "token.json", "id_rsa", "id_ed25519"}
# Hidden files (shell history, .git-credentials, .pgpass, ...) stay home; only these travel with the code.
HIDDEN_ALLOWED = {".python-version"}


class BundleError(ValueError):
    pass


def packable(path: Path) -> bool:
    """Not a symlink or a hard-linked file (either may be another of the user's files, a credential
    included), not hidden (outside the allowlist), not a credential file, not bytecode."""
    name = path.name
    if path.is_symlink() or path.suffix in SKIP_SUFFIXES or name in SECRET_NAMES:
        return False
    with contextlib.suppress(OSError):
        if path.is_file() and path.lstat().st_nlink > 1:
            return False
    return not name.startswith(".") or name in HIDDEN_ALLOWED


def _files(root: Path, skip: Callable[[Path], bool]):
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")
                         and not skip(Path(base) / d))
        for name in sorted(files):
            path = Path(base) / name
            if packable(path) and path.is_file():
                yield path


def build_bundle(source: Path, entry: str, *, max_bytes: int,
                 skip: Callable[[Path], bool] = lambda _path: False) -> tuple[bytes, list[str]]:
    """Zip *source* (a directory, or a single file) into memory. Returns (zip bytes, file list).

    Hidden directories and files (except ``HIDDEN_ALLOWED``), virtualenvs, caches, symlinks, common
    credential files and folders for which *skip* is true are never packed.
    """
    files: list[tuple[Path, str]] = []
    if source.is_file():
        if not packable(source):
            raise BundleError(f"refusing to upload {source.name}: hidden, credential, symlinked or hard-linked file")
        files.append((source, source.name))
        sibling = source.with_name("requirements.txt")
        if sibling != source and packable(sibling) and sibling.is_file():
            files.append((sibling, sibling.name))
    else:
        files = [(p, p.relative_to(source).as_posix()) for p in _files(source, skip)]
    names = [name for _, name in files]
    if entry not in names:
        raise BundleError(f"entry point {entry!r} is not in the files to upload")
    total = sum(p.stat().st_size for p, _ in files)
    if total > max_bytes:
        raise BundleError(f"{total / 2**20:.1f} MB to upload exceeds max_upload_mb ({max_bytes / 2**20:.0f} MB); "
                          "download large datasets inside the job instead (Hugging Face, URLs)")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, name in files:
            zf.write(path, name)
    return buf.getvalue(), names


def zip_dir(directory: Path) -> bytes:
    """Zip the synced outputs for a resume: regular files only, symlinks neither packed nor followed."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for base, _dirs, files in os.walk(directory):  # os.walk does not enter symlinked folders
            for name in sorted(files):
                path = Path(base) / name
                # A hard-linked file may be another of the user's files planted in the outputs.
                if (path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
                        and not name.endswith(".part")):
                    zf.write(path, path.relative_to(directory).as_posix())
    return buf.getvalue()
