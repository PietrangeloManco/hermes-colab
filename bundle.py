"""Package the local code a run uploads to its runtime."""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "env", "node_modules", ".mypy_cache",
             ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints", "colab_runs", ".hermes"}
SKIP_SUFFIXES = {".pyc", ".pyo"}
SECRET_NAMES = {".env", ".netrc", ".npmrc", ".pypirc", "credentials.json", "token.json", "id_rsa", "id_ed25519"}


class BundleError(ValueError):
    pass


def _files(root: Path):
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(files):
            path = Path(base) / name
            if path.suffix in SKIP_SUFFIXES or path.is_symlink():
                continue
            if name in SECRET_NAMES or name.startswith(".env"):
                continue
            yield path


def build_bundle(source: Path, entry: str, *, max_bytes: int) -> tuple[bytes, list[str]]:
    """Zip *source* (a directory, or a single file) into memory. Returns (zip bytes, file list).

    Hidden directories, virtualenvs, caches, symlinks and common credential files are never packed.
    """
    files: list[tuple[Path, str]] = []
    if source.is_file():
        files.append((source, source.name))
        sibling = source.with_name("requirements.txt")
        if sibling.is_file() and sibling != source:
            files.append((sibling, sibling.name))
    else:
        files = [(p, p.relative_to(source).as_posix()) for p in _files(source)]
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
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and not path.is_symlink() and not path.name.endswith(".part"):
                zf.write(path, path.relative_to(directory).as_posix())
    return buf.getvalue()
