from __future__ import annotations

import io
import zipfile

import pytest

from hermes_colab.bundle import BundleError, build_bundle


def _names(data: bytes) -> set[str]:
    return set(zipfile.ZipFile(io.BytesIO(data)).namelist())


def test_directory_bundle_skips_secrets_caches_and_hidden(tmp_path):
    (tmp_path / "train.py").write_text("print(1)")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "m.py").write_text("x = 1")
    for name in (".env", ".env.local", "credentials.json", "token.json"):
        (tmp_path / name).write_text("secret")
    for d in (".git", ".venv", "__pycache__", "colab_runs", ".hidden"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "f.py").write_text("x")
    (tmp_path / "link.py").symlink_to(tmp_path / "train.py")
    data, names = build_bundle(tmp_path, "train.py", max_bytes=10**6)
    assert _names(data) == set(names) == {"train.py", "pkg/m.py"}


def test_single_file_takes_sibling_requirements(tmp_path):
    (tmp_path / "nb.ipynb").write_text("{}")
    (tmp_path / "requirements.txt").write_text("peft")
    (tmp_path / "other.py").write_text("x")
    data, names = build_bundle(tmp_path / "nb.ipynb", "nb.ipynb", max_bytes=10**6)
    assert set(names) == {"nb.ipynb", "requirements.txt"}


def test_entry_must_be_in_bundle(tmp_path):
    (tmp_path / "a.py").write_text("x")
    with pytest.raises(BundleError):
        build_bundle(tmp_path, "missing.py", max_bytes=10**6)


def test_size_limit(tmp_path):
    (tmp_path / "a.py").write_text("x" * 2000)
    with pytest.raises(BundleError, match="max_upload_mb"):
        build_bundle(tmp_path, "a.py", max_bytes=1000)
