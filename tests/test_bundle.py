from __future__ import annotations

import io
import zipfile

import pytest

from hermes_colab.bundle import BundleError, build_bundle, zip_dir


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


def test_directory_bundle_skips_hidden_files_but_keeps_the_allowlist(tmp_path):
    (tmp_path / "train.py").write_text("print(1)")
    (tmp_path / "pkg").mkdir()
    for name in (".git-credentials", ".pgpass", ".bash_history", "pkg/.secret"):
        (tmp_path / name).write_text("secret")
    (tmp_path / ".python-version").write_text("3.11")
    data, names = build_bundle(tmp_path, "train.py", max_bytes=10**6)
    assert _names(data) == set(names) == {"train.py", ".python-version"}


def test_single_file_must_be_packable_and_skips_a_symlinked_requirements(tmp_path):
    (tmp_path / ".secret.py").write_text("x")
    with pytest.raises(BundleError, match="hidden"):
        build_bundle(tmp_path / ".secret.py", ".secret.py", max_bytes=10**6)
    (tmp_path / ".pgpass").write_text("secret")
    (tmp_path / "requirements.txt").symlink_to(tmp_path / ".pgpass")
    (tmp_path / "a.py").write_text("x")
    data, names = build_bundle(tmp_path / "a.py", "a.py", max_bytes=10**6)
    assert _names(data) == set(names) == {"a.py"}


def test_hard_linked_files_are_never_packed(tmp_path):
    import os

    private = tmp_path / "auth.json"
    private.write_text("token")
    project = tmp_path / "proj"
    project.mkdir()
    (project / "train.py").write_text("print(1)")
    for name in ("data.json", "linked.py"):
        os.link(private, project / name)
    data, names = build_bundle(project, "train.py", max_bytes=10**6)
    assert _names(data) == set(names) == {"train.py"}
    os.link(private, project / "requirements.txt")
    _, names = build_bundle(project / "train.py", "train.py", max_bytes=10**6)
    assert names == ["train.py"]
    with pytest.raises(BundleError, match="hard-linked"):
        build_bundle(project / "linked.py", "linked.py", max_bytes=10**6)


def test_skip_prunes_folders(tmp_path):
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "vault").mkdir()
    (tmp_path / "vault" / "k.py").write_text("x")
    _, names = build_bundle(tmp_path, "a.py", max_bytes=10**6, skip=lambda p: p.name == "vault")
    assert names == ["a.py"]


def test_zip_dir_does_not_follow_symlinks(tmp_path):
    outputs, outside = tmp_path / "outputs", tmp_path / "outside"
    for folder in (outputs, outside):
        folder.mkdir()
    (outside / "secret").write_text("x")
    (outputs / "ckpt.pt").write_text("w")
    (outputs / "linked").symlink_to(outside)
    (outputs / "file-link").symlink_to(outside / "secret")
    assert _names(zip_dir(outputs)) == {"ckpt.pt"}


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
