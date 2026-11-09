import sys
import types
from pathlib import Path

import pytest

from dental_3d_vae.runtime import env


def test_describe_environment_locally() -> None:
    info = env.describe_environment()

    assert info["in_colab"] is False
    assert info["torch"]
    assert info["git_commit"]  # Tests run inside the repo, so a commit exists.
    assert isinstance(info["git_dirty"], bool)


def test_runs_dir_locally_is_under_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(env, "REPO_ROOT", tmp_path)

    runs_dir = env.get_runs_dir()

    assert runs_dir == tmp_path / "artifacts" / "runs"
    assert runs_dir.is_dir()


def _fake_colab(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Pretend to be in Colab with Drive at tmp_path/MyDrive; returns mount calls."""
    mounted: list[str] = []
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=mounted.append)
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(env, "DRIVE_ROOT", tmp_path / "MyDrive")
    return mounted


def test_data_dir_locally_and_in_colab(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env, "REPO_ROOT", tmp_path / "my-project")
    assert env.get_data_dir() == tmp_path / "my-project" / "data"

    mounted = _fake_colab(tmp_path, monkeypatch)
    data_dir = env.get_data_dir()

    assert mounted == [str(tmp_path)]
    assert data_dir == tmp_path / "MyDrive" / "my-project" / "data"
    assert data_dir.is_dir()


def test_runs_dir_in_colab_mounts_drive_and_uses_subdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mounted = _fake_colab(tmp_path, monkeypatch)

    runs_dir = env.get_runs_dir("experiments/custom")

    assert mounted == [str(tmp_path)]
    assert runs_dir == tmp_path / "MyDrive" / "experiments" / "custom"
    assert runs_dir.is_dir()

    monkeypatch.setattr(env, "REPO_ROOT", tmp_path / "my-project")
    assert env.get_runs_dir() == tmp_path / "MyDrive" / "my-project" / "runs"


def test_copy_to_runtime_is_a_no_op_outside_colab(tmp_path) -> None:
    assert env.copy_to_runtime(tmp_path) == tmp_path
