"""Runtime environment: Colab detection, environment description, output folders."""

import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]  # src/dental_3d_vae/runtime/env.py -> repo root
DRIVE_ROOT = Path("/content/drive/MyDrive")


def in_colab() -> bool:
    """True when running inside a Google Colab runtime."""
    return "google.colab" in sys.modules


def _git(*args: str) -> str:
    result = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True)
    return result.stdout.strip()


def describe_environment() -> dict[str, str | bool | None]:
    """Versions, hardware and code version, for printing and saving with each run.

    ``git_dirty`` is True when there are uncommitted changes, meaning
    ``git_commit`` alone does not fully describe the code that ran.
    """
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "in_colab": in_colab(),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }


def _mount_drive() -> None:
    from google.colab import drive

    drive.mount(str(DRIVE_ROOT.parent))


def get_data_dir() -> Path:
    """Folder for datasets, created if missing.

    In Colab: mounts Google Drive and returns ``MyDrive/<repo name>/data``, so data
    is uploaded once rather than every session. Locally: ``<repo>/data`` (ignored
    by Git; Teeth3DS+ data must never be committed).
    """
    if in_colab():
        _mount_drive()
        data_dir = DRIVE_ROOT / REPO_ROOT.name / "data"
    else:
        data_dir = REPO_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_runs_dir(drive_subdir: str | None = None) -> Path:
    """Folder for run outputs, created if missing.

    In Colab: mounts Google Drive and returns ``MyDrive/<drive_subdir>``, by default
    ``MyDrive/<repo name>/runs``, so outputs survive the runtime and stay separate per
    project. Locally: ``<repo>/artifacts/runs`` (ignored by Git).
    """
    if in_colab():
        _mount_drive()
        runs_dir = DRIVE_ROOT / (drive_subdir or f"{REPO_ROOT.name}/runs")
    else:
        runs_dir = REPO_ROOT / "artifacts" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def copy_to_runtime(folder: Path) -> Path:
    """In Colab, a copy of ``folder`` on the runtime's own disk; locally, ``folder`` itself.

    Data loaders read one tooth at a time, which is slow from Google Drive.
    """
    if not in_colab():
        return folder
    local = Path("/content/data") / folder.name
    shutil.copytree(folder, local, dirs_exist_ok=True)
    return local
