import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).resolve().parents[1]


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True)


def _copy_as_repo(destination: Path, remote: str) -> None:
    """Copy the template's project files into a new Git repo with the given remote."""
    listing = _run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], TEMPLATE)
    for relative in listing.stdout.splitlines():
        source = TEMPLATE / relative
        if source.exists():
            (destination / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination / relative)
    _run(["git", "init", "-q"], destination)
    _run(["git", "remote", "add", "origin", remote], destination)
    _run(["git", "add", "-A"], destination)
    _run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "x"], destination)


def test_setup_renames_the_project_and_its_tests_still_pass(tmp_path: Path) -> None:
    _copy_as_repo(tmp_path, "https://github.com/someone/my-project.git")

    result = _run([sys.executable, "setup_project.py"], tmp_path)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "src" / "my_project" / "ml" / "train.py").exists()
    assert not (tmp_path / "src" / "ai_lab").exists()
    assert not (tmp_path / "setup_project.py").exists()
    assert not (tmp_path / "SETUP.md").exists()
    assert 'name = "my-project"' in (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    notebook = json.loads((tmp_path / "notebooks" / "01_gpu_check.ipynb").read_text(encoding="utf-8"))
    assert 'REPO = "someone/my-project"' in "".join(notebook["cells"][1]["source"])

    tests = _run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], tmp_path)
    assert tests.returncode == 0, tests.stdout


def test_setup_refuses_to_run_on_the_template_itself(tmp_path: Path) -> None:
    _copy_as_repo(tmp_path, "https://github.com/dgelbier23/ml-project-template.git")

    result = _run([sys.executable, "setup_project.py"], tmp_path)

    assert result.returncode != 0
    assert (tmp_path / "src" / "ai_lab").exists()
