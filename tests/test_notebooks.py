import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_notebook_bootstraps_this_repo() -> None:
    # A wrong REPO fails silently: Colab would run another repo's code.
    bootstraps = set()
    for path in (ROOT / "notebooks").glob("*.ipynb"):
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        bootstraps.add("".join(next(c for c in cells if c["cell_type"] == "code")["source"]))
    origin = subprocess.run(
        ["git", "-C", str(ROOT), "remote", "get-url", "origin"], capture_output=True, text=True
    ).stdout.strip()

    assert len(bootstraps) == 1
    repo = re.search(r'REPO = "(.+?)"', bootstraps.pop()).group(1)
    assert origin.removesuffix(".git").endswith(repo)


def test_notebooks_have_no_outputs() -> None:
    # Outputs can contain Teeth3DS+ data, which must not be published.
    # Fix: VS Code "Clear All Outputs", then save.
    with_outputs = [
        path.name
        for path in (ROOT / "notebooks").glob("*.ipynb")
        if any(c.get("outputs") for c in json.loads(path.read_text(encoding="utf-8"))["cells"])
    ]
    assert with_outputs == []
