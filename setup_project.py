"""One-time setup for a new project created from this template. Deletes itself when done.

Names come from the Git remote: github.com/you/my-project gets the package my_project.
"""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATE_REPO = "dgelbier23/ml-project-template"
TEMPLATE_ONLY_FILES = ("setup_project.py", "SETUP.md", "tests/test_setup_project.py")


def git(*args: str) -> str:
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def main() -> None:
    repo = re.search(r"github\.com[:/](.+?)(?:\.git)?$", git("remote", "get-url", "origin")).group(1)
    if repo == TEMPLATE_REPO:
        raise SystemExit("This is the template itself. Run this in a repo made with 'Use this template'.")

    name = repo.split("/")[1]
    package = re.sub(r"\W", "_", name).lower()
    if not package.isidentifier():
        raise SystemExit(f"Cannot make a Python package name from '{name}'. Rename the repo.")

    replacements = {TEMPLATE_REPO: repo, "ai_lab": package, "ai-lab": name, "AI Lab": name}
    for relative in git("ls-files").splitlines():
        path = ROOT / relative
        if relative in TEMPLATE_ONLY_FILES or not path.exists():
            continue
        text = path.read_bytes().decode("utf-8")
        for old, new in replacements.items():
            text = text.replace(old, new)
        path.write_bytes(text.encode("utf-8"))

    (ROOT / "src" / "ai_lab").rename(ROOT / "src" / package)
    for relative in TEMPLATE_ONLY_FILES:
        (ROOT / relative).unlink()
    print(f"Set up '{repo}' with package '{package}'. Review the changes, then commit them.")


if __name__ == "__main__":
    main()
