
# AI Lab

A development workspace for machine learning and deep learning.

## Environments

- VS Code: primary editor.
- Claude Code: AI-assisted development.
- Local Python: CPU development and testing.
- Google Colab: cloud GPU experiments.
- GitHub: version control and collaboration.

## Setup

From the repo folder in PowerShell (no need to activate the environment):

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m pytest -q
```

In VS Code, select the `.venv` kernel for notebooks.

## Everyday workflow

1. Edit code in VS Code and run the tests.
2. Commit and push to GitHub.
3. In Colab, use Runtime > Restart session, then run all cells.
4. Results save to Google Drive. Download a run folder into `artifacts/runs/`
   to compare it locally with `notebooks/04_compare_runs.ipynb`.

## Notebooks

Notebooks run locally (`.venv` kernel) or in Google Colab, unchanged.

- The first code cell of every notebook is the same bootstrap cell. In Colab it
  clones the latest `main` from GitHub using the `GITHUB_TOKEN` Colab secret
  (fine-grained, read-only); locally it does nothing.
- In Colab, open notebooks via File > Open notebook > GitHub, pick a GPU runtime,
  and treat them as read-only: edit in VS Code, push, then re-run in Colab.
- Run outputs go to `get_runs_dir()`: `MyDrive/<repo name>/runs` in Colab (pass a
  different subfolder to change it), `artifacts/runs` locally (ignored by Git).

## Reproducibility

### PyTorch version parity

Local `torch` is pinned in `requirements.txt` to Colab's preinstalled version
(currently `2.11.0`; Colab ships it with CUDA 13.0). Local uses the CPU build.

- Never `pip install torch` in Colab; use the version Colab provides.
- When Colab's version changes (shown by `notebooks/01_gpu_check.ipynb`),
  update the pin and reinstall locally:

```powershell
pip install "torch==<version>" --index-url https://download.pytorch.org/whl/cpu
```

### Experiments

Record important dependency versions, random seeds,
dataset versions, and experiment configuration.

Never commit secrets or large generated model artifacts.
