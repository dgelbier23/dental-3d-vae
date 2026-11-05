"""Run folders: one self-contained directory of outputs per experiment run."""

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from dental_3d_vae.experiments.config import ExperimentConfig, save_config
from dental_3d_vae.runtime.env import describe_environment


def create_run_dir(runs_dir: Path, config: ExperimentConfig) -> Path:
    """Create ``<runs_dir>/<UTC timestamp>_<run_name>/`` holding config.json and environment.json."""
    run_dir = runs_dir / f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}_{config.run_name}"
    run_dir.mkdir(parents=True)
    save_config(config, run_dir / "config.json")
    (run_dir / "environment.json").write_text(
        json.dumps(describe_environment(), indent=2), encoding="utf-8"
    )
    return run_dir


def log_metrics(run_dir: Path, metrics: dict) -> None:
    """Append one line of metrics to metrics.jsonl. Earlier epochs survive a crash."""
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(metrics) + "\n")


def load_metrics(run_dir: Path) -> list[dict]:
    """Read metrics.jsonl back as a list of dicts, one per logged line."""
    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def load_run(run_dir: Path) -> dict:
    """Everything saved for one run: name, config, environment and per-epoch metrics.

    The config is returned as a plain dict, so runs saved by older code still load.
    """

    def read_json(name: str) -> dict:
        return json.loads((run_dir / name).read_text(encoding="utf-8"))

    return {
        "name": run_dir.name,
        "config": read_json("config.json"),
        "environment": read_json("environment.json"),
        "metrics": load_metrics(run_dir),
    }


def save_checkpoint(
    run_dir: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int
) -> None:
    """Overwrite checkpoint.pt with the latest model and optimizer state.

    Writes to a temporary file first, so a disconnect mid-save cannot leave a
    corrupted checkpoint behind.
    """
    path = run_dir / "checkpoint.pt"
    tmp_path = path.with_suffix(".tmp")
    state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(state, tmp_path)
    tmp_path.replace(path)
