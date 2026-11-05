"""Experiment configuration: one object that fully describes a run."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Every setting that defines a run. Saved alongside the run's results.

    Frozen so it cannot change mid-run; create variants with
    ``dataclasses.replace(config, epochs=20)``.
    """

    run_name: str = "synthetic_baseline"
    seed: int = 42
    deterministic: bool = False
    # Data
    n_samples: int = 2000
    input_dim: int = 20
    num_classes: int = 2
    val_fraction: float = 0.2
    # Model
    hidden_dim: int = 64
    # Training
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-3


def save_config(config: ExperimentConfig, path: Path) -> None:
    """Write the config as human-readable JSON."""
    path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def load_config(path: Path) -> ExperimentConfig:
    """Read a saved config. Unknown keys (e.g. typos) raise TypeError."""
    return ExperimentConfig(**json.loads(path.read_text(encoding="utf-8")))
