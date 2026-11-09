"""Experiment configuration: one object that fully describes a run."""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Every setting that defines a run. Saved alongside the run's results.

    Defaults are the dissertation's final model (P0: DGCNN + FoldingNet, trained to
    complete plane-cut teeth). Field names match the dissertation's configs. Frozen
    so it cannot change mid-run; create variants with
    ``dataclasses.replace(config, kl_weight=0.01)``.
    """

    run_name: str = "P0_dgcnn_foldingnet_plane"
    seed: int = 42
    deterministic: bool = False
    # Data: the encoder sees "complete" or "partial" teeth; the target is always complete.
    reconstruction_mode: str = "partial"
    method: str = "plane"  # Partial runs only: "plane" or "connected_patch".
    num_points: int = 2048
    # Model
    encoder: str = "dgcnnv2"
    decoder: str = "foldingnet"
    latent_dim: int = 32
    # Loss: Chamfer distance + kl_weight * KL + feature_transform_weight * T-Net regulariser
    kl_weight: float = 0.001  # beta
    feature_transform_weight: float = 0.0  # Only the tnet_pointnet encoder has the regulariser.
    # Training: stop after `patience` epochs without a better validation Chamfer distance.
    batch_size: int = 32
    epochs: int = 300
    patience: int = 20
    learning_rate: float = 1e-3

    def __post_init__(self) -> None:
        # Typos here would otherwise train the wrong model for an hour.
        if self.reconstruction_mode not in ("complete", "partial"):
            raise ValueError(f"reconstruction_mode must be 'complete' or 'partial', not {self.reconstruction_mode!r}")
        if self.method not in ("plane", "connected_patch"):
            raise ValueError(f"method must be 'plane' or 'connected_patch', not {self.method!r}")


def save_config(config: ExperimentConfig, path: Path) -> None:
    """Write the config as human-readable JSON."""
    path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def load_config(path: Path) -> ExperimentConfig:
    """Read a saved config. Unknown keys (e.g. typos) raise TypeError."""
    return ExperimentConfig(**json.loads(path.read_text(encoding="utf-8")))


def config_from_checkpoint(saved: dict) -> ExperimentConfig:
    """The ``ExperimentConfig`` of a saved model's ``checkpoint["config"]``.

    Accepts this repo's model.pt and the dissertation's checkpoints, whose ``name``
    becomes ``run_name`` and whose settings this repo no longer uses are dropped.
    """
    known = {field.name for field in fields(ExperimentConfig)}
    settings = {key: value for key, value in saved.items() if key in known}
    if "name" in saved:
        settings["run_name"] = saved["name"]
    return ExperimentConfig(**settings)
