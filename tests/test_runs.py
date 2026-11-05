import json
from pathlib import Path

import torch
from torch import nn

from ai_lab.experiments.config import ExperimentConfig, load_config
from ai_lab.experiments.runs import create_run_dir, load_metrics, load_run, log_metrics, save_checkpoint


def test_create_run_dir_saves_config_and_environment(tmp_path: Path) -> None:
    config = ExperimentConfig(run_name="unit_test", seed=3)

    run_dir = create_run_dir(tmp_path, config)

    assert run_dir.parent == tmp_path
    assert run_dir.name.endswith("_unit_test")
    assert load_config(run_dir / "config.json") == config
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    assert {"torch", "git_commit", "git_dirty"} <= set(environment)


def test_metrics_are_appended_in_order(tmp_path: Path) -> None:
    log_metrics(tmp_path, {"epoch": 1, "val_loss": 0.5})
    log_metrics(tmp_path, {"epoch": 2, "val_loss": 0.4})

    assert load_metrics(tmp_path) == [
        {"epoch": 1, "val_loss": 0.5},
        {"epoch": 2, "val_loss": 0.4},
    ]


def test_load_run_returns_everything_saved(tmp_path: Path) -> None:
    run_dir = create_run_dir(tmp_path, ExperimentConfig(run_name="loaded", epochs=2))
    log_metrics(run_dir, {"epoch": 1, "val_accuracy": 0.9})

    run = load_run(run_dir)

    assert run["name"] == run_dir.name
    assert run["config"]["epochs"] == 2
    assert "git_commit" in run["environment"]
    assert run["metrics"] == [{"epoch": 1, "val_accuracy": 0.9}]


def test_checkpoint_restores_model_and_is_overwritten(tmp_path: Path) -> None:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters())

    save_checkpoint(tmp_path, model, optimizer, epoch=1)
    save_checkpoint(tmp_path, model, optimizer, epoch=2)

    checkpoint = torch.load(tmp_path / "checkpoint.pt")
    restored = nn.Linear(3, 2)
    restored.load_state_dict(checkpoint["model"])
    assert checkpoint["epoch"] == 2
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters()))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint.pt"]  # No .tmp left.
