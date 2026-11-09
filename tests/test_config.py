import dataclasses
import json
from pathlib import Path

import pytest

from dental_3d_vae.experiments.config import ExperimentConfig, load_config, save_config


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    config = ExperimentConfig(run_name="round_trip", seed=7, epochs=3)
    path = tmp_path / "config.json"

    save_config(config, path)

    assert load_config(path) == config


def test_load_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"learnig_rate": 0.1}), encoding="utf-8")

    with pytest.raises(TypeError, match="learnig_rate"):
        load_config(path)


def test_load_fills_missing_keys_with_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"seed": 5}), encoding="utf-8")

    config = load_config(path)

    assert config.seed == 5
    assert config.epochs == ExperimentConfig().epochs


def test_config_is_immutable() -> None:
    config = ExperimentConfig()

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.epochs = 99  # type: ignore[misc]


def test_config_rejects_unknown_modes_and_methods() -> None:
    with pytest.raises(ValueError, match="reconstruction_mode"):
        ExperimentConfig(reconstruction_mode="partal")
    with pytest.raises(ValueError, match="method"):
        ExperimentConfig(method="planes")
