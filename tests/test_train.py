import json
from dataclasses import replace

import pytest
import torch

from dental_3d_vae.experiments.config import ExperimentConfig
from dental_3d_vae.experiments.runs import load_metrics
from dental_3d_vae.ml.models.vae import build_vae, load_checkpoint
from dental_3d_vae.ml import train
from dental_3d_vae.ml.train import fit, kl_divergence, train_one_epoch

CPU = torch.device("cpu")
CONFIG = ExperimentConfig(reconstruction_mode="complete", encoder="simple_pointnet", decoder="mlp",
                          latent_dim=4, num_points=16, epochs=8, patience=2)


def _loader(batches: int = 2) -> list[dict]:
    """Stands in for a DataLoader: batches of complete teeth."""
    generator = torch.Generator().manual_seed(0)
    return [{"complete": torch.randn(4, 16, 3, generator=generator)} for _ in range(batches)]


def test_kl_divergence_is_zero_only_for_the_prior() -> None:
    assert kl_divergence(torch.zeros(2, 4), torch.zeros(2, 4)) == 0
    assert kl_divergence(torch.ones(2, 4), torch.zeros(2, 4)) == pytest.approx(0.5)  # 0.5 * mu^2


def test_training_loss_adds_the_weighted_terms() -> None:
    config = replace(CONFIG, encoder="tnet_pointnet", kl_weight=0.1, feature_transform_weight=0.01)
    model = build_vae(config.encoder, config.decoder, config.latent_dim, config.num_points)

    # Two batches: the T-Net regulariser is 0 until the first update moves the transforms from the identity.
    losses = train_one_epoch(model, _loader(), torch.optim.Adam(model.parameters()), config, CPU)

    expected = losses["train_chamfer"] + 0.1 * losses["train_kl"] + 0.01 * losses["train_feature_transform"]
    assert losses["train_total"] == pytest.approx(expected)
    assert losses["train_feature_transform"] > 0


def test_fit_stops_early_and_keeps_the_best_model(tmp_path, monkeypatch) -> None:
    scores = iter([3.0, 1.0, 2.0, 1.5, 0.5])  # Best at epoch 2, then no improvement for 2 epochs.
    monkeypatch.setattr(train, "validate", lambda *args: {"val_chamfer": next(scores), "val_kl": 0.0})
    torch.manual_seed(0)
    model = build_vae(CONFIG.encoder, CONFIG.decoder, CONFIG.latent_dim, CONFIG.num_points)

    summary = fit(model, {"train": _loader(), "val": _loader(1)}, CONFIG, CPU, tmp_path)

    assert (summary["best_epoch"], summary["best_val_chamfer"], summary["epochs_trained"]) == (2, 1.0, 4)
    assert summary["early_stopped"] and len(load_metrics(tmp_path)) == 4
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == summary
    saved, checkpoint = load_checkpoint(tmp_path / "model.pt")
    assert checkpoint["epoch"] == 2
    assert all(torch.equal(a, b) for a, b in zip(saved.state_dict().values(), model.state_dict().values()))
