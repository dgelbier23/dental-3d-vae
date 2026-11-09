"""VAE training, as in the dissertation.

Loss = Chamfer distance (sampled reconstruction vs complete tooth) + kl_weight * KL
+ feature_transform_weight * T-Net regulariser, minimised with Adam. The best model is
chosen by validation Chamfer distance of the deterministic (z = mu) reconstruction.
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dental_3d_vae.experiments.config import ExperimentConfig
from dental_3d_vae.experiments.runs import log_metrics, save_model
from dental_3d_vae.ml.metrics import chamfer_distance
from dental_3d_vae.ml.models.vae import VAE


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)), averaged over the batch and the latent dimensions.

    The mean over latent dimensions (the dissertation's form) makes it 1/latent_dim of
    the usual per-tooth sum, so kl_weight is smaller than the equivalent beta in papers.
    """
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()


def model_input_and_target(batch: dict, config: ExperimentConfig,
                           device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """A batch's (model input, complete target) on ``device``: the input is partial or complete per config."""
    model_input = batch["partial" if config.reconstruction_mode == "partial" else "complete"]
    return model_input.to(device, non_blocking=True), batch["complete"].to(device, non_blocking=True)


def train_one_epoch(model: VAE, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    config: ExperimentConfig, device: torch.device) -> dict[str, float]:
    """One pass over ``loader``; returns the mean of each loss term over batches."""
    model.train()
    totals = defaultdict(float)
    for batch in loader:
        model_input, complete = model_input_and_target(batch, config, device)
        optimizer.zero_grad(set_to_none=True)
        reconstruction, mu, logvar = model(model_input)
        terms = {
            "chamfer": chamfer_distance(reconstruction, complete).mean(),
            "kl": kl_divergence(mu, logvar),
            "feature_transform": getattr(model.encoder, "regularization_loss", torch.zeros((), device=device)),
        }
        loss = (terms["chamfer"] + config.kl_weight * terms["kl"]
                + config.feature_transform_weight * terms["feature_transform"])
        loss.backward()
        optimizer.step()
        for name, value in {"total": loss, **terms}.items():
            totals[name] += value.item()
    return {f"train_{name}": total / len(loader) for name, total in totals.items()}


@torch.no_grad()
def validate(model: VAE, loader: DataLoader, config: ExperimentConfig, device: torch.device) -> dict[str, float]:
    """Mean over batches of the deterministic (z = mu) Chamfer distance and of the KL."""
    model.eval()
    chamfer = kl = 0.0
    for batch in loader:
        model_input, complete = model_input_and_target(batch, config, device)
        mu, logvar = model.encoder(model_input)
        chamfer += chamfer_distance(model.decoder(mu), complete).mean().item()
        kl += kl_divergence(mu, logvar).item()
    return {"val_chamfer": chamfer / len(loader), "val_kl": kl / len(loader)}


def fit(model: VAE, loaders: dict[str, DataLoader], config: ExperimentConfig,
        device: torch.device, run_dir: Path) -> dict:
    """Train until ``config.patience`` epochs pass without a lower validation Chamfer distance.

    Each epoch is logged to metrics.jsonl and every new best model is saved to model.pt
    straight away. Ends with the best weights loaded into ``model``; returns the run
    summary, also written to summary.json.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_chamfer, best_epoch, start = float("inf"), 0, time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        metrics = {"epoch": epoch, **train_one_epoch(model, loaders["train"], optimizer, config, device),
                   **validate(model, loaders["val"], config, device)}
        metrics["seconds"] = time.perf_counter() - epoch_start  # .item() calls have synchronised the GPU.
        log_metrics(run_dir, metrics)
        if metrics["val_chamfer"] < best_chamfer:
            best_chamfer, best_epoch = metrics["val_chamfer"], epoch
            save_model(run_dir, model, config, epoch)
        print(f"Epoch {epoch:3d} | train loss {metrics['train_total']:.5f} | val CD {metrics['val_chamfer']:.5f}"
              f" | best {best_chamfer:.5f} (epoch {best_epoch}) | {metrics['seconds']:.0f} s")
        if epoch - best_epoch >= config.patience:
            break

    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device)["model_state_dict"])
    summary = {
        "best_epoch": best_epoch,
        "best_val_chamfer": best_chamfer,
        "epochs_trained": epoch,
        "early_stopped": epoch < config.epochs,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "training_minutes": (time.perf_counter() - start) / 60,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
