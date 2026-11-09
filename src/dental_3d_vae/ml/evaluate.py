"""Test-set evaluation: one row of metrics per tooth."""

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dental_3d_vae.experiments.config import ExperimentConfig
from dental_3d_vae.ml.metrics import (
    chamfer_distance,
    missing_region_error,
    observed_region_error,
    stochastic_consistency_error,
)
from dental_3d_vae.ml.models.vae import VAE
from dental_3d_vae.ml.train import model_input_and_target


@torch.no_grad()
def evaluate(model: VAE, loader: DataLoader, config: ExperimentConfig, device: torch.device,
             num_samples: int = 10, seed: int = 42) -> pd.DataFrame:
    """The loader's metadata (its rows in order, so the loader must not shuffle) plus metrics.

    ``cd``, and for partial inputs ``mrre`` and ``orpe``, score the deterministic
    reconstruction (z = mu); ``stochastic_cd`` scores one sampled reconstruction; ``sce``
    compares ``num_samples`` of them. Sampling uses its own generator seeded with
    ``seed``, so results do not depend on the global random state; GPUs and CPUs still
    draw different numbers.
    """
    model.to(device).eval()
    generator = torch.Generator(device).manual_seed(seed)
    metrics = []
    for batch in loader:
        model_input, complete = model_input_and_target(batch, config, device)
        mu, logvar = model.encoder(model_input)
        reconstruction = model.decoder(mu)
        samples = torch.stack([model.decoder(model.reparameterize(mu, logvar, generator))
                               for _ in range(num_samples)])
        batch_metrics = {
            "cd": chamfer_distance(reconstruction, complete),
            "stochastic_cd": chamfer_distance(samples[0], complete),
            "sce": stochastic_consistency_error(samples),
        }
        if config.reconstruction_mode == "partial":
            removed = batch["removed"].to(device)
            batch_metrics["mrre"] = missing_region_error(reconstruction, complete, removed)
            batch_metrics["orpe"] = observed_region_error(reconstruction, complete, removed)
        metrics.append(pd.DataFrame({name: values.cpu().numpy() for name, values in batch_metrics.items()}))
    return pd.concat([loader.dataset.metadata, pd.concat(metrics, ignore_index=True)], axis=1)


def summarise(results: dict[str, pd.DataFrame], metric: str = "cd") -> pd.DataFrame:
    """One row per model: mean, median, interquartile range, min and max of ``metric`` over teeth."""
    def stats(values: pd.Series) -> pd.Series:
        return pd.Series({"Mean": values.mean(), "Median": values.median(),
                          "IQR": values.quantile(0.75) - values.quantile(0.25),
                          "Min": values.min(), "Max": values.max()})
    return pd.DataFrame({label: stats(frame[metric]) for label, frame in results.items()}).T
