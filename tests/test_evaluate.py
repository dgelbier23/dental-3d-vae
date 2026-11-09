from dataclasses import replace

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dental_3d_vae.experiments.config import ExperimentConfig
from dental_3d_vae.ml.data import CompleteToothDataset, PartialToothDataset
from dental_3d_vae.ml.evaluate import evaluate
from dental_3d_vae.ml.metrics import chamfer_distance
from dental_3d_vae.ml.models.vae import build_vae
from dental_3d_vae.ml.partial import build_partial_dataset

CPU = torch.device("cpu")
CONFIG = ExperimentConfig(encoder="simple_pointnet", decoder="mlp", latent_dim=4, num_points=32)


def _teeth(tmp_path) -> pd.DataFrame:
    """Three 32-point teeth on disk; returns their partials' metadata."""
    points = np.random.default_rng(0).random((3, 32, 3), dtype=np.float32)
    with h5py.File(tmp_path / "teeth_dataset.h5", "w") as f:
        f.create_dataset("points", data=points)
    pd.DataFrame({"sample_id": range(3), "tooth_id": [11, 13, 16],
                  "tooth_type": ["incisor", "canine", "molar"]}).to_csv(tmp_path / "metadata.csv", index=False)
    return build_partial_dataset(tmp_path, tmp_path, methods=("plane",))


def test_evaluate_partial_run_gives_one_row_of_metrics_per_tooth(tmp_path) -> None:
    partials = _teeth(tmp_path)
    loader = DataLoader(PartialToothDataset(tmp_path / "teeth_dataset.h5", tmp_path / "partial_masks.h5",
                                            partials), batch_size=5)
    torch.manual_seed(0)
    model = build_vae(CONFIG.encoder, CONFIG.decoder, CONFIG.latent_dim, CONFIG.num_points).eval()

    results = evaluate(model, loader, CONFIG, CPU, num_samples=3)

    assert results[list(partials.columns)].equals(partials.reset_index(drop=True))  # metadata, in order
    assert results["mrre"].isna().tolist() == (partials["missing_fraction"] == 0).tolist()
    first = loader.dataset[0]
    with torch.no_grad():
        expected = chamfer_distance(model.reconstruct(first["partial"][None]), first["complete"][None])
    assert np.isclose(results["cd"][0], expected.item())
    assert (results[["stochastic_cd", "sce", "orpe"]] > 0).all().all()


def test_evaluate_is_seeded_and_complete_runs_have_no_region_errors(tmp_path) -> None:
    _teeth(tmp_path)
    metadata = pd.read_csv(tmp_path / "metadata.csv")
    loader = DataLoader(CompleteToothDataset(tmp_path / "teeth_dataset.h5", metadata), batch_size=2)
    config = replace(CONFIG, reconstruction_mode="complete")
    model = build_vae(config.encoder, config.decoder, config.latent_dim, config.num_points)

    first, again, other = (evaluate(model, loader, config, CPU, num_samples=3, seed=s) for s in (1, 1, 2))

    assert first.equals(again)
    assert first["cd"].equals(other["cd"]) and not first["sce"].equals(other["sce"])
    assert "mrre" not in first and "orpe" not in first
