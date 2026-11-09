"""Read the dissertation's checkpoints into the same formats as this repo's runs.

Every results chart then works unchanged on the 16 dissertation runs and on new ones.
"""

from pathlib import Path

import pandas as pd
import torch

# Dissertation key -> this repo's name.
_RESULTS = {"test_deterministic_chamfer_per_sample": "cd", "test_stochastic_chamfer_per_sample": "stochastic_cd",
            "test_mrre_per_sample": "mrre", "test_orpe_per_sample": "orpe"}
_HISTORY = {"train_total": "train_total", "train_stochastic_chamfer": "train_chamfer", "train_kl": "train_kl",
            "train_feature_transform": "train_feature_transform", "val_deterministic_chamfer": "val_chamfer",
            "val_kl": "val_kl"}


def load_dissertation_run(path: Path) -> dict:
    """``{"name", "config", "results", "metrics", "summary"}`` from a dissertation checkpoint.

    ``results`` matches ``evaluate``'s output, but with no ``sce`` (never stored), and
    with ``orpe`` measured against the upsampled input (the dissertation's definition,
    0.3-0.4% below this repo's). ``metrics`` matches metrics.jsonl: one dict per epoch.
    Unpickles arbitrary objects, so only load trusted files.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    results = checkpoint["test_metadata"].reset_index(drop=True)
    for old, new in _RESULTS.items():
        if checkpoint[old] is not None:
            results[new] = checkpoint[old]
    history = pd.DataFrame({new: checkpoint["history"][old] for old, new in _HISTORY.items()})
    history.insert(0, "epoch", range(1, len(history) + 1))
    return {
        "name": path.stem,
        "config": checkpoint["config"],
        "results": results,
        "metrics": history.to_dict("records"),
        "summary": {"best_epoch": checkpoint["best_epoch"], "epochs_trained": checkpoint["epochs_trained"],
                    "num_parameters": checkpoint["num_parameters"],
                    "training_minutes": checkpoint["training_time_minutes"], "gpu": checkpoint["gpu_name"],
                    "test_kl": checkpoint["verified_kl"]},
    }
