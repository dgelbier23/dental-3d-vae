"""Runtime utilities shared by notebooks, scripts and tests."""

import os
import random

import numpy as np
import torch


def get_device() -> torch.device:
    """Return the best available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch so runs are repeatable.

    Seeding alone does not make GPU results bit-identical. ``deterministic=True``
    forces deterministic algorithms: slower, raises on operations that have no
    deterministic version, and stays on for the whole process. Call it before
    any CUDA work, because the cuBLAS setting is ignored once CUDA has started.
    """
    random.seed(seed)
    np.random.seed(seed)  # Raises ValueError unless 0 <= seed < 2**32.
    torch.manual_seed(seed)  # Also seeds every CUDA device.

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
