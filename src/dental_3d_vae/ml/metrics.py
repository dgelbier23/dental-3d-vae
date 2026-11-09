"""Reconstruction metrics for (B, N, 3) point clouds, one value per tooth: shape (B,).

Chamfer distance uses plain Euclidean distances, as in all the dissertation's results;
MRRE and ORPE use squared distances, so their values are not on the same scale as CD.
Above 25 points torch.cdist computes through matrix products, so near-zero distances
come out as ~1e-4 rather than 0 (the dissertation's values include the same effect).
"""

from itertools import combinations

import torch


def nearest_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(B, N): Euclidean distance from each point of a (B, N, 3) to its nearest point in b (B, M, 3)."""
    return torch.cdist(a, b).min(dim=-1).values


def chamfer_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Symmetric Chamfer distance: mean nearest distance a -> b plus b -> a.

    Its batch mean is the dissertation's training loss.
    """
    return nearest_distance(a, b).mean(dim=-1) + nearest_distance(b, a).mean(dim=-1)


def _region_error(reconstruction: torch.Tensor, complete: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    """Mean squared distance from the complete tooth's points in ``region`` (B, N bool) to the reconstruction."""
    squared = nearest_distance(complete, reconstruction) ** 2
    return (squared * region).sum(dim=-1) / region.sum(dim=-1)


def missing_region_error(reconstruction: torch.Tensor, complete: torch.Tensor, removed: torch.Tensor) -> torch.Tensor:
    """MRRE: how well the reconstruction recovers the removed points. NaN where nothing was removed."""
    return _region_error(reconstruction, complete, removed)


def observed_region_error(reconstruction: torch.Tensor, complete: torch.Tensor, removed: torch.Tensor) -> torch.Tensor:
    """ORPE: how well the reconstruction keeps the observed (not removed) points.

    The dissertation measured this against the upsampled model input instead (mean
    0.3-0.4% lower on its test sets).
    """
    return _region_error(reconstruction, complete, ~removed)


def stochastic_consistency_error(samples: torch.Tensor) -> torch.Tensor:
    """SCE: mean Chamfer distance over all pairs of K reconstructions (K, B, N, 3) of the same teeth."""
    pairs = combinations(range(len(samples)), 2)
    return torch.stack([chamfer_distance(samples[i], samples[j]) for i, j in pairs]).mean(dim=0)
