"""Decoders: latent vector z (B, latent_dim) -> point cloud (B, num_points, 3)."""

import math

import torch
from torch import nn


def _square_grid(side: int, low: float, high: float) -> torch.Tensor:
    """(side * side, 2) evenly spaced 2D points covering [low, high] x [low, high]."""
    axis = torch.linspace(low, high, side)
    u, v = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([u.reshape(-1), v.reshape(-1)], dim=1)


class MLPDecoder(nn.Module):
    """Fully connected decoder that predicts every coordinate directly.

    Each output point is an independent set of three numbers, so nothing makes
    neighbouring points form a surface. Layer names match the dissertation checkpoints.
    """

    def __init__(self, latent_dim: int = 32, num_points: int = 2048) -> None:
        super().__init__()
        self.num_points = num_points
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, num_points * 3),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, self.num_points, 3)


def _folding(in_features: int) -> nn.Sequential:
    """Shared per-point MLP (kernel-size-1 Conv1d) mapping each point's input to 3D."""
    return nn.Sequential(
        nn.Conv1d(in_features, 512, 1), nn.ReLU(),
        nn.Conv1d(512, 512, 1), nn.ReLU(),
        nn.Conv1d(512, 3, 1),
    )


class FoldingNetDecoder(nn.Module):
    """FoldingNet-style decoder (Yang et al., 2018): folds a flat 2D grid into a surface.

    Every grid point, paired with z, goes through a shared MLP to a 3D point (first
    fold); each 3D point, paired with z again, is refined by a second shared MLP. Because
    nearby grid points get nearby inputs, the output tends to form a continuous surface.

    The grid is the largest square that fits (45 x 45 = 2,025 for 2,048 points), padded by
    repeating its first points, so a few output points are duplicates. It is stored in
    checkpoints as ``grid``; layer names match the dissertation checkpoints.
    """

    def __init__(self, latent_dim: int = 32, num_points: int = 2048) -> None:
        super().__init__()
        self.num_points = num_points
        grid = _square_grid(math.isqrt(num_points), -1.0, 1.0)
        self.register_buffer("grid", grid[torch.arange(num_points) % len(grid)])  # (num_points, 2)
        self.fold1 = _folding(latent_dim + 2)
        self.fold2 = _folding(latent_dim + 3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        latent = z.unsqueeze(2).expand(-1, -1, self.num_points)  # (B, latent_dim, N)
        grid = self.grid.t().unsqueeze(0).expand(len(z), -1, -1)  # (B, 2, N)
        folded = self.fold1(torch.cat([grid, latent], dim=1))  # (B, 3, N)
        return self.fold2(torch.cat([folded, latent], dim=1)).transpose(1, 2)


class _AtlasPatch(nn.Module):
    """One AtlasNet patch: maps [u, v, z] to a 3D point with its own MLP."""

    def __init__(self, latent_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 3),
        )


class AtlasNetDecoder(nn.Module):
    """AtlasNet-style decoder (Groueix et al., 2018): the surface as several deformed squares.

    Each of ``num_patches`` patches has its own MLP that maps every point of a shared
    2D grid, paired with z, to 3D; the patches' points are concatenated. Each patch's grid
    is the smallest square covering its share of points, truncated (23 x 23 cut to 512 for
    2,048 points), stored in checkpoints as ``uv_grid``. Layer names match the
    dissertation checkpoints.
    """

    def __init__(self, latent_dim: int = 32, num_points: int = 2048, num_patches: int = 4) -> None:
        super().__init__()
        per_patch = num_points // num_patches
        self.register_buffer("uv_grid", _square_grid(math.isqrt(per_patch - 1) + 1, 0.0, 1.0)[:per_patch])
        self.patches = nn.ModuleList(_AtlasPatch(latent_dim) for _ in range(num_patches))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        uv = self.uv_grid.unsqueeze(0).expand(len(z), -1, -1)  # (B, P, 2)
        x = torch.cat([uv, z.unsqueeze(1).expand(-1, len(self.uv_grid), -1)], dim=-1)  # (B, P, 2 + latent)
        return torch.cat([patch.mlp(x) for patch in self.patches], dim=1)
