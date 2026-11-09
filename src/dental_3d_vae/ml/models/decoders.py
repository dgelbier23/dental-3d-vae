"""Decoders: latent vector z (B, latent_dim) -> point cloud (B, num_points, 3)."""

import torch
from torch import nn


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
