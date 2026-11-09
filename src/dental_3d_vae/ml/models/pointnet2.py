"""PointNet++ encoder (Qi et al., 2017b), single-scale grouping."""

import torch
import torch.nn.functional as F
from torch import nn


def farthest_point_sample(xyz: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Indices (B, num_samples) of evenly spread points of xyz (B, N, 3).

    Greedy: start at point 0, then repeatedly take the point farthest from those already
    taken. Starting at point 0 (as the original CUDA implementation does) makes the
    encoder deterministic; the dissertation started at a random point.
    """
    batch = torch.arange(len(xyz), device=xyz.device)
    chosen = torch.zeros(len(xyz), num_samples, dtype=torch.long, device=xyz.device)
    distance = torch.full(xyz.shape[:2], float("inf"), device=xyz.device)
    farthest = torch.zeros(len(xyz), dtype=torch.long, device=xyz.device)
    for i in range(num_samples):
        chosen[:, i] = farthest
        distance = torch.minimum(distance, ((xyz - xyz[batch, farthest].unsqueeze(1)) ** 2).sum(-1))
        farthest = distance.argmax(dim=-1)
    return chosen


def _gather(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """points (B, N, C) at indices (B, ...) -> (B, ..., C)."""
    batch = torch.arange(len(points), device=points.device).view(-1, *[1] * (indices.ndim - 1))
    return points[batch, indices]


def ball_query(radius: float, num_neighbours: int, xyz: torch.Tensor, centres: torch.Tensor) -> torch.Tensor:
    """Indices (B, S, num_neighbours) of each centre's nearest points within ``radius``.

    Slots with no point in range repeat the centre's nearest point.
    """
    distance = torch.cdist(centres, xyz)  # (B, S, N), Euclidean
    nearest = distance.argsort(dim=-1)[..., :num_neighbours]
    inside = distance.gather(2, nearest) <= radius
    return torch.where(inside, nearest, nearest[..., :1].expand_as(nearest))


class SetAbstraction(nn.Module):
    """Choose ``num_centres`` centres by farthest point sampling, group each centre's
    neighbours, and summarise every group with a shared MLP and a max-pool.

    Each neighbour's input is [its xyz minus the centre's xyz, its features], so
    ``in_features`` = 3 + incoming feature count.
    """

    def __init__(self, num_centres: int, radius: float, num_neighbours: int,
                 in_features: int, widths: tuple[int, ...]) -> None:
        super().__init__()
        self.num_centres, self.radius, self.num_neighbours = num_centres, radius, num_neighbours
        layers = []
        for width in widths:
            layers += [nn.Conv2d(in_features, width, kernel_size=1), nn.BatchNorm2d(width), nn.ReLU()]
            in_features = width
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        centres = _gather(xyz, farthest_point_sample(xyz, self.num_centres))  # (B, S, 3)
        groups = ball_query(self.radius, self.num_neighbours, xyz, centres)  # (B, S, K)
        grouped = torch.cat([_gather(xyz, groups) - centres.unsqueeze(2), _gather(features, groups)], dim=-1)
        summary = self.mlp(grouped.permute(0, 3, 1, 2)).max(dim=-1).values  # (B, C_out, S)
        return centres, summary.transpose(1, 2)


class PointNetPlusPlusEncoder(nn.Module):
    """PointNet++ encoder: maps (B, N, 3) to ``(mu, logvar)``, each (B, latent_dim).

    Three set abstractions: 512 centres (radius 0.2, 32 neighbours), then 128 (radius 0.4,
    64), then one global group of all 128; its 1024 features go through 512 -> 256 and
    linear heads. Kept from the dissertation for its trained weights: the global group
    measures every point relative to one chosen point (the input's first point) rather
    than using absolute coordinates as the paper does. Layer names match the dissertation
    checkpoints.
    """

    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.sa1 = SetAbstraction(512, 0.2, 32, 3 + 3, (64, 64, 128))
        self.sa2 = SetAbstraction(128, 0.4, 64, 3 + 128, (128, 128, 256))
        self.sa3 = SetAbstraction(1, float("inf"), 128, 3 + 256, (256, 512, 1024))
        self.fc1, self.bn_fc1 = nn.Linear(1024, 512), nn.BatchNorm1d(512)
        self.fc2, self.bn_fc2 = nn.Linear(512, 256), nn.BatchNorm1d(256)
        self.fc_mu, self.fc_logvar = nn.Linear(256, latent_dim), nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xyz, features = x, x  # The first features are the coordinates themselves.
        for layer in (self.sa1, self.sa2, self.sa3):
            xyz, features = layer(xyz, features)
        x = features.squeeze(1)  # (B, 1024)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        return self.fc_mu(x), self.fc_logvar(x)
