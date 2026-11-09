"""DGCNN encoder (Wang et al., 2019): edge convolutions over k-nearest-neighbour graphs."""

import torch
import torch.nn.functional as F
from torch import nn


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Indices (B, N, k) of each point's k nearest points in feature space x (B, C, N).

    A point is its own nearest neighbour, as in the reference implementation.
    """
    # -|a - b|^2 = 2 a.b - |a|^2 - |b|^2, for every pair at once.
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    return (-xx - inner - xx.transpose(2, 1)).topk(k=k, dim=-1).indices


def edge_features(x: torch.Tensor, k: int) -> torch.Tensor:
    """For every point and each of its k nearest neighbours, ``[neighbour - point, point]``.

    x (B, C, N) -> (B, 2C, N, k): the difference describes the local shape around the
    point; the point itself keeps its absolute position or feature. Returned contiguous,
    as in the dissertation: the memory layout sets BatchNorm's summation order in
    training, so this keeps training bit-identical to the original code.
    """
    points = x.transpose(2, 1)  # (B, N, C)
    batch = torch.arange(len(x), device=x.device).view(-1, 1, 1)
    neighbours = points[batch, knn(x, k)]  # (B, N, k, C)
    centre = points.unsqueeze(2).expand_as(neighbours)
    return torch.cat((neighbours - centre, centre), dim=3).permute(0, 3, 1, 2).contiguous()


def _edge_conv(in_features: int, out_features: int) -> nn.Sequential:
    """Shared MLP over every edge (kernel-size-1 Conv2d), then BatchNorm and ReLU."""
    return nn.Sequential(
        nn.Conv2d(2 * in_features, out_features, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_features),
        nn.ReLU(),
    )


class DGCNNEncoder(nn.Module):
    """Reduced DGCNN: maps (B, N, 3) to ``(mu, logvar)``, each (B, latent_dim).

    Three edge convolutions take each point to 32, 32 and 64 features. Each one rebuilds
    the neighbour graph from the current features, so after the first, a point's
    neighbours are the points with similar learned features rather than just the closest
    ones (the "dynamic graph"). Each layer max-pools over neighbours; the three layers'
    features (128) are max-pooled over all points, then 128 -> 256 -> 128 and linear
    heads. Narrower than the original DGCNN (64/64/128/256) because building a graph
    compares every pair of points, the main cost. Layer names match the dissertation
    checkpoints.
    """

    def __init__(self, latent_dim: int = 32, k: int = 20) -> None:
        super().__init__()
        self.k = k
        self.conv1, self.conv2, self.conv3 = _edge_conv(3, 32), _edge_conv(32, 32), _edge_conv(32, 64)
        self.fc1, self.bn_fc1 = nn.Linear(128, 256), nn.BatchNorm1d(256)
        self.fc2, self.bn_fc2 = nn.Linear(256, 128), nn.BatchNorm1d(128)
        self.fc_mu, self.fc_logvar = nn.Linear(128, latent_dim), nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)  # (B, 3, N)
        layer_features = []
        for conv in (self.conv1, self.conv2, self.conv3):
            x = conv(edge_features(x, self.k)).max(dim=-1).values  # (B, C, N): max over neighbours
            layer_features.append(x)
        x = torch.cat(layer_features, dim=1).max(dim=2).values  # Global feature: (B, 128)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        return self.fc_mu(x), self.fc_logvar(x)
