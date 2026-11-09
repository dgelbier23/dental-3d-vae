"""PointNet-style encoders: a shared per-point MLP, then a max-pool over all points."""

import torch
import torch.nn.functional as F
from torch import nn


class PointNetEncoder(nn.Module):
    """Simplified PointNet (Qi et al., 2017), the baseline encoder.

    Maps (B, N, 3) to ``(mu, logvar)``, each (B, latent_dim). The max-pool makes the
    output independent of point order. Layer names match the dissertation checkpoints.
    """

    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        # Shared per-point MLP: a kernel-size-1 Conv1d applies the same layer to every point.
        self.conv1, self.bn1 = nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64)
        self.conv2, self.bn2 = nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128)
        self.conv3, self.bn3 = nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256)
        self.conv4, self.bn4 = nn.Conv1d(256, 512, 1), nn.BatchNorm1d(512)
        self.conv5, self.bn5 = nn.Conv1d(512, 1024, 1), nn.BatchNorm1d(1024)
        self.fc1, self.fc2 = nn.Linear(1024, 512), nn.Linear(512, 256)
        self.fc_mu, self.fc_logvar = nn.Linear(256, latent_dim), nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)  # (B, 3, N): Conv1d wants channels first.
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))  # (B, 1024, N)
        x = x.max(dim=2).values  # Global feature: (B, 1024)
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.fc_mu(x), self.fc_logvar(x)
