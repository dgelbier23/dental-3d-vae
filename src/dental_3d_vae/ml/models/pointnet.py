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


class TNet(nn.Module):
    """Predicts a (k x k) transform that aligns its (B, k, N) input; starts as the identity."""

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
        self.conv1, self.bn1 = nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64)
        self.conv2, self.bn2 = nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128)
        self.conv3, self.bn3 = nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024)
        self.fc1, self.bn4 = nn.Linear(1024, 512), nn.BatchNorm1d(512)
        self.fc2, self.bn5 = nn.Linear(512, 256), nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, k * k)
        nn.init.zeros_(self.fc3.weight)  # Output = bias = identity until training moves it.
        with torch.no_grad():
            self.fc3.bias.copy_(torch.eye(k).flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.max(dim=2).values
        x = F.relu(self.bn5(self.fc2(F.relu(self.bn4(self.fc1(x))))))
        return self.fc3(x).view(-1, self.k, self.k)


class TNetPointNetEncoder(nn.Module):
    """PointNet with input and feature transforms (Qi et al., 2017).

    One T-Net aligns the input points (3 x 3); another aligns the 64 per-point features
    (64 x 64) before the rest of the shared MLP, the max-pool and the heads. After each
    forward pass, ``regularization_loss`` holds the batch mean of ||A A^T - I|| (Frobenius)
    for the feature transform A; training adds it to the loss to keep A near orthogonal.
    The paper squares this norm; the dissertation did not. Layer names match the
    dissertation checkpoints.
    """

    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.input_tnet = TNet(3)
        self.conv1, self.bn1 = nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64)
        self.conv2, self.bn2 = nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64)
        self.feature_tnet = TNet(64)
        self.conv3, self.bn3 = nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128)
        self.conv4, self.bn4 = nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024)
        self.fc1, self.bn5 = nn.Linear(1024, 512), nn.BatchNorm1d(512)
        self.fc2, self.bn6 = nn.Linear(512, 256), nn.BatchNorm1d(256)
        self.fc_mu, self.fc_logvar = nn.Linear(256, latent_dim), nn.Linear(256, latent_dim)
        self.regularization_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)  # (B, 3, N)
        x = torch.bmm(self.input_tnet(x), x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        transform = self.feature_tnet(x)  # (B, 64, 64)
        x = torch.bmm(transform, x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.max(dim=2).values  # Global feature: (B, 1024)
        x = F.relu(self.bn6(self.fc2(F.relu(self.bn5(self.fc1(x))))))
        identity = torch.eye(transform.shape[1], device=transform.device, dtype=transform.dtype)
        self.regularization_loss = torch.linalg.matrix_norm(transform @ transform.transpose(1, 2) - identity).mean()
        return self.fc_mu(x), self.fc_logvar(x)
