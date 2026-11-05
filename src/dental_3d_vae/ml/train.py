"""Model training."""

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Train for one pass over ``loader``; return mean per-sample training loss.

    Assumes ``loss_fn`` averages over the batch (the PyTorch default).
    """
    model.train()
    total_loss, count = 0.0, 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(features), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        count += len(labels)
    return total_loss / count
