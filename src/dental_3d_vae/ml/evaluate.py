"""Model evaluation."""

import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device
) -> dict[str, float]:
    """Return mean per-sample loss and accuracy over ``loader``.

    Assumes ``loss_fn`` averages over the batch (the PyTorch default).
    Leaves the model in eval mode; the training loop calls ``model.train()``.
    """
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            total_loss += loss_fn(outputs, labels).item() * len(labels)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            count += len(labels)
    return {"loss": total_loss / count, "accuracy": correct / count}
