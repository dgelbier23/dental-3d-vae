"""Datasets and data loaders."""

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split


def make_synthetic_classification(
    n_samples: int, input_dim: int, num_classes: int, seed: int
) -> TensorDataset:
    """Random features labelled by a hidden linear "teacher", so labels are learnable.

    Uses its own generator, so the data depends only on ``seed``.
    """
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(n_samples, input_dim, generator=generator)
    teacher_weights = torch.randn(input_dim, num_classes, generator=generator)
    labels = (features @ teacher_weights).argmax(dim=1)
    return TensorDataset(features, labels)


def make_dataloaders(
    dataset: Dataset, val_fraction: float, batch_size: int, seed: int
) -> tuple[DataLoader, DataLoader]:
    """Seeded train/validation split; training batches are shuffled reproducibly."""
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [1 - val_fraction, val_fraction], generator=generator)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    return train_loader, val_loader
