"""Datasets and data loaders."""

from pathlib import Path

import h5py
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from dental_3d_vae.ml.partial import complete_fingerprint, upsample


def stratified_split(
    metadata: pd.DataFrame, val_fraction: float = 0.15, test_fraction: float = 0.15, seed: int = 42
) -> pd.DataFrame:
    """Return ``metadata`` with a ``split`` column ("train"/"val"/"test").

    Each split keeps the overall mix of ``tooth_type``. With the defaults this
    reproduces the dissertation's split exactly.
    """
    train, rest = train_test_split(
        metadata, test_size=val_fraction + test_fraction,
        stratify=metadata["tooth_type"], random_state=seed,
    )
    val, test = train_test_split(
        rest, test_size=test_fraction / (val_fraction + test_fraction),
        stratify=rest["tooth_type"], random_state=seed,
    )
    split = pd.Series("train", index=metadata.index)
    split[val.index] = "val"
    split[test.index] = "test"
    return metadata.assign(split=split)


class CompleteToothDataset(Dataset):
    """Complete tooth point clouds for the rows of ``metadata``.

    Items are ``{"complete": (num_points, 3) float32, "sample_id", "tooth_id"}``.
    The HDF5 file opens on first access, so each DataLoader worker gets its own
    handle.
    """

    def __init__(self, h5_path: Path, metadata: pd.DataFrame) -> None:
        self.h5_path = h5_path
        self.metadata = metadata.reset_index(drop=True)
        self.points = None

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict:
        if self.points is None:
            self.points = h5py.File(self.h5_path, "r")["points"]
        row = self.metadata.iloc[index]
        sample_id = int(row["sample_id"])
        return {
            "complete": torch.from_numpy(self.points[sample_id]),
            "sample_id": sample_id,
            "tooth_id": int(row["tooth_id"]),
        }


class PartialToothDataset(Dataset):
    """Partial teeth with their complete originals, for the rows of partial ``metadata``.

    Items: ``partial`` (num_points, 3): the observed points, then upsampled points;
    ``complete`` (num_points, 3); ``removed`` (num_points,) bool, True where
    ``complete`` lost a point; plus ``partial_id``, ``source_sample_id``,
    ``tooth_id``, ``method`` and ``missing_fraction``. Every item has the same
    shapes, so the default DataLoader batching works. HDF5 files open on first
    access, so each DataLoader worker gets its own handles.
    """

    def __init__(self, complete_h5: Path, masks_h5: Path, metadata: pd.DataFrame) -> None:
        with h5py.File(complete_h5, "r") as complete, h5py.File(masks_h5, "r") as masks:
            if masks.attrs["complete_fingerprint"] != complete_fingerprint(complete["points"]):
                raise ValueError(f"{masks_h5} was built from a different complete dataset than {complete_h5}.")
        self.complete_h5, self.masks_h5 = complete_h5, masks_h5
        self.metadata = metadata.reset_index(drop=True)
        self.complete = self.removed = None

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict:
        if self.complete is None:
            self.complete = h5py.File(self.complete_h5, "r")["points"]
            self.removed = h5py.File(self.masks_h5, "r")["removed"]
        row = self.metadata.iloc[index]
        partial_id = int(row["partial_id"])
        complete = self.complete[int(row["source_sample_id"])]
        removed = self.removed[partial_id]
        return {
            "partial": torch.from_numpy(upsample(complete[~removed], len(complete), seed=partial_id)),
            "complete": torch.from_numpy(complete),
            "removed": torch.from_numpy(removed),
            "partial_id": partial_id,
            "source_sample_id": int(row["source_sample_id"]),
            "tooth_id": int(row["tooth_id"]),
            "method": row["method"],
            "missing_fraction": float(row["missing_fraction"]),
        }


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
