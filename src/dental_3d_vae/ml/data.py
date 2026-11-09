"""Datasets and data loaders."""

from pathlib import Path

import h5py
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from dental_3d_vae.experiments.config import ExperimentConfig
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


def make_loaders(
    config: ExperimentConfig, complete_dir: Path, partial_dir: Path | None = None, num_workers: int = 2
) -> dict[str, DataLoader]:
    """Train, validation and test loaders for ``config``, keyed "train", "val", "test".

    Uses the dissertation's split (``stratified_split`` with ``config.seed``). Partial runs
    use every missing fraction of ``config.method``, each partial in its source tooth's
    split. Only training batches are shuffled, reproducibly from ``config.seed``.
    """
    complete = stratified_split(pd.read_csv(complete_dir / "metadata.csv"), seed=config.seed)
    complete_h5 = complete_dir / "teeth_dataset.h5"
    if config.reconstruction_mode == "partial":
        partials = pd.read_csv(partial_dir / "partial_metadata.csv")
        partials = partials[partials["method"] == config.method]
        split = partials["source_sample_id"].map(complete.set_index("sample_id")["split"])
        datasets = {name: PartialToothDataset(complete_h5, partial_dir / "partial_masks.h5", partials[split == name])
                    for name in ("train", "val", "test")}
    else:
        datasets = {name: CompleteToothDataset(complete_h5, complete[complete["split"] == name])
                    for name in ("train", "val", "test")}
    generator = torch.Generator().manual_seed(config.seed)
    return {
        name: DataLoader(dataset, batch_size=config.batch_size, shuffle=name == "train", generator=generator,
                         num_workers=num_workers, pin_memory=torch.cuda.is_available())
        for name, dataset in datasets.items()
    }
