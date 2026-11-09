import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from dental_3d_vae.ml.data import (
    CompleteToothDataset,
    PartialToothDataset,
    make_dataloaders,
    make_synthetic_classification,
    stratified_split,
)
from dental_3d_vae.ml.partial import build_partial_dataset


# --- stratified_split / CompleteToothDataset --------------------------------------


def _metadata(n_per_type: int = 20) -> pd.DataFrame:
    tooth_types = ["incisor", "canine", "molar"] * n_per_type
    return pd.DataFrame({
        "sample_id": range(len(tooth_types)),
        "tooth_id": [11, 13, 16] * n_per_type,
        "tooth_type": tooth_types,
    })


def test_stratified_split_sizes_and_balance() -> None:
    split = stratified_split(_metadata())

    assert split["split"].value_counts().to_dict() == {"train": 42, "val": 9, "test": 9}
    per_type = split.groupby("tooth_type")["split"].value_counts().unstack()
    assert (per_type["train"] == 14).all() and (per_type["test"] == 3).all()


def test_stratified_split_is_reproducible() -> None:
    def test_ids(seed: int) -> list[int]:
        split = stratified_split(_metadata(), seed=seed)
        return split.loc[split["split"] == "test", "sample_id"].tolist()

    assert test_ids(1) == test_ids(1)
    assert test_ids(1) != test_ids(2)


def test_complete_tooth_dataset_reads_rows_by_sample_id(tmp_path) -> None:
    points = np.random.default_rng(0).random((5, 8, 3), dtype=np.float32)
    with h5py.File(tmp_path / "teeth.h5", "w") as f:
        f.create_dataset("points", data=points)
    subset = _metadata(2).iloc[[4, 1]]  # out of order, like a split

    dataset = CompleteToothDataset(tmp_path / "teeth.h5", subset)

    assert len(dataset) == 2
    item = dataset[0]
    assert item["sample_id"] == 4 and item["tooth_id"] == 13
    assert torch.equal(item["complete"], torch.from_numpy(points[4]))


def _complete_and_partials(tmp_path, seed: int = 0):
    complete = tmp_path / f"complete{seed}"
    complete.mkdir()
    points = np.random.default_rng(seed).random((6, 32, 3), dtype=np.float32)
    with h5py.File(complete / "teeth_dataset.h5", "w") as f:
        f.create_dataset("points", data=points)
    _metadata(2).to_csv(complete / "metadata.csv", index=False)
    partials = build_partial_dataset(complete, tmp_path / f"partial{seed}")
    return complete / "teeth_dataset.h5", tmp_path / f"partial{seed}" / "partial_masks.h5", partials


def test_partial_tooth_dataset_items_and_batching(tmp_path) -> None:
    complete_h5, masks_h5, partials = _complete_and_partials(tmp_path)
    subset = partials[partials["missing_fraction"] == 0.45]

    dataset = PartialToothDataset(complete_h5, masks_h5, subset)
    item = dataset[1]

    observed = item["complete"][~item["removed"]]
    assert item["removed"].sum() == 14  # int(32 x 0.45)
    assert torch.equal(item["partial"][: len(observed)], observed)  # observed points first
    assert item["partial"].shape == item["complete"].shape == (32, 3)
    batch = next(iter(DataLoader(dataset, batch_size=4)))  # default collate: fixed shapes
    assert batch["partial"].shape == (4, 32, 3) and batch["removed"].shape == (4, 32)


def test_partial_tooth_dataset_rejects_masks_from_another_complete_dataset(tmp_path) -> None:
    complete_h5, _, partials = _complete_and_partials(tmp_path, seed=0)
    _, other_masks_h5, _ = _complete_and_partials(tmp_path, seed=1)

    with pytest.raises(ValueError, match="different complete dataset"):
        PartialToothDataset(complete_h5, other_masks_h5, partials)


def _dataset(seed: int = 0, n_samples: int = 200, num_classes: int = 3):
    return make_synthetic_classification(
        n_samples=n_samples, input_dim=8, num_classes=num_classes, seed=seed
    )


def _loaders(seed: int = 0, batch_size: int = 16):
    return make_dataloaders(_dataset(), val_fraction=0.25, batch_size=batch_size, seed=seed)


# --- make_synthetic_classification ----------------------------------------------


def test_synthetic_shapes_and_dtypes() -> None:
    features, labels = _dataset(n_samples=50).tensors

    assert features.shape == (50, 8)
    assert labels.shape == (50,)
    assert features.dtype == torch.float32
    assert labels.dtype == torch.int64  # required by nn.CrossEntropyLoss


def test_synthetic_labels_cover_all_classes() -> None:
    _, labels = _dataset(n_samples=1000, num_classes=3).tensors

    assert set(labels.tolist()) == {0, 1, 2}


def test_synthetic_depends_only_on_seed() -> None:
    torch.manual_seed(0)  # Global RNG state must not matter.
    a = _dataset(seed=1).tensors[0]
    torch.manual_seed(999)
    b = _dataset(seed=1).tensors[0]
    c = _dataset(seed=2).tensors[0]

    assert torch.equal(a, b)
    assert not torch.equal(a, c)


# --- make_dataloaders -----------------------------------------------------------


def test_split_sizes_and_no_overlap() -> None:
    train_loader, val_loader = _loaders()
    train_idx = set(train_loader.dataset.indices)
    val_idx = set(val_loader.dataset.indices)

    assert (len(train_idx), len(val_idx)) == (150, 50)
    assert train_idx.isdisjoint(val_idx)


def test_split_is_reproducible() -> None:
    def val_indices(seed: int) -> list[int]:
        return list(_loaders(seed)[1].dataset.indices)

    assert val_indices(3) == val_indices(3)
    assert val_indices(3) != val_indices(4)


def test_train_shuffle_is_reproducible_and_changes_each_epoch() -> None:
    def first_two_epochs(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        train_loader, _ = _loaders(seed, batch_size=150)  # One batch per epoch.
        return next(iter(train_loader))[1], next(iter(train_loader))[1]

    run_a = first_two_epochs(5)
    run_b = first_two_epochs(5)

    assert all(torch.equal(a, b) for a, b in zip(run_a, run_b))
    assert not torch.equal(run_a[0], run_a[1])


def test_val_loader_is_not_shuffled() -> None:
    _, val_loader = _loaders()

    first = torch.cat([features for features, _ in val_loader])
    second = torch.cat([features for features, _ in val_loader])

    assert torch.equal(first, second)
