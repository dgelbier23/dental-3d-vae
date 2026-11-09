from dataclasses import replace

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from dental_3d_vae.experiments.config import ExperimentConfig
from dental_3d_vae.ml.data import (
    CompleteToothDataset,
    PartialToothDataset,
    make_loaders,
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


def _complete_and_partials(tmp_path, seed: int = 0, n_per_type: int = 2):
    complete = tmp_path / f"complete{seed}"
    complete.mkdir()
    points = np.random.default_rng(seed).random((3 * n_per_type, 32, 3), dtype=np.float32)
    with h5py.File(complete / "teeth_dataset.h5", "w") as f:
        f.create_dataset("points", data=points)
    _metadata(n_per_type).to_csv(complete / "metadata.csv", index=False)
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


# --- make_loaders ------------------------------------------------------------------


def test_make_loaders_complete_run_uses_the_split(tmp_path) -> None:
    complete_h5, _, _ = _complete_and_partials(tmp_path, n_per_type=20)
    config = ExperimentConfig(reconstruction_mode="complete", batch_size=8)

    loaders = make_loaders(config, complete_h5.parent, num_workers=0)

    split = stratified_split(_metadata(20))
    assert {name: len(loader.dataset) for name, loader in loaders.items()} == {"train": 42, "val": 9, "test": 9}
    test_ids = torch.cat([batch["sample_id"] for batch in loaders["test"]]).tolist()
    assert test_ids == split.loc[split["split"] == "test", "sample_id"].tolist()  # unshuffled, in split order


def test_make_loaders_partial_run_uses_one_method_and_inherits_the_split(tmp_path) -> None:
    complete_h5, masks_h5, _ = _complete_and_partials(tmp_path, n_per_type=20)
    config = ExperimentConfig(method="connected_patch", batch_size=100)

    loaders = make_loaders(config, complete_h5.parent, masks_h5.parent, num_workers=0)

    split = stratified_split(_metadata(20))
    test = next(iter(loaders["test"]))
    assert len(loaders["train"].dataset) == 42 * 4  # every missing fraction
    assert set(test["method"]) == {"connected_patch"}
    assert set(test["source_sample_id"].tolist()) == set(split.loc[split["split"] == "test", "sample_id"])


def test_make_loaders_shuffles_training_reproducibly(tmp_path) -> None:
    complete_h5, _, _ = _complete_and_partials(tmp_path, n_per_type=20)
    config = ExperimentConfig(reconstruction_mode="complete", batch_size=42)

    def first_two_epochs(seed: int) -> list[list[int]]:
        loader = make_loaders(replace(config, seed=seed), complete_h5.parent, num_workers=0)["train"]
        return [next(iter(loader))["sample_id"].tolist() for _ in range(2)]

    epochs = first_two_epochs(1)
    assert epochs == first_two_epochs(1)
    assert epochs[0] != epochs[1] and sorted(epochs[0]) == sorted(epochs[1])
