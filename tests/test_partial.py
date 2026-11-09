import json

import h5py
import numpy as np
import pandas as pd
import pytest

from dental_3d_vae.ml.partial import (
    build_partial_dataset,
    connected_patch,
    plane_cut,
    upsample,
)

POINTS = np.random.default_rng(0).normal(size=(200, 3)).astype(np.float32)


@pytest.mark.parametrize("fraction, expected", [(0.0, 0), (0.15, 30), (0.45, 90)])
@pytest.mark.parametrize("method", [plane_cut, connected_patch])
def test_removes_exactly_int_n_times_fraction(method, fraction: float, expected: int) -> None:
    assert method(POINTS, fraction, np.random.default_rng(1)).sum() == expected


def test_plane_cut_removes_one_side_of_a_plane() -> None:
    removed = plane_cut(POINTS, 0.3, np.random.default_rng(1))

    direction = np.random.default_rng(1).normal(size=3)  # The direction plane_cut drew.
    height = POINTS @ direction
    assert height[removed].min() >= height[~removed].max()


def test_connected_patch_removes_the_points_nearest_a_seed() -> None:
    removed = connected_patch(POINTS, 0.3, np.random.default_rng(1))

    seed_point = POINTS[np.random.default_rng(1).integers(len(POINTS))]  # The seed it drew.
    distance = np.linalg.norm(POINTS - seed_point, axis=1)
    assert removed[np.argmin(distance)]
    assert distance[removed].max() <= distance[~removed].min()


def test_upsample_keeps_observed_points_first_and_is_seeded() -> None:
    partial = POINTS[:120]

    out = upsample(partial, 200, seed=3)

    assert out.shape == (200, 3) and out.dtype == np.float32
    np.testing.assert_array_equal(out[:120], partial)
    np.testing.assert_array_equal(out, upsample(partial, 200, seed=3))
    assert not np.array_equal(out, upsample(partial, 200, seed=4))
    assert (out[120:] >= partial.min(axis=0)).all() and (out[120:] <= partial.max(axis=0)).all()


def _complete_dataset(directory, num_teeth: int = 3) -> None:
    directory.mkdir()
    points = np.random.default_rng(0).normal(size=(num_teeth, 64, 3)).astype(np.float32)
    with h5py.File(directory / "teeth_dataset.h5", "w") as f:
        f.create_dataset("points", data=points)
    pd.DataFrame({
        "sample_id": range(num_teeth), "tooth_id": [11, 21, 31][:num_teeth],
        "tooth_type": ["a", "b", "c"][:num_teeth],
    }).to_csv(directory / "metadata.csv", index=False)


def test_build_partial_dataset_layout(tmp_path) -> None:
    _complete_dataset(tmp_path / "complete")

    metadata = build_partial_dataset(tmp_path / "complete", tmp_path / "partial")

    assert len(metadata) == 3 * 2 * 4  # teeth x methods x fractions
    assert metadata["partial_id"].tolist() == list(range(24))
    first_tooth = metadata[metadata["source_sample_id"] == 0]
    assert first_tooth["method"].tolist() == ["plane"] * 4 + ["connected_patch"] * 4
    assert first_tooth["num_removed"].tolist() == [0, 9, 19, 28] * 2  # int(64 x fraction)
    with h5py.File(tmp_path / "partial" / "partial_masks.h5") as f:
        assert f["removed"].shape == (24, 64) and f["removed"].dtype == bool
        np.testing.assert_array_equal(f["removed"][:].sum(axis=1), metadata["num_removed"])
    info = json.loads((tmp_path / "partial" / "partial_dataset_info.json").read_text())
    assert info["complete_dataset"] == "complete" and info["num_partials"] == 24


def test_each_partial_is_independent_of_what_else_is_built(tmp_path) -> None:
    _complete_dataset(tmp_path / "complete")

    both = build_partial_dataset(tmp_path / "complete", tmp_path / "both")
    plane_only = build_partial_dataset(tmp_path / "complete", tmp_path / "plane", methods=("plane",))

    with h5py.File(tmp_path / "both" / "partial_masks.h5") as a, \
            h5py.File(tmp_path / "plane" / "partial_masks.h5") as b:
        np.testing.assert_array_equal(a["removed"][both["method"] == "plane"], b["removed"][:])
    assert len(plane_only) == 12
