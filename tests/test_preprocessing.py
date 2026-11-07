import json

import h5py
import numpy as np
import pandas as pd
import trimesh

from dental_3d_vae.ml.preprocessing import (
    FDI_TOOTH_NAMES,
    build_complete_dataset,
    extract_teeth,
    farthest_point_sampling,
    normalise,
    tooth_to_pointcloud,
)


def _fake_jaw() -> tuple[trimesh.Trimesh, np.ndarray]:
    """Two sphere 'teeth' (11, 21), a stray fragment labelled 11, a tooth (31)
    too small to keep, and gum (0). Returns the jaw and its per-vertex labels."""
    parts = [  # (mesh, label)
        (trimesh.creation.icosphere(subdivisions=2), 11),
        (trimesh.creation.icosphere(subdivisions=2).apply_translation([5, 0, 0]), 21),
        (trimesh.creation.box().apply_translation([0, 10, 0]), 11),  # fragment
        (trimesh.creation.box().apply_translation([0, -10, 0]), 31),  # 8 vertices
        (trimesh.creation.box(extents=[20, 1, 20]).apply_translation([0, 0, -5]), 0),
    ]
    jaw = trimesh.util.concatenate([mesh for mesh, _ in parts])
    labels = np.concatenate([np.full(len(mesh.vertices), label) for mesh, label in parts])
    return jaw, labels


def test_fdi_names() -> None:
    assert len(FDI_TOOTH_NAMES) == 32
    assert FDI_TOOTH_NAMES[11] == "Upper right central incisor"
    assert FDI_TOOTH_NAMES[48] == "Lower right third molar"


def test_extract_teeth_keeps_each_tooth_without_fragments_or_gum() -> None:
    teeth = extract_teeth(*_fake_jaw())

    assert sorted(teeth) == [11, 21]  # 31 is below min_vertices; 0 is gum
    sphere = trimesh.creation.icosphere(subdivisions=2)
    assert len(teeth[11].vertices) == len(sphere.vertices)  # fragment dropped
    np.testing.assert_allclose(teeth[21].centroid, [5, 0, 0], atol=1e-6)


def test_farthest_point_sampling_spreads_points_out() -> None:
    line = np.linspace([0, 0, 0], [1, 0, 0], 101)

    picked = farthest_point_sampling(line, 3)

    np.testing.assert_allclose(picked[:, 0], [0, 1, 0.5])


def test_normalise_centres_and_scales_to_unit_sphere() -> None:
    points = normalise(np.random.default_rng(0).normal(5, 3, size=(500, 3)))

    np.testing.assert_allclose(points.mean(axis=0), 0, atol=1e-12)
    assert np.isclose(np.linalg.norm(points, axis=1).max(), 1)


def test_tooth_to_pointcloud_is_seeded() -> None:
    tooth = trimesh.creation.icosphere()

    def sample(seed: int) -> np.ndarray:
        return tooth_to_pointcloud(tooth, 64, np.random.default_rng(seed))

    assert sample(0).shape == (64, 3) and sample(0).dtype == np.float32
    np.testing.assert_array_equal(sample(0), sample(0))
    assert not np.array_equal(sample(0), sample(1))


def test_build_complete_dataset_writes_aligned_points_and_metadata(tmp_path) -> None:
    jaw, labels = _fake_jaw()
    (tmp_path / "raw").mkdir()
    jaw.export(tmp_path / "raw" / "P1_upper.obj")
    (tmp_path / "raw" / "P1_upper.json").write_text(
        json.dumps({"id_patient": "P1", "jaw": "upper", "labels": labels.tolist()})
    )

    def build(name: str) -> pd.DataFrame:
        return build_complete_dataset(tmp_path / "raw", tmp_path / name, num_points=32)

    metadata = build("a")
    build("b")

    assert metadata["tooth_id"].tolist() == [11, 21]
    assert metadata["sample_id"].tolist() == [0, 1]
    assert metadata["tooth_type"].tolist() == [FDI_TOOTH_NAMES[11], FDI_TOOTH_NAMES[21]]
    with h5py.File(tmp_path / "a" / "teeth_dataset.h5") as a, h5py.File(tmp_path / "b" / "teeth_dataset.h5") as b:
        assert a["points"].shape == (2, 32, 3)
        np.testing.assert_array_equal(a["points"][:], b["points"][:])  # reproducible
    assert pd.read_csv(tmp_path / "a" / "metadata.csv").equals(metadata)
