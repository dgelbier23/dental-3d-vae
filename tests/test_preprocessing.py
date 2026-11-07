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


def _write_scan(raw_dir, patient: str, labels: list) -> None:
    raw_dir.mkdir(exist_ok=True)
    _fake_jaw()[0].export(raw_dir / f"{patient}_upper.obj")
    (raw_dir / f"{patient}_upper.json").write_text(
        json.dumps({"id_patient": patient, "jaw": "upper", "labels": labels})
    )


def test_build_complete_dataset_is_aligned_and_independent_of_workers(tmp_path) -> None:
    labels = _fake_jaw()[1].tolist()
    _write_scan(tmp_path / "raw", "P2", labels)
    _write_scan(tmp_path / "raw", "P1", labels)

    def build(name: str, workers: int) -> pd.DataFrame:
        return build_complete_dataset(tmp_path / "raw", tmp_path / name, num_points=32, workers=workers)

    metadata = build("serial", workers=1)
    build("parallel", workers=2)

    assert metadata["scan_id"].tolist() == ["P1_upper"] * 2 + ["P2_upper"] * 2  # name order
    assert metadata["tooth_id"].tolist() == [11, 21, 11, 21]
    assert metadata["sample_id"].tolist() == [0, 1, 2, 3]
    assert pd.read_csv(tmp_path / "serial" / "metadata.csv").equals(metadata)
    with h5py.File(tmp_path / "serial" / "teeth_dataset.h5") as serial, \
            h5py.File(tmp_path / "parallel" / "teeth_dataset.h5") as parallel:
        assert serial["points"].shape == (4, 32, 3)
        np.testing.assert_array_equal(serial["points"][:], parallel["points"][:])


def test_build_complete_dataset_skips_and_reports_failed_scans(tmp_path) -> None:
    _write_scan(tmp_path / "raw", "GOOD", _fake_jaw()[1].tolist())
    _write_scan(tmp_path / "raw", "BAD", [11, 21])  # Too few labels for the mesh.

    metadata = build_complete_dataset(tmp_path / "raw", tmp_path / "out", num_points=32)

    assert metadata["scan_id"].unique().tolist() == ["GOOD_upper"]
    info = json.loads((tmp_path / "out" / "dataset_info.json").read_text())
    assert list(info["failed_scans"]) == ["BAD_upper"]
