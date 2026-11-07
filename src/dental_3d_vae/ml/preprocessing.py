"""Teeth3DS+ jaw scans -> normalised point clouds of individual teeth."""

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import trimesh

# FDI notation: tens digit = quadrant, units digit = position from the midline.
_QUADRANTS = {1: "Upper right", 2: "Upper left", 3: "Lower left", 4: "Lower right"}
_POSITIONS = [
    "central incisor", "lateral incisor", "canine", "first premolar",
    "second premolar", "first molar", "second molar", "third molar",
]
FDI_TOOTH_NAMES = {
    10 * quadrant + position: f"{side} {name}"
    for quadrant, side in _QUADRANTS.items()
    for position, name in enumerate(_POSITIONS, start=1)
}


def extract_teeth(
    jaw: trimesh.Trimesh, labels: np.ndarray, min_vertices: int = 50
) -> dict[int, trimesh.Trimesh]:
    """Split a labelled jaw scan into one mesh per tooth, keyed by FDI number.

    ``labels`` gives one label per jaw vertex (0 = gum). A tooth keeps only the
    faces whose three vertices all carry its label, then only its largest
    connected piece. Teeth with fewer than ``min_vertices`` vertices are skipped.
    """
    teeth = {}
    for tooth_id in np.unique(labels[labels != 0]):
        in_tooth = labels == tooth_id
        faces = jaw.faces[in_tooth[jaw.faces].all(axis=1)]
        if in_tooth.sum() < min_vertices or len(faces) == 0:
            continue
        # Renumber vertices: the k-th tooth vertex in the jaw becomes vertex k.
        new_index = np.cumsum(in_tooth) - 1
        tooth = trimesh.Trimesh(jaw.vertices[in_tooth], new_index[faces])
        teeth[int(tooth_id)] = max(
            tooth.split(only_watertight=False), key=lambda piece: len(piece.vertices)
        )
    return teeth


def farthest_point_sampling(points: np.ndarray, num_points: int) -> np.ndarray:
    """Pick ``num_points`` evenly spread points, greedily, starting from ``points[0]``.

    Each step adds the point farthest from everything picked so far.
    """
    # Separate contiguous x/y/z arrays make each step ~6x faster than (N, 3) maths.
    x, y, z = points.T.copy()
    chosen = [0]
    sq_dist = (x - x[0]) ** 2 + (y - y[0]) ** 2 + (z - z[0]) ** 2
    for _ in range(num_points - 1):
        i = int(sq_dist.argmax())
        chosen.append(i)
        np.minimum(sq_dist, (x - x[i]) ** 2 + (y - y[i]) ** 2 + (z - z[i]) ** 2, out=sq_dist)
    return points[chosen]


def normalise(points: np.ndarray) -> np.ndarray:
    """Centre on the centroid and scale so the farthest point is at distance 1."""
    centred = points - points.mean(axis=0)
    return centred / np.linalg.norm(centred, axis=1).max()


def tooth_to_pointcloud(
    tooth: trimesh.Trimesh, num_points: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample 4x ``num_points`` points uniformly over the surface, thin them with
    farthest point sampling, then normalise. Returns float32 (num_points, 3)."""
    surface, _ = trimesh.sample.sample_surface(tooth, 4 * num_points, seed=rng)
    return normalise(farthest_point_sampling(surface, num_points)).astype(np.float32)


def build_complete_dataset(
    raw_dir: Path, out_dir: Path, num_points: int = 2048, seed: int = 42
) -> pd.DataFrame:
    """Turn every jaw scan under ``raw_dir`` into tooth point clouds and save them.

    Each scan is an ``.obj`` with a same-named Teeth3DS+ ``.json`` label file
    beside it. Writes to ``out_dir``:

    - ``teeth_dataset.h5``: ``points``, float32 (N, num_points, 3)
    - ``metadata.csv``: row i describes ``points[i]``
    - ``dataset_info.json``: how the dataset was made

    Scans are processed in name order and teeth in FDI order, so the output
    depends only on the input files and ``seed``. Returns the metadata.
    """
    rng = np.random.default_rng(seed)
    clouds, rows = [], []
    scan_paths = sorted(Path(raw_dir).rglob("*.obj"), key=lambda path: path.stem)
    for i, obj_path in enumerate(scan_paths, start=1):
        scan = json.loads(obj_path.with_suffix(".json").read_text(encoding="utf-8"))
        # process=False keeps the file's vertex order, which the labels refer to.
        jaw = trimesh.load_mesh(obj_path, process=False)
        for tooth_id, tooth in extract_teeth(jaw, np.asarray(scan["labels"])).items():
            clouds.append(tooth_to_pointcloud(tooth, num_points, rng))
            bbox_x, bbox_y, bbox_z = tooth.extents
            rows.append({
                "sample_id": len(rows),
                "patient_id": scan["id_patient"],
                "scan_id": obj_path.stem,
                "jaw": scan["jaw"],
                "tooth_id": tooth_id,
                "tooth_type": FDI_TOOTH_NAMES[tooth_id],
                "num_vertices": len(tooth.vertices),
                "num_faces": len(tooth.faces),
                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "bbox_z": bbox_z,
            })
        print(f"[{i}/{len(scan_paths)}] {obj_path.stem}: {len(rows)} teeth so far")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_dir / "teeth_dataset.h5", "w") as f:
        f.create_dataset("points", data=np.stack(clouds))
    metadata = pd.DataFrame(rows)
    metadata.to_csv(out_dir / "metadata.csv", index=False)
    info = {
        "num_samples": len(rows),
        "num_points": num_points,
        "sampling": f"uniform surface ({4 * num_points} points) + farthest point",
        "normalisation": "zero mean, unit sphere",
        "seed": seed,
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return metadata
