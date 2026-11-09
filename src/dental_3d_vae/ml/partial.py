"""Synthetic partial teeth: remove a region of each complete point cloud, store which points went."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import KDTree


def plane_cut(points: np.ndarray, missing_fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Remove everything beyond a randomly oriented plane.

    Returns a boolean mask, True for the ``int(N * missing_fraction)`` points that
    lie farthest along a random direction. A normal draw per axis makes every
    direction equally likely, so no separate choice of side is needed.
    """
    num_remove = int(len(points) * missing_fraction)
    direction = rng.normal(size=3)
    removed = np.zeros(len(points), dtype=bool)
    removed[np.argsort(points @ direction)[len(points) - num_remove:]] = True
    return removed


def connected_patch(points: np.ndarray, missing_fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Remove one connected patch of surface around a random point.

    Returns a boolean mask, True for the random seed point and its nearest
    neighbours, ``int(N * missing_fraction)`` points in all.
    """
    num_remove = int(len(points) * missing_fraction)
    seed_point = points[rng.integers(len(points))]
    removed = np.zeros(len(points), dtype=bool)
    removed[np.argsort(((points - seed_point) ** 2).sum(axis=1))[:num_remove]] = True
    return removed


REMOVAL_METHODS = {"plane": plane_cut, "connected_patch": connected_patch}


def upsample(points: np.ndarray, num_points: int, seed: int) -> np.ndarray:
    """Pad a partial cloud to ``num_points`` using only its own geometry.

    The observed points come first, unchanged. Each added point lies a random
    25-75% of the way from a random observed point to that point's nearest
    observed neighbour, so nothing is taken from the missing region.
    """
    rng = np.random.default_rng(seed)
    num_new = num_points - len(points)
    nearest = KDTree(points).query(points, k=2)[1][:, 1]  # Column 0 is the point itself.
    source = rng.integers(0, len(points), size=num_new)
    alpha = rng.uniform(0.25, 0.75, size=(num_new, 1)).astype(np.float32)
    new_points = (1 - alpha) * points[source] + alpha * points[nearest[source]]
    return np.concatenate([points, new_points]).astype(np.float32)


def complete_fingerprint(complete_points: h5py.Dataset) -> str:
    """Short hash identifying a complete dataset; masks only make sense with their own."""
    return hashlib.sha256(complete_points[0].tobytes() + str(complete_points.shape).encode()).hexdigest()[:16]


def build_partial_dataset(
    complete_dir: Path,
    out_dir: Path,
    methods: tuple[str, ...] = ("plane", "connected_patch"),
    missing_fractions: tuple[float, ...] = (0.0, 0.15, 0.30, 0.45),
    seed: int = 42,
) -> pd.DataFrame:
    """Make one partial of every complete tooth per method and missing fraction.

    Reads ``teeth_dataset.h5`` and ``metadata.csv`` from ``complete_dir``. Writes to
    ``out_dir``:

    - ``partial_masks.h5``: ``removed``, bool (P, num_points); row i is True where
      partial i lost a point of its complete tooth
    - ``partial_metadata.csv``: row i describes partial i
    - ``partial_dataset_info.json``: how the partials were made

    Each partial is seeded from (``seed``, tooth, method, fraction), so it never
    depends on which other partials are built. Returns the metadata.
    """
    complete_dir, out_dir = Path(complete_dir), Path(out_dir)
    teeth = pd.read_csv(complete_dir / "metadata.csv")
    with h5py.File(complete_dir / "teeth_dataset.h5", "r") as f:
        complete, fingerprint = f["points"][:], complete_fingerprint(f["points"])

    masks, rows = [], []
    for tooth, points in zip(teeth.itertuples(), complete):
        for method in methods:
            for fraction in missing_fractions:
                key = [seed, tooth.sample_id, list(REMOVAL_METHODS).index(method), round(fraction * 1000)]
                masks.append(REMOVAL_METHODS[method](points, fraction, np.random.default_rng(key)))
                rows.append({
                    "partial_id": len(rows),
                    "source_sample_id": tooth.sample_id,
                    "tooth_id": tooth.tooth_id,
                    "tooth_type": tooth.tooth_type,
                    "method": method,
                    "missing_fraction": fraction,
                    "num_removed": int(masks[-1].sum()),
                })

    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_dir / "partial_masks.h5", "w") as f:
        f.create_dataset("removed", data=np.stack(masks), compression="gzip")
        f.attrs["complete_fingerprint"] = fingerprint
    metadata = pd.DataFrame(rows)
    metadata.to_csv(out_dir / "partial_metadata.csv", index=False)
    info = {
        "num_partials": len(rows),
        "complete_dataset": complete_dir.name,
        "complete_fingerprint": fingerprint,
        "methods": list(methods),
        "missing_fractions": list(missing_fractions),
        "seed": seed,
    }
    (out_dir / "partial_dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return metadata
