
# dental-3d-vae

A development workspace for machine learning and deep learning.

## Environments

- VS Code: primary editor.
- Claude Code: AI-assisted development.
- Local Python: CPU development and testing.
- Google Colab: cloud GPU experiments.
- GitHub: version control and collaboration.

## Setup

From the repo folder in PowerShell (no need to activate the environment):

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m pytest -q
```

In VS Code, select the `.venv` kernel for notebooks.

## Data preparation

Turns intra-oral jaw scans into a dataset of individual teeth, each a normalised 2,048-point
cloud, then makes controlled incomplete versions of every tooth for learning to reconstruct
missing geometry.

<p align="center">
  <img src="docs/figures/jaw_segmentation.gif" width="49%" alt="A segmented jaw scan rotating, one colour per tooth">
  <img src="docs/figures/example_teeth.gif" width="49%" alt="Eight tooth types as rotating point clouds">
</p>

### Pipeline

1. **Jaw scan → teeth.** Each scan comes with one label per vertex (0 = gum, otherwise the tooth's
   [FDI number](https://en.wikipedia.org/wiki/FDI_World_Dental_Federation_notation)). Each tooth
   keeps only the faces whose three vertices carry its label, then only its largest connected piece.
2. **Tooth → point cloud.** 8,192 points are sampled uniformly over the tooth surface, then thinned
   to 2,048 evenly spread points by farthest point sampling.
3. **Normalise.** Centre on the centroid and scale so the farthest point is at distance 1.
4. **Save.** `teeth_dataset.h5` (`points`: N × 2048 × 3, float32), `metadata.csv` (one row per
   tooth: patient, jaw, FDI number, tooth type, mesh size, bounding box) and `dataset_info.json`.

Scans are processed in parallel on every CPU core, and each scan's sampling is seeded from its
name, so the output is identical whatever the number of workers. A scan that fails is skipped and
listed in `dataset_info.json`. On 300 scans (20 cores) the build takes under two minutes and gives
4,038 teeth.

### Partial teeth

Intra-oral scans are often incomplete (occlusion, patient movement, braces). To study how well a
model recovers missing geometry, every tooth is cut in two controlled ways, removing 0, 15, 30 or
45% of its points:

- **Plane cut:** everything beyond a randomly oriented plane, like a scan's field of view ending.
- **Connected patch:** a random point and its nearest neighbours, like a localised hole.

<p align="center">
  <img src="docs/figures/partial_teeth.gif" width="90%" alt="A first molar under plane cuts and connected-patch removal at 15, 30 and 45 percent missing, rotating; kept points teal, removed points dark brown, cutting planes in sand">
</p>

This gives 32,304 partials (4,038 teeth × 2 methods × 4 levels) in about four seconds. Only *which*
points each partial lost is stored (`partial_masks.h5`, one true/false per point, 9 MB), so every
partial and its missing region are rebuilt exactly from the complete tooth, and each plane cut's
plane can be recovered for display. Each partial is seeded from (seed, tooth, method, fraction),
so every model sees exactly the same incomplete teeth, and the masks carry a fingerprint of the
complete dataset they belong to, so they cannot be paired with the wrong one.

For training, the kept points are padded back to 2,048 by adding points between observed
neighbours, so nothing comes from the missing region. Every partial inherits its source tooth's
train/validation/test split, so no tooth is used for training in one form and testing in another.

### Dataset

[Teeth3DS+](https://osf.io/xctdy/) (Ben-Hamadou et al., 2022): clinically acquired intra-oral
scans with per-vertex tooth labels. This project uses `data_part_1`: 300 jaw scans (150 upper,
150 lower) from 300 patients. Third molars are rare in it (7–11 of each, against 115–150 for
every other position):

<p align="center">
  <img src="docs/figures/teeth_per_position.png" width="80%" alt="Bar chart of teeth per FDI position">
</p>

The data is distributed under its own licence, so it is **not included in this repository** and
must not be committed. `data/` is ignored by Git, and a test fails if a notebook is saved with
outputs (they would contain the data).

### Running it

1. Download `data_part_1` from the [Teeth3DS+ page](https://osf.io/xctdy/) and put its `upper/`
   and `lower/` folders in `data/raw/teeth3ds/`.
2. Open `notebooks/02_data_preparation.ipynb`, set `REBUILD = True` in section 1 and run all cells.
   The notebook shows a segmented jaw, its teeth as point clouds, an overview of the dataset,
   example teeth and the train/validation/test split, then builds and shows the partial teeth
   (automatically, if they do not exist yet).

The outputs go to the folders set in section 1, currently `data/processed/complete_rebuilt/` and
`data/processed/partial_rebuilt/`. With processed data but no raw scans, the notebook skips the
scan-to-teeth demo and runs the rest. Build locally rather than in Colab: Colab runtimes
usually have only two CPU cores.

From a Python script, guard the call (worker processes re-import the script on Windows):

```python
from pathlib import Path

from dental_3d_vae.ml.preprocessing import build_complete_dataset

if __name__ == "__main__":
    build_complete_dataset(Path("data/raw/teeth3ds"), Path("data/processed/complete"))
```

### Code

| File | Contents |
|---|---|
| `src/dental_3d_vae/ml/preprocessing.py` | `extract_teeth`, `farthest_point_sampling`, `normalise`, `build_complete_dataset` |
| `src/dental_3d_vae/ml/partial.py` | `plane_cut`, `connected_patch`, `upsample`, `cutting_plane`, `build_partial_dataset` |
| `src/dental_3d_vae/ml/data.py` | `stratified_split` (70/15/15 by tooth type); PyTorch datasets `CompleteToothDataset` and `PartialToothDataset` (lazy HDF5) |
| `src/dental_3d_vae/viz/pointclouds.py` | Interactive 3D jaw, point-cloud and partial-tooth views; `save_rotating_gif` for figures |
| `notebooks/02_data_preparation.ipynb` | The pipeline, step by step, with visualisations |

To regenerate the figures above, set `SAVE_FIGURES = True` in the notebook (local only; needs
`kaleido` and Chrome).

## Everyday workflow

1. Edit code in VS Code and run the tests.
2. Commit and push to GitHub.
3. In Colab, use Runtime > Restart session, then run all cells.
4. Results save to Google Drive. Download a run folder into `artifacts/runs/`
   to compare it locally with `notebooks/04_compare_runs.ipynb`.

## Notebooks

Notebooks run locally (`.venv` kernel) or in Google Colab, unchanged.

- The first code cell of every notebook is the same bootstrap cell. In Colab it
  clones the latest `main` from GitHub using the `GITHUB_TOKEN` Colab secret
  (fine-grained, read-only); locally it does nothing.
- In Colab, open notebooks via File > Open notebook > GitHub, pick a GPU runtime,
  and treat them as read-only: edit in VS Code, push, then re-run in Colab.
- Run outputs go to `get_runs_dir()`: `MyDrive/<repo name>/runs` in Colab (pass a
  different subfolder to change it), `artifacts/runs` locally (ignored by Git).

## Reproducibility

### PyTorch version parity

Local `torch` is pinned in `requirements.txt` to Colab's preinstalled version
(currently `2.11.0`; Colab ships it with CUDA 13.0). Local uses the CPU build.

- Never `pip install torch` in Colab; use the version Colab provides.
- When Colab's version changes (shown by `notebooks/01_gpu_check.ipynb`),
  update the pin and reinstall locally:

```powershell
pip install "torch==<version>" --index-url https://download.pytorch.org/whl/cpu
```

### Experiments

Record important dependency versions, random seeds,
dataset versions, and experiment configuration.

Never commit secrets or large generated model artifacts.
