"""Interactive 3D views of jaw scans and tooth point clouds, in the LifeHistories colours."""

import io
import warnings
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import trimesh
from PIL import Image
from plotly.subplots import make_subplots
from scipy.spatial import KDTree

from dental_3d_vae.viz.colours import LIFE_HISTORIES
from dental_3d_vae.viz.plotting import SEQUENTIAL

GUM_COLOUR = LIFE_HISTORIES[3]  # light sand
TOOTH_COLOURS = LIFE_HISTORIES[:3] + LIFE_HISTORIES[4:]  # the other six, cycled
# The palette's most distinct pair (largest CIEDE2000 difference, 2.8:1 lightness contrast):
# dark brown makes the removed region stand out against white; teal recedes.
KEPT_COLOUR = LIFE_HISTORIES[5]  # teal
REMOVED_COLOUR = LIFE_HISTORIES[0]  # dark brown
PLANE_COLOUR = LIFE_HISTORIES[3]  # light sand, drawn translucent
# Completion views: warm input against the cool reconstruction, with teal for the missing region;
# all three pairs pass a colour-vision check.
INPUT_COLOUR = LIFE_HISTORIES[1]  # terracotta
RECONSTRUCTION_COLOUR = LIFE_HISTORIES[6]  # dark blue
MISSING_COLOUR = LIFE_HISTORIES[5]  # teal


def _hide_axes(fig: go.Figure) -> None:
    """True proportions (aspectmode="data") with no axes, for every 3D panel."""
    hidden = dict(visible=False)
    fig.update_scenes(aspectmode="data", xaxis=hidden, yaxis=hidden, zaxis=hidden)


def plot_jaw(jaw: trimesh.Trimesh, labels: np.ndarray) -> go.Figure:
    """Jaw scan with each tooth in its own colour; ``labels`` has one FDI label per
    vertex (0 = gum). Neighbouring teeth always get different colours."""
    teeth = np.unique(labels[labels != 0])
    colour_of = {0: GUM_COLOUR} | {
        tooth: TOOTH_COLOURS[i % len(TOOTH_COLOURS)] for i, tooth in enumerate(teeth)
    }
    (x, y, z), (i, j, k) = jaw.vertices.T, jaw.faces.T
    fig = go.Figure(go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        vertexcolor=[colour_of[label] for label in labels], hoverinfo="skip",
    ))
    _hide_axes(fig)
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
    return fig


def _panel_grid(titles: list[str], cols: int) -> tuple[go.Figure, int]:
    """Empty grid of 3D panels, one per title; returns the figure and its column count."""
    cols = min(cols, len(titles))
    rows = -(-len(titles) // cols)  # ceiling division
    fig = make_subplots(
        rows=rows, cols=cols, specs=[[{"type": "scene"}] * cols] * rows,
        subplot_titles=titles, horizontal_spacing=0.01, vertical_spacing=0.04,
    )
    _hide_axes(fig)
    fig.update_layout(height=280 * rows, margin=dict(l=0, r=0, t=30, b=0))
    return fig, cols


def plot_pointclouds(clouds: list, titles: list[str], cols: int = 4) -> go.Figure:
    """One rotatable 3D panel per (N, 3) point cloud, shaded dark to light by height."""
    fig, cols = _panel_grid(titles, cols)
    for n, cloud in enumerate(clouds):
        x, y, z = np.asarray(cloud).T
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z, mode="markers", hoverinfo="skip", showlegend=False,
                marker=dict(size=1.5, color=z, colorscale=list(LIFE_HISTORIES)),
            ),
            row=n // cols + 1, col=n % cols + 1,
        )
    return fig


def _plane_square(cloud: np.ndarray, normal: np.ndarray, offset: float) -> go.Mesh3d:
    """Square patch of the plane ``points @ normal == offset``, centred on the cloud
    and as wide as the cloud."""
    centroid = cloud.mean(axis=0)
    centre = centroid - (centroid @ normal - offset) * normal  # Nearest plane point.
    helper = np.eye(3)[np.argmin(np.abs(normal))]  # Any axis not parallel to the normal.
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    half = np.linalg.norm(cloud - centroid, axis=1).max()
    x, y, z = np.array([centre + half * (su * u + sv * v) for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]).T
    return go.Mesh3d(x=x, y=y, z=z, i=[0, 0], j=[1, 2], k=[2, 3], color=PLANE_COLOUR,
                     opacity=0.5, hoverinfo="skip", name="Cutting plane")


def plot_partials(
    clouds: list, removed_masks: list, titles: list[str], planes: list | None = None, cols: int = 3
) -> go.Figure:
    """One 3D panel per complete (N, 3) cloud: kept points in teal, removed in dark brown.

    ``planes`` optionally gives each panel's cutting plane as ``(normal, offset)``,
    or None for no plane, and draws it as a translucent square.
    """
    fig, cols = _panel_grid(titles, cols)
    planes = planes or [None] * len(clouds)
    for n, (cloud, removed, plane) in enumerate(zip(clouds, removed_masks, planes)):
        cloud, removed = np.asarray(cloud), np.asarray(removed)
        traces = [
            go.Scatter3d(
                x=cloud[mask, 0], y=cloud[mask, 1], z=cloud[mask, 2], mode="markers",
                hoverinfo="skip", name=name, marker=dict(size=1.5, color=colour),
            )
            for name, mask, colour in (("Kept", ~removed, KEPT_COLOUR), ("Removed", removed, REMOVED_COLOUR))
        ]
        if plane is not None:
            traces.append(_plane_square(cloud, *plane))
        for trace in traces:
            first = not any(t.name == trace.name for t in fig.data)
            trace.update(legendgroup=trace.name, showlegend=first)  # One legend entry per kind.
            fig.add_trace(trace, row=n // cols + 1, col=n % cols + 1)
    _same_box(fig, clouds)
    return fig


def _same_box(fig: go.Figure, clouds: list) -> None:
    """The same box around every (centred) cloud: equal scale in every panel, and planes
    are clipped to the box instead of stretching it. Legend dots readable, not 1.5 px."""
    reach = max(float(np.abs(np.asarray(cloud)).max()) for cloud in clouds)
    box = dict(range=[-reach, reach])
    fig.update_scenes(xaxis=box, yaxis=box, zaxis=box, aspectmode="cube")
    fig.update_layout(legend=dict(itemsizing="constant"))


def plot_completion(complete: np.ndarray, removed: np.ndarray, reconstruction: np.ndarray,
                    ground_truth: bool = True) -> go.Figure:
    """One partial tooth's completion, left to right: the model's input (the observed
    points), the reconstruction over it, the reconstruction alone and, if ``ground_truth``,
    the complete tooth with its missing region."""
    complete, removed, reconstruction = np.asarray(complete), np.asarray(removed), np.asarray(reconstruction)
    observed = ("Observed points", complete[~removed], INPUT_COLOUR, 1.5)
    rebuilt = ("Reconstruction", reconstruction, RECONSTRUCTION_COLOUR, 1.5)
    panels = {
        "Partial input": [observed],
        "Reconstruction over the input": [observed, (*rebuilt[:3], 1.2)],
        "Reconstruction": [rebuilt],
    }
    if ground_truth:
        panels["Ground truth"] = [observed, ("Missing region", complete[removed], MISSING_COLOUR, 1.5)]
    fig, _ = _panel_grid(list(panels), cols=len(panels))
    for col, traces in enumerate(panels.values(), start=1):
        for name, cloud, colour, size in traces:
            first = not any(trace.name == name for trace in fig.data)
            fig.add_trace(go.Scatter3d(x=cloud[:, 0], y=cloud[:, 1], z=cloud[:, 2], mode="markers", name=name,
                                       legendgroup=name, showlegend=first, hoverinfo="skip",
                                       marker=dict(size=size, color=colour)), row=1, col=col)
    _same_box(fig, [complete, reconstruction])
    fig.update_layout(height=360, legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.02, yanchor="top"))
    return fig


def plot_point_errors(complete: np.ndarray, reconstruction: np.ndarray, removed: np.ndarray | None = None) -> go.Figure:
    """The ground-truth tooth, each point coloured by its distance to the nearest
    reconstructed point; removed points (if ``removed`` is given) drawn as larger diamonds.

    The colour scale stops at the 95th percentile, so a few outliers cannot wash it out.
    """
    complete, reconstruction = np.asarray(complete), np.asarray(reconstruction)
    removed = np.zeros(len(complete), dtype=bool) if removed is None else np.asarray(removed)
    distance = KDTree(reconstruction).query(complete)[0]
    fig = go.Figure()
    for name, mask, symbol, size in (("Observed points", ~removed, "circle", 2.5),
                                     ("Missing region", removed, "diamond", 3.5)):
        if mask.any():
            fig.add_scatter3d(x=complete[mask, 0], y=complete[mask, 1], z=complete[mask, 2], mode="markers",
                              name=name, showlegend=False, customdata=distance[mask],
                              hovertemplate=f"{name}<br>distance %{{customdata:.4f}}<extra></extra>",
                              marker=dict(size=size, symbol=symbol, color=distance[mask], coloraxis="coloraxis"))
    summary = ", ".join(f"{name} {distance[mask].mean():.4f}" for name, mask in
                        (("observed", ~removed), ("missing region (diamonds)", removed)) if mask.any())
    _hide_axes(fig)
    fig.update_layout(
        title=dict(text="Where the reconstruction misses", subtitle_text=(
            f"Mean distance: {summary} · scale capped at the 95th percentile")),
        coloraxis=dict(colorscale=SEQUENTIAL, cmin=0, cmax=float(np.percentile(distance, 95)),
                       colorbar=dict(title=dict(text="Distance", side="right"), thickness=12)),
        height=520, margin=dict(l=0, r=0, t=90, b=10))
    return fig


def save_rotating_gif(
    fig: go.Figure, path: Path, frames: int = 30, seconds: float = 6, width: int = 900, height: int = 500
) -> None:
    """Save a looping GIF in which every 3D panel of ``fig`` turns one full circle.

    Needs ``kaleido`` and Chrome (local use only); takes about 1.3 s per frame.
    """
    import kaleido  # Only needed here, so the module still imports without it (e.g. Colab).

    fig = go.Figure(fig)  # Rotate a copy, not the caller's figure.
    images = []
    kaleido.start_sync_server(silence_warnings=True)  # Keeps Chrome open between frames.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Harmless kaleido notices about its server mode.
            for angle in np.linspace(0, 2 * np.pi, frames, endpoint=False):
                fig.update_scenes(camera_eye=dict(x=1.5 * np.cos(angle), y=1.5 * np.sin(angle), z=1.0))
                png = fig.to_image(format="png", width=width, height=height)
                images.append(Image.open(io.BytesIO(png)))
    finally:
        kaleido.stop_sync_server(silence_warnings=True)
    images[0].save(path, save_all=True, append_images=images[1:], loop=0,
                   duration=round(1000 * seconds / frames), optimize=True)
