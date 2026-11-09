"""Interactive 3D views of jaw scans and tooth point clouds, in the LifeHistories colours."""

import io
import warnings
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import trimesh
from PIL import Image
from plotly.subplots import make_subplots

from dental_3d_vae.viz.colours import LIFE_HISTORIES

GUM_COLOUR = LIFE_HISTORIES[3]  # light sand
TOOTH_COLOURS = LIFE_HISTORIES[:3] + LIFE_HISTORIES[4:]  # the other six, cycled
KEPT_COLOUR = LIFE_HISTORIES[6]  # deep blue
REMOVED_COLOUR = LIFE_HISTORIES[1]  # terracotta


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


def plot_partials(clouds: list, removed_masks: list, titles: list[str], cols: int = 3) -> go.Figure:
    """One 3D panel per complete (N, 3) cloud: kept points in blue, removed in terracotta."""
    fig, cols = _panel_grid(titles, cols)
    for n, (cloud, removed) in enumerate(zip(clouds, removed_masks)):
        cloud, removed = np.asarray(cloud), np.asarray(removed)
        for name, mask, colour in (("Kept", ~removed, KEPT_COLOUR), ("Removed", removed, REMOVED_COLOUR)):
            x, y, z = cloud[mask].T
            fig.add_trace(
                go.Scatter3d(
                    x=x, y=y, z=z, mode="markers", hoverinfo="skip", name=name,
                    legendgroup=name, showlegend=n == 0, marker=dict(size=1.5, color=colour),
                ),
                row=n // cols + 1, col=n % cols + 1,
            )
    fig.update_layout(legend=dict(itemsizing="constant"))  # Legend dots readable, not 1.5 px.
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
