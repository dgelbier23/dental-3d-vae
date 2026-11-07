"""Interactive 3D views of jaw scans and tooth point clouds, in the LifeHistories colours."""

import numpy as np
import plotly.graph_objects as go
import trimesh
from plotly.subplots import make_subplots

from dental_3d_vae.viz.colours import LIFE_HISTORIES

GUM_COLOUR = LIFE_HISTORIES[3]  # light sand
TOOTH_COLOURS = LIFE_HISTORIES[:3] + LIFE_HISTORIES[4:]  # the other six, cycled


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


def plot_pointclouds(clouds: list, titles: list[str], cols: int = 4) -> go.Figure:
    """One rotatable 3D panel per (N, 3) point cloud, shaded dark to light by height."""
    cols = min(cols, len(clouds))
    rows = -(-len(clouds) // cols)  # ceiling division
    fig = make_subplots(
        rows=rows, cols=cols, specs=[[{"type": "scene"}] * cols] * rows,
        subplot_titles=titles, horizontal_spacing=0.01, vertical_spacing=0.04,
    )
    for n, cloud in enumerate(clouds):
        x, y, z = np.asarray(cloud).T
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z, mode="markers", hoverinfo="skip", showlegend=False,
                marker=dict(size=1.5, color=z, colorscale=list(LIFE_HISTORIES)),
            ),
            row=n // cols + 1, col=n % cols + 1,
        )
    _hide_axes(fig)
    fig.update_layout(height=280 * rows, margin=dict(l=0, r=0, t=30, b=0))
    return fig
