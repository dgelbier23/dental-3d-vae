"""Plotly theme (LifeHistories colours on a quiet white layout) and training plots."""

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from dental_3d_vae.viz.colours import LIFE_HISTORIES

# Series colours in order, chosen with a colour-vision check: any first 2 or 3 stay
# distinguishable for colour-blind readers; from the 4th on, charts also label lines directly.
SERIES = tuple(LIFE_HISTORIES[i] for i in (6, 1, 5, 0, 2, 4, 3))
# Magnitude: light sand -> teal -> dark blue (lightness falls steadily).
SEQUENTIAL = [[0.0, LIFE_HISTORIES[3]], [0.55, LIFE_HISTORIES[5]], [1.0, LIFE_HISTORIES[6]]]
FONT = "Inter, Segoe UI, Helvetica Neue, Arial, sans-serif"


def rgba(colour: str, alpha: float) -> str:
    """A palette hex colour as a CSS rgba() string with the given opacity."""
    red, green, blue = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({red}, {green}, {blue}, {alpha})"


def use_theme() -> None:
    """Make LifeHistories and a clean layout the default for new figures.

    Left-aligned titles with room for a subtitle, faint sand gridlines, no axis lines.
    """
    grid = rgba(LIFE_HISTORIES[3], 0.45)
    axis = dict(gridcolor=grid, zeroline=False, showline=False, ticks="", title_standoff=8)
    pio.templates["life_histories"] = go.layout.Template(
        layout=dict(
            colorway=list(SERIES),
            colorscale=dict(sequential=SEQUENTIAL),
            font=dict(family=FONT, size=13),
            title=dict(x=0, xanchor="left", font=dict(size=19)),
            xaxis=axis, yaxis=axis,
            legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", title_text=""),
            hoverlabel=dict(font=dict(family=FONT)),
            margin=dict(l=70, r=40, t=110, b=60),
        ),
        data=dict(heatmap=[go.Heatmap(colorscale=SEQUENTIAL)]),
    )
    pio.templates.default = "plotly_white+life_histories"


def plot_training_history(metrics: list[dict]) -> go.Figure:
    """Loss curves from a run's metrics.jsonl: Chamfer distance and KL per epoch, best epoch marked.

    Training Chamfer distance comes from sampled latents and validation from z = mu, so
    the two are not computed identically.
    """
    epochs = [m["epoch"] for m in metrics]
    best = min(metrics, key=lambda m: m["val_chamfer"])
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Chamfer distance", "KL divergence"), horizontal_spacing=0.1)
    for column, key in ((1, "chamfer"), (2, "kl")):
        for split, label, colour in (("train", "Training (sampled z)", SERIES[1]), ("val", "Validation (z = μ)", SERIES[0])):
            fig.add_scatter(x=epochs, y=[m[f"{split}_{key}"] for m in metrics], name=label, mode="lines",
                            line=dict(color=colour, width=2), legendgroup=split, showlegend=column == 1,
                            row=1, col=column)
        fig.add_vline(x=best["epoch"], line=dict(dash="dot", color=LIFE_HISTORIES[4], width=1.5), row=1, col=column)
    fig.update_xaxes(title="Epoch")
    fig.update_layout(title=dict(text="Training history", subtitle_text=(
        f"Best validation Chamfer distance {best['val_chamfer']:.5f} at epoch {best['epoch']} (dotted line): "
        "the saved model")), hovermode="x unified", height=440)
    return fig
