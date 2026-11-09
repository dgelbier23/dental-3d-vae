"""Plotly theme (the LifeHistories colours on Plotly's white template) and training plots."""

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from dental_3d_vae.viz.colours import LIFE_HISTORIES


def use_theme() -> None:
    """Make LifeHistories the default colours for traces and heatmaps in new figures."""
    colours = list(LIFE_HISTORIES)
    pio.templates["life_histories"] = go.layout.Template(
        layout=dict(colorway=colours),
        data=dict(heatmap=[go.Heatmap(colorscale=colours)]),
    )
    pio.templates.default = "plotly_white+life_histories"


def plot_training_history(metrics: list[dict]) -> go.Figure:
    """Loss curves from a run's metrics.jsonl: Chamfer distance and KL per epoch, best epoch marked.

    Training Chamfer distance comes from sampled latents and validation from z = mu, so
    the two are not computed identically.
    """
    epochs = [m["epoch"] for m in metrics]
    best = min(metrics, key=lambda m: m["val_chamfer"])
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Chamfer distance", "KL divergence"))
    for column, key in ((1, "chamfer"), (2, "kl")):
        for split, label, colour in (("train", "Training", LIFE_HISTORIES[1]), ("val", "Validation", LIFE_HISTORIES[6])):
            fig.add_scatter(x=epochs, y=[m[f"{split}_{key}"] for m in metrics], name=label, mode="lines",
                            line_color=colour, legendgroup=split, showlegend=column == 1, row=1, col=column)
        fig.add_vline(x=best["epoch"], line_dash="dot", line_color=LIFE_HISTORIES[4], row=1, col=column)
    fig.update_xaxes(title="Epoch")
    fig.update_layout(title=f"Best validation Chamfer distance {best['val_chamfer']:.5f} at epoch {best['epoch']}",
                      hovermode="x unified", height=420)
    return fig
