"""Plotly theme: the LifeHistories colours on Plotly's white template."""

import plotly.graph_objects as go
import plotly.io as pio

from ai_lab.viz.colours import LIFE_HISTORIES


def use_theme() -> None:
    """Make LifeHistories the default colours for traces and heatmaps in new figures."""
    colours = list(LIFE_HISTORIES)
    pio.templates["life_histories"] = go.layout.Template(
        layout=dict(colorway=colours),
        data=dict(heatmap=[go.Heatmap(colorscale=colours)]),
    )
    pio.templates.default = "plotly_white+life_histories"
