"""Results charts: models' per-tooth test results and training runs, side by side.

``results`` arguments map a label to per-tooth results (``evaluate`` output, or
``load_dissertation_run(...)["results"]``), in display order; ``histories`` map a label
to a run's per-epoch metrics. Colours follow the label's position, so keep the order
fixed when comparing figures.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

from dental_3d_vae.viz.colours import LIFE_HISTORIES
from dental_3d_vae.viz.plotting import SEQUENTIAL, SERIES, rgba

LABELS = {
    "cd": "Chamfer distance",
    "stochastic_cd": "Chamfer distance, sampled z",
    "mrre": "Missing-region error (MRRE)",
    "orpe": "Observed-region error (ORPE)",
    "sce": "Stochastic consistency error (SCE)",
    "train_chamfer": "Training Chamfer distance",
    "val_chamfer": "Validation Chamfer distance",
    "train_kl": "Training KL divergence",
    "val_kl": "Validation KL divergence",
}
# FDI numbers as seen facing the patient: upper arch on top, the patient's right on the left.
# Multi-panel charts: legend centred below the panels, clear of the panel titles.
PANEL_LAYOUT = dict(legend=dict(y=-0.2, x=0.5, xanchor="center", yanchor="top"), margin=dict(b=120))
ARCHES = {"Upper": [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28],
          "Lower": [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]}


def plot_loss_curves(histories: dict[str, list[dict]], metric: str = "val_chamfer") -> go.Figure:
    """One line per run over epochs (log scale), with each run's minimum marked."""
    fig = go.Figure()
    for colour, (label, history) in zip(SERIES, histories.items()):
        epochs, values = [m["epoch"] for m in history], [m[metric] for m in history]
        best = int(np.nanargmin(values))
        fig.add_scatter(x=epochs, y=values, name=label, mode="lines", line=dict(color=colour, width=2),
                        legendgroup=label, hovertemplate=f"{label}<br>epoch %{{x}}: %{{y:.5f}}<extra></extra>")
        fig.add_scatter(x=[epochs[best]], y=[values[best]], mode="markers", legendgroup=label, showlegend=False,
                        marker=dict(color=colour, size=9, line=dict(color="white", width=2)),
                        hovertemplate=f"{label}: best %{{y:.5f}} at epoch %{{x}}<extra></extra>")
    fig.update_yaxes(type="log", title=LABELS.get(metric, metric))
    fig.update_xaxes(title="Epoch")
    fig.update_layout(title=dict(text=LABELS.get(metric, metric), subtitle_text="Dots mark each run's best epoch"),
                      height=460)
    return fig


def plot_distributions(results: dict[str, pd.DataFrame], metric: str = "cd") -> go.Figure:
    """Each model's per-tooth distribution of ``metric``: a violin with its quartile box and mean."""
    fig = go.Figure()
    for colour, (label, frame) in zip(SERIES, results.items()):
        fig.add_violin(x=frame[metric], name=label, orientation="h", side="negative", width=1.6, points=False,
                       line=dict(color=colour, width=1.5), fillcolor=rgba(colour, 0.35), meanline_visible=True,
                       box=dict(visible=True, width=0.15, fillcolor="white", line_color=colour),
                       hoveron="kde+points")
    medians = {label: frame[metric].median() for label, frame in results.items()}
    best = min(medians, key=medians.get)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title=LABELS.get(metric, metric))
    fig.update_layout(title=dict(text=f"{LABELS.get(metric, metric)} per tooth", subtitle_text=(
        f"Lowest median: {best} ({medians[best]:.4f}). Box: middle 50% of teeth; line: mean")),
                      showlegend=False, height=140 + 110 * len(results), violingap=0.25)
    return fig


def plot_odontogram(results: dict[str, pd.DataFrame], metric: str = "cd", mark_best: bool = True) -> go.Figure:
    """Mean ``metric`` per tooth, laid out as a dental chart: one chart per entry of ``results``.

    One colour scale for all charts. With ``mark_best``, each tooth's lowest value across
    the charts is outlined and each chart's title counts its outlined teeth.
    """
    means = pd.DataFrame({label: frame.groupby("tooth_id")[metric].mean() for label, frame in results.items()})
    best = means.idxmin(axis=1)
    titles = [f"<b>{label}</b>" + (f"  ·  lowest on {(best == label).sum()} of {len(means)} teeth"
                                   if mark_best and len(results) > 1 else "") for label in results]
    fig = make_subplots(rows=len(results), cols=1, subplot_titles=titles, vertical_spacing=0.2 / len(results))
    for row, label in enumerate(results, start=1):
        z = [[means[label].get(tooth, np.nan) for tooth in arch] for arch in ARCHES.values()]
        text = [[f"<b>{tooth}</b><br>{value:.4f}" if not np.isnan(value) else str(tooth)
                 for tooth, value in zip(arch, values)] for arch, values in zip(ARCHES.values(), z)]
        fig.add_heatmap(z=z, x=list(range(16)), y=list(ARCHES), text=text, texttemplate="%{text}",
                        textfont=dict(size=10), xgap=3, ygap=3, coloraxis="coloraxis",
                        hovertemplate=f"%{{y}} tooth %{{text}}<extra>{label}</extra>", row=row, col=1)
        if mark_best and len(results) > 1:
            for y, arch in enumerate(ARCHES.values()):
                for x, tooth in enumerate(arch):
                    if best.get(tooth) == label:
                        fig.add_shape(type="rect", x0=x - 0.46, x1=x + 0.46, y0=y - 0.44, y1=y + 0.44,
                                      line=dict(color=LIFE_HISTORIES[0], width=2.5), row=row, col=1)
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(
        title=dict(text=f"{LABELS.get(metric, metric)} by tooth", subtitle_text=(
            "Mean per tooth as seen facing the patient (FDI numbers); darker = higher error"
            + ("; outlined = lowest of the models" if mark_best and len(results) > 1 else ""))),
        coloraxis=dict(colorscale=SEQUENTIAL, colorbar=dict(title=dict(text="Mean", side="right"), thickness=12)),
        height=150 + 190 * len(results), width=1150)
    return fig


def plot_beta_ablation(results: dict[float, pd.DataFrame], kl: dict[float, float],
                       selected: float | None = None) -> go.Figure:
    """Test Chamfer distance (mean, with the middle 50% of teeth shaded) and KL against beta."""
    betas = list(results)
    x = [_beta_label(beta) for beta in betas]
    cd = pd.DataFrame({beta: frame["cd"].describe() for beta, frame in results.items()}).T
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("Test Chamfer distance", "KL divergence (latent regularisation)"))
    fig.add_scatter(x=x + x[::-1], y=list(cd["75%"]) + list(cd["25%"])[::-1], fill="toself", mode="none",
                    fillcolor=rgba(SERIES[0], 0.15), name="Middle 50% of teeth", hoverinfo="skip", row=1, col=1)
    fig.add_scatter(x=x, y=cd["mean"], name="Mean", mode="lines+markers", line=dict(color=SERIES[0], width=2),
                    marker_size=9, hovertemplate="β = %{x}: mean %{y:.5f}<extra></extra>", row=1, col=1)
    fig.add_scatter(x=x, y=[kl[beta] for beta in betas], name="KL", mode="lines+markers", showlegend=False,
                    line=dict(color=SERIES[1], width=2), marker_size=9,
                    hovertemplate="β = %{x}: KL %{y:.3f}<extra></extra>", row=1, col=2)
    if selected is not None:
        for col in (1, 2):
            fig.add_vline(x=x.index(_beta_label(selected)), line=dict(color=LIFE_HISTORIES[4], dash="dot", width=1.5),
                          row=1, col=col)
        fig.add_annotation(x=_beta_label(selected), y=1, yref="y domain", text=f"chosen β = {_beta_label(selected)}",
                           showarrow=False, xanchor="left", xshift=6, font=dict(color=LIFE_HISTORIES[0]),
                           row=1, col=1)
    fig.update_xaxes(title="β (KL weight)", type="category", tickangle=0)
    fig.update_layout(title=dict(text="Effect of the KL weight β", subtitle_text=(
        "More regularisation shrinks the KL divergence but costs reconstruction accuracy")), height=500,
                      **PANEL_LAYOUT)
    return fig


def _beta_label(beta: float) -> str:
    """0.001 -> "10⁻³"; values that read well as decimals stay decimals."""
    if 0 < beta < 0.01:
        exponent = round(np.log10(beta))
        return "10" + str(exponent).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{beta:g}"


def plot_metrics_vs_missingness(results: dict[str, pd.DataFrame],
                                metrics: tuple[str, ...] = ("cd", "mrre", "orpe")) -> go.Figure:
    """Mean of each metric per missing fraction, one line per model, with 95% confidence intervals."""
    fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=[LABELS.get(m, m) for m in metrics],
                        horizontal_spacing=0.1)
    for colour, (label, frame) in zip(SERIES, results.items()):
        for col, metric in enumerate(metrics, start=1):
            stats = frame.groupby("missing_fraction")[metric].agg(["mean", "sem"]).dropna()
            fig.add_scatter(x=stats.index * 100, y=stats["mean"], name=label, legendgroup=label,
                            showlegend=col == 1, mode="lines+markers", line=dict(color=colour, width=2),
                            marker_size=8, error_y=dict(array=1.96 * stats["sem"], thickness=1.5, width=4),
                            hovertemplate=f"{label}, %{{x}}% missing: %{{y:.5f}}<extra></extra>", row=1, col=col)
    fig.update_xaxes(title="Points removed (%)", tickvals=[0, 15, 30, 45])
    fig.update_layout(title=dict(text="Reconstruction error as more of the tooth is missing", subtitle_text=(
        "Mean over test teeth, with 95% confidence intervals; MRRE is undefined when nothing is removed")),
                      height=480, **PANEL_LAYOUT)
    return fig


def plot_metric_relationship(results: dict[str, pd.DataFrame], x: str = "cd", y: str = "mrre") -> go.Figure:
    """Per-tooth scatter of ``y`` against ``x`` for each model, with a fitted line and Pearson r.

    Points are coloured by missing fraction when the results have one.
    """
    fig = make_subplots(rows=1, cols=len(results), horizontal_spacing=0.07,
                        subplot_titles=[f"<b>{label}</b>" for label in results])
    for col, (label, frame) in enumerate(results.items(), start=1):
        frame = frame.dropna(subset=[x, y])
        groups = frame.groupby("missing_fraction") if "missing_fraction" in frame else [(None, frame)]
        shades = sample_colorscale(SEQUENTIAL, np.linspace(0.4, 1, len(groups)))  # Palest sand is too faint.
        for shade, (fraction, group) in zip(shades, groups):
            name = "All teeth" if fraction is None else f"{fraction:.0%} missing"
            fig.add_scatter(x=group[x], y=group[y], mode="markers", name=name, legendgroup=name,
                            showlegend=col == 1, marker=dict(color=shade, size=5, opacity=0.65),
                            hovertemplate=f"{name}<br>{x.upper()} %{{x:.4f}}<br>{y.upper()} %{{y:.5f}}<extra></extra>",
                            row=1, col=col)
        slope, intercept = np.polyfit(frame[x], frame[y], 1)
        ends = np.array([frame[x].min(), frame[x].max()])
        fig.add_scatter(x=ends, y=slope * ends + intercept, mode="lines", showlegend=False, hoverinfo="skip",
                        line=dict(color=LIFE_HISTORIES[0], width=2, dash="dash"), row=1, col=col)
        fig.add_annotation(text=f"r = {frame[x].corr(frame[y]):.3f}", x=0.03, y=0.97, xref="x domain",
                           yref="y domain", showarrow=False, xanchor="left", yanchor="top",
                           font=dict(size=14), bgcolor="white", row=1, col=col)
    fig.update_xaxes(title=LABELS.get(x, x))
    fig.update_yaxes(title=LABELS.get(y, y), col=1)
    fig.update_layout(title=dict(text=f"{LABELS.get(y, y)} against {LABELS.get(x, x)}",
                                 subtitle_text="One point per test tooth; dashed: least-squares fit; r: Pearson correlation"),
                      height=520, **PANEL_LAYOUT)
    return fig


def plot_table(table: pd.DataFrame, title: str, lowest_is_best: tuple[str, ...] = ()) -> go.Figure:
    """A results table as a clean figure; in ``lowest_is_best`` columns the best value is bold."""
    columns = []
    for name in table.columns:
        values = table[name]
        if pd.api.types.is_float_dtype(values):
            text = [f"{v:.5f}" if abs(v) < 1 else f"{v:,.2f}" for v in values]
        else:
            text = [f"{v:,}" if isinstance(v, (int, np.integer)) else str(v) for v in values]
        if name in lowest_is_best:
            text = [f"<b>{t}</b>" if v == values.min() else t for t, v in zip(text, values)]
        columns.append(text)
    stripes = [rgba(LIFE_HISTORIES[3], 0.25) if i % 2 else "white" for i in range(len(table))]
    fig = go.Figure(go.Table(
        header=dict(values=[f"<b>{table.index.name or ''}</b>", *[f"<b>{c}</b>" for c in table.columns]],
                    fill_color=LIFE_HISTORIES[6], font=dict(color="white", size=13), align="left", height=34),
        cells=dict(values=[[f"<b>{i}</b>" for i in table.index], *columns], fill_color=[stripes],
                   align="left", height=30, font=dict(size=13)),
    ))
    fig.update_layout(title=dict(text=title), height=130 + 30 * len(table), margin=dict(t=70, b=10, l=0, r=0))
    return fig
