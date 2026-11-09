import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

from dental_3d_vae.ml.evaluate import summarise  # noqa: E402
from dental_3d_vae.viz.results import (  # noqa: E402
    ARCHES,
    plot_beta_ablation,
    plot_distributions,
    plot_loss_curves,
    plot_metric_relationship,
    plot_metrics_vs_missingness,
    plot_odontogram,
    plot_table,
)

TEETH = [tooth for arch in ARCHES.values() for tooth in arch]


def _results(offset: float, seed: int = 0) -> pd.DataFrame:
    """Two test teeth per FDI position at each missing fraction; error grows with offset."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame([(tooth, f) for tooth in TEETH for f in (0.0, 0.15, 0.3, 0.45) for _ in range(2)],
                         columns=["tooth_id", "missing_fraction"])
    frame["cd"] = offset + rng.random(len(frame)) / 100
    frame["mrre"] = np.where(frame["missing_fraction"] > 0, frame["cd"] / 30, np.nan)
    frame["orpe"] = frame["cd"] / 40
    return frame


RESULTS = {"Good": _results(0.05), "Worse": _results(0.07, seed=1)}


def test_odontogram_lays_out_every_tooth_and_credits_the_better_model() -> None:
    fig = plot_odontogram(RESULTS)

    assert [len(trace.z) * len(trace.z[0]) for trace in fig.data] == [32, 32]
    assert fig.data[0].text[0][0].startswith("<b>18</b>") and fig.data[0].text[1][0].startswith("<b>48</b>")
    assert "lowest on 32 of 32 teeth" in fig.layout.annotations[0].text
    assert len(fig.layout.shapes) == 32  # every tooth outlined, all on the better model


def test_summary_table_and_charts_build_from_results() -> None:
    table = summarise(RESULTS)
    history = [{"epoch": e, "val_chamfer": 1 / e, "val_kl": 0.1} for e in range(1, 6)]

    assert list(table.index) == ["Good", "Worse"] and table.loc["Good", "Mean"] < table.loc["Worse", "Mean"]
    assert "<b>" in plot_table(table, "t", lowest_is_best=("Mean",)).data[0].cells.values[1][0]
    assert len(plot_loss_curves({"a": history, "b": history}).data) == 4  # line + best-epoch dot each
    assert "Lowest median: Good" in plot_distributions(RESULTS).layout.title.subtitle.text
    assert len(plot_metrics_vs_missingness(RESULTS).data) == 6  # 2 models x 3 metrics
    scatter = plot_metric_relationship(RESULTS, "cd", "mrre")
    assert sum(a.text.startswith("r = ") for a in scatter.layout.annotations) == 2
    beta = plot_beta_ablation({0.0: RESULTS["Good"], 1e-3: RESULTS["Worse"]}, {0.0: 6.0, 1e-3: 2.8}, selected=1e-3)
    assert list(beta.data[1].x) == ["0", "10⁻³"]
