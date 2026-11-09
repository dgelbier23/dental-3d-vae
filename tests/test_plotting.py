from collections.abc import Iterator

import pytest

go = pytest.importorskip("plotly.graph_objects")
pio = pytest.importorskip("plotly.io")

from dental_3d_vae.viz.colours import LIFE_HISTORIES  # noqa: E402
from dental_3d_vae.viz.plotting import plot_training_history, use_theme  # noqa: E402


@pytest.fixture(autouse=True)
def restore_default_template() -> Iterator[None]:
    previous = pio.templates.default
    yield
    pio.templates.default = previous


def test_new_figures_use_all_seven_colours() -> None:
    use_theme()
    template = go.Figure().layout.template

    assert list(template.layout.colorway) == list(LIFE_HISTORIES)
    assert [stop[1] for stop in template.data.heatmap[0].colorscale] == list(LIFE_HISTORIES)


def test_training_history_plots_both_losses_and_marks_the_best_epoch() -> None:
    metrics = [{"epoch": e, "train_chamfer": 1 / e, "val_chamfer": v, "train_kl": 0.1, "val_kl": 0.2}
               for e, v in ((1, 0.9), (2, 0.4), (3, 0.6))]

    fig = plot_training_history(metrics)

    assert [trace.name for trace in fig.data] == ["Training", "Validation"] * 2
    assert [shape.x0 for shape in fig.layout.shapes] == [2, 2]
    assert "epoch 2" in fig.layout.title.text
