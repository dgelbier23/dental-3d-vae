from collections.abc import Iterator

import pytest

go = pytest.importorskip("plotly.graph_objects")
pio = pytest.importorskip("plotly.io")

from dental_3d_vae.viz.colours import LIFE_HISTORIES  # noqa: E402
from dental_3d_vae.viz.plotting import use_theme  # noqa: E402


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
