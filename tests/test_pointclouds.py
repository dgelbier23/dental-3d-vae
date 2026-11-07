import numpy as np
import trimesh

from dental_3d_vae.viz.colours import LIFE_HISTORIES
from dental_3d_vae.viz.pointclouds import GUM_COLOUR, plot_jaw, plot_pointclouds


def test_plot_jaw_colours_gum_and_neighbouring_teeth_differently() -> None:
    jaw = trimesh.creation.icosphere(subdivisions=1)
    labels = np.array([0, 11, 12] * len(jaw.vertices))[: len(jaw.vertices)]

    colours = plot_jaw(jaw, labels).data[0].vertexcolor

    assert colours[0] == GUM_COLOUR
    assert colours[1] != colours[2] and GUM_COLOUR not in (colours[1], colours[2])
    assert set(colours) <= set(LIFE_HISTORIES)


def test_plot_pointclouds_one_true_proportion_panel_per_cloud() -> None:
    clouds = [np.random.default_rng(seed).random((20, 3)) for seed in range(5)]

    fig = plot_pointclouds(clouds, titles=[f"tooth {n}" for n in range(5)], cols=4)

    assert len(fig.data) == 5
    assert [a.text for a in fig.layout.annotations] == [f"tooth {n}" for n in range(5)]
    assert fig.layout.scene.aspectmode == "data" and fig.layout.scene5.aspectmode == "data"
