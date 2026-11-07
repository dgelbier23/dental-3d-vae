import numpy as np
import pytest
import trimesh
from PIL import Image

from dental_3d_vae.viz.colours import LIFE_HISTORIES
from dental_3d_vae.viz.pointclouds import GUM_COLOUR, plot_jaw, plot_pointclouds, save_rotating_gif


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


def test_save_rotating_gif_writes_one_frame_per_angle_and_leaves_figure_unchanged(tmp_path) -> None:
    pytest.importorskip("kaleido")
    fig = plot_pointclouds([np.random.default_rng(0).random((50, 3))], titles=["tooth"])
    camera_before = fig.layout.scene.camera.to_plotly_json()

    save_rotating_gif(fig, tmp_path / "spin.gif", frames=3, width=200, height=150)

    with Image.open(tmp_path / "spin.gif") as gif:
        assert gif.n_frames == 3 and gif.size == (200, 150)
    assert fig.layout.scene.camera.to_plotly_json() == camera_before
