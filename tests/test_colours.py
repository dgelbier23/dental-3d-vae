from dental_3d_vae.viz.colours import LIFE_HISTORIES


def test_life_histories_matches_original_source_exactly() -> None:
    # Copied from MVZlibrary R/MVZlibrary_palettes.R (commit 6421b45); order matters.
    assert LIFE_HISTORIES == (
        "#844836", "#b07156", "#b99364", "#dac3a9", "#8fa997", "#6baeb0", "#376988",
    )
