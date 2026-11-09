import pytest
import torch

from dental_3d_vae.ml.models.decoders import AtlasNetDecoder, FoldingNetDecoder
from dental_3d_vae.ml.models.dgcnn import edge_features, knn
from dental_3d_vae.ml.models.vae import DECODERS, ENCODERS, build_vae, load_checkpoint

POINTS = torch.randn(4, 256, 3)


def test_knn_matches_brute_force_and_includes_the_point_itself() -> None:
    x = torch.randn(2, 5, 40)  # (B, C, N): 40 points with 5 features

    idx = knn(x, k=6)

    brute = torch.cdist(x.transpose(1, 2), x.transpose(1, 2)).argsort(dim=-1)[..., :6]
    assert torch.equal(idx.sort(dim=-1).values, brute.sort(dim=-1).values)
    assert (idx[..., 0] == torch.arange(40)).all()  # nearest is itself


def test_edge_features_are_neighbour_minus_point_then_point() -> None:
    x = torch.randn(2, 3, 30)

    features = edge_features(x, k=4)  # (B, 2C, N, k)

    assert features.shape == (2, 6, 30, 4)
    point = x[:, :, :, None].expand(-1, -1, -1, 4)
    assert torch.equal(features[:, 3:], point)
    assert torch.equal(features[:, :3, :, 0], torch.zeros(2, 3, 30))  # neighbour 0 is the point


@pytest.mark.parametrize("encoder", sorted(ENCODERS))
@pytest.mark.parametrize("decoder", sorted(DECODERS))
def test_every_encoder_decoder_pair_has_the_right_shapes(encoder: str, decoder: str) -> None:
    model = build_vae(encoder, decoder, latent_dim=8, num_points=128)

    reconstruction, mu, logvar = model(POINTS)

    assert reconstruction.shape == (4, 128, 3)
    assert mu.shape == logvar.shape == (4, 8)


def test_decoder_grids_have_the_layout_stored_in_checkpoints() -> None:
    folding = FoldingNetDecoder(num_points=2048).grid
    atlas = AtlasNetDecoder(num_points=2048).uv_grid

    assert folding.shape == (2048, 2) and len(folding[:2025].unique(dim=0)) == 2025  # 45 x 45
    assert torch.equal(folding[2025:], folding[:23])  # padded by repeating the first points
    assert atlas.shape == (512, 2) and torch.equal(atlas[23], torch.tensor([1 / 22, 0.0]))  # 23 per row


def test_reconstruct_uses_mu_unless_sampling() -> None:
    model = build_vae("simple_pointnet", "mlp", latent_dim=8, num_points=128).eval()

    with torch.no_grad():
        assert torch.equal(model.reconstruct(POINTS), model.reconstruct(POINTS))
        assert not torch.equal(model.reconstruct(POINTS, sample=True), model.reconstruct(POINTS, sample=True))


def test_load_checkpoint_rebuilds_the_saved_model(tmp_path) -> None:
    config = {"encoder": "simple_pointnet", "decoder": "mlp", "latent_dim": 8, "num_points": 128}
    saved = build_vae(**config).eval()
    torch.save({"config": config, "model_state_dict": saved.state_dict(), "best_epoch": 3},
               tmp_path / "run.pt")

    model, checkpoint = load_checkpoint(tmp_path / "run.pt")

    assert not model.training and checkpoint["best_epoch"] == 3
    with torch.no_grad():
        assert torch.equal(model.reconstruct(POINTS), saved.reconstruct(POINTS))
