import pytest
import torch

from dental_3d_vae.ml.models.vae import DECODERS, ENCODERS, build_vae, load_checkpoint

POINTS = torch.randn(4, 256, 3)


@pytest.mark.parametrize("encoder", sorted(ENCODERS))
@pytest.mark.parametrize("decoder", sorted(DECODERS))
def test_every_encoder_decoder_pair_has_the_right_shapes(encoder: str, decoder: str) -> None:
    model = build_vae(encoder, decoder, latent_dim=8, num_points=128)

    reconstruction, mu, logvar = model(POINTS)

    assert reconstruction.shape == (4, 128, 3)
    assert mu.shape == logvar.shape == (4, 8)


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
