"""Point-cloud VAE: encoder -> (mu, logvar) -> sampled z -> decoder, plus building and loading."""

from pathlib import Path

import torch
from torch import nn

from dental_3d_vae.ml.models.decoders import MLPDecoder
from dental_3d_vae.ml.models.dgcnn import DGCNNEncoder
from dental_3d_vae.ml.models.pointnet import PointNetEncoder

# Names as stored in experiment configs and the dissertation checkpoints.
ENCODERS = {"simple_pointnet": PointNetEncoder, "dgcnnv2": DGCNNEncoder}
DECODERS = {"mlp": MLPDecoder}


class VAE(nn.Module):
    """Variational autoencoder for (B, N, 3) point clouds."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = mu + sigma * noise, keeping gradients flowing to mu and logvar."""
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training pass: ``(reconstruction, mu, logvar)``, decoding a sampled z."""
        mu, logvar = self.encoder(x)
        return self.decoder(self.reparameterize(mu, logvar)), mu, logvar

    def reconstruct(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """Reconstruction from z = mu (deterministic), or from a sampled z if ``sample``."""
        mu, logvar = self.encoder(x)
        return self.decoder(self.reparameterize(mu, logvar) if sample else mu)


def build_vae(encoder: str, decoder: str, latent_dim: int = 32, num_points: int = 2048) -> VAE:
    """A new VAE from encoder and decoder names (keys of ENCODERS and DECODERS)."""
    return VAE(ENCODERS[encoder](latent_dim), DECODERS[decoder](latent_dim, num_points))


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> tuple[VAE, dict]:
    """Rebuild a saved VAE from its config and ``model_state_dict``.

    Returns the model in eval mode and the whole checkpoint dict (results and
    metadata included). Unpickles arbitrary objects, so only load trusted files.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_vae(config["encoder"], config["decoder"], config["latent_dim"], config["num_points"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint
