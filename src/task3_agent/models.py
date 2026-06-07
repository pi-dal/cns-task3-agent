"""VAE-based protein conformational ensemble generator.

Per-residue VAE: each CA atom is an independent training example.
This learns transferable local backbone geometry across proteins.
Ensemble generation: sample per-residue latents multiple times.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PerResidueVAE(nn.Module):
    """Per-residue VAE for protein backbone ensemble generation.

    Each CA coordinate is encoded independently into a latent distribution.
    The decoder reconstructs from sampled latent + positional encoding.

    This architecture:
    - Shares parameters across all residues (weight tying)
    - Learns local backbone geometry patterns (generalizes across proteins)
    - Generates diverse ensembles via per-residue latent sampling

    Args:
        latent_dim: Dimensionality of per-residue latent space.
        hidden_dim: Hidden layer width.
    """

    def __init__(self, latent_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # --- Encoder: (x,y,z) → μ, logvar ---
        self.enc_fc1 = nn.Linear(3, hidden_dim)
        self.enc_bn1 = nn.BatchNorm1d(hidden_dim)
        self.enc_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.enc_bn2 = nn.BatchNorm1d(hidden_dim)
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        # --- Decoder: z + position encoding → (x,y,z) ---
        self.dec_fc1 = nn.Linear(latent_dim + 1, hidden_dim)
        self.dec_bn1 = nn.BatchNorm1d(hidden_dim)
        self.dec_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dec_bn2 = nn.BatchNorm1d(hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, 3)

    def encode(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode each CA coordinate into a latent distribution.

        Args:
            coords: (N, 3) tensor of CA coordinates.

        Returns:
            mu: (N, latent_dim) per-residue means.
            logvar: (N, latent_dim) per-residue log variances.
        """
        h = F.relu(self.enc_bn1(self.enc_fc1(coords)))   # (N, hidden)
        h = F.relu(self.enc_bn2(self.enc_fc2(h)))         # (N, hidden)
        mu = self.enc_mu(h)                                # (N, latent)
        logvar = self.enc_logvar(h)                        # (N, latent)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, n_residues: int) -> torch.Tensor:
        """Decode latents + position encoding to coordinates.

        Args:
            z: (N, latent_dim) per-residue latents.
            n_residues: Number of residues (may differ from z.shape[0] for
                        generation on different-length protein).

        Returns:
            coords: (N, 3) decoded CA coordinates.
        """
        pos = torch.linspace(-1.0, 1.0, n_residues, device=z.device).unsqueeze(-1)  # (N, 1)
        # If z has different N than n_residues, we broadcast
        dec_in = torch.cat([z, pos], dim=-1)  # (N, latent+1)
        h = F.relu(self.dec_bn1(self.dec_fc1(dec_in)))
        h = F.relu(self.dec_bn2(self.dec_fc2(h)))
        out = self.dec_out(h)  # (N, 3)
        return out

    def forward(self, coords: torch.Tensor, beta: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """VAE forward pass.

        Args:
            coords: (N, 3) input CA coordinates.
            beta: KL divergence weight (β-VAE).

        Returns:
            recon: (N, 3) reconstructed coordinates.
            loss: scalar VAE loss = recon_loss + β * mean_KL.
            kl: mean KL divergence per residue.
        """
        N = coords.shape[0]
        mu, logvar = self.encode(coords)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, N)

        # Reconstruction loss
        recon_loss = F.mse_loss(recon, coords)

        # KL divergence per residue: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_per_res = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # (N,)
        kl_mean = kl_per_res.mean()

        total_loss = recon_loss + beta * kl_mean
        return recon, total_loss, kl_mean

    @torch.no_grad()
    def generate_ensemble(self, coords: torch.Tensor, n_samples: int = 20, temp: float = 1.0) -> torch.Tensor:
        """Generate an ensemble of conformations by repeated latent sampling.

        Args:
            coords: (N, 3) input CA coordinates.
            n_samples: Number of ensemble members.
            temp: Temperature scaling of latent std (higher = more diverse).

        Returns:
            ensemble: (n_samples, N, 3) generated conformations.
        """
        mu, logvar = self.encode(coords)
        samples = []
        for _ in range(n_samples):
            std = torch.exp(0.5 * logvar) * temp
            eps = torch.randn_like(std)
            z = mu + eps * std
            sample = self.decode(z, coords.shape[0])
            samples.append(sample.unsqueeze(0))
        return torch.cat(samples, dim=0)

    def get_config(self) -> dict:
        return {
            "model_type": "PerResidueVAE",
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
        }
