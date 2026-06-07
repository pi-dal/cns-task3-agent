"""VAE-based protein conformational ensemble generator.

Per-residue VAE: each CA atom is an independent training example.
This learns transferable local backbone geometry across proteins.
Ensemble generation: sample per-residue latents multiple times.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _sinusoidal_pos(n_residues: int, dim: int = 32, device: torch.device = None) -> torch.Tensor:
    """Sinusoidal positional encoding for Transformer."""
    pos = torch.linspace(0.0, 1.0, n_residues, device=device).unsqueeze(-1)
    freqs = 10000.0 ** (torch.arange(0, dim, 2, device=device).float() / dim)
    angles = pos * freqs.unsqueeze(0)
    pe = torch.zeros(n_residues, dim, device=device)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe


class TransformerVAE(nn.Module):
    """VAE with Transformer encoder for residue-residue interaction modeling.

    Encodes (N, 3) CA coordinates through a Transformer encoder with
    self-attention to capture non-local residue interactions.
    Decoder is the same per-residue MLP as PerResidueVAE.
    """

    def __init__(self, latent_dim: int = 24, hidden_dim: int = 192,
                 num_layers: int = 2, dropout: float = 0.0,
                 nhead: int = 4, num_transformer_layers: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pe_dim = 32

        # Input projection: (x,y,z) -> hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Output: hidden_dim -> mu, logvar
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        # --- Decoder: same as PerResidueVAE ---
        dec_layers = []
        in_dim = latent_dim + 1  # z + position
        for i in range(num_layers):
            dec_layers.append(MLPBlock(in_dim, hidden_dim, dropout))
            in_dim = hidden_dim
        self.decoder = nn.Sequential(*dec_layers)
        self.dec_out = nn.Linear(hidden_dim, 3)

    def encode(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode through Transformer with self-attention."""
        N = coords.shape[0]
        h = self.input_proj(coords).unsqueeze(0)  # (1, N, hidden)
        pe = _sinusoidal_pos(N, self.pe_dim, coords.device).unsqueeze(0)  # (1, N, 32)
        h = h + pe  # Add positional encoding
        h = self.transformer(h).squeeze(0)  # (N, hidden)
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, n_residues: int) -> torch.Tensor:
        """Decode latents + position encoding to coordinates."""
        pos = torch.linspace(-1.0, 1.0, n_residues, device=z.device).unsqueeze(-1)
        dec_in = torch.cat([z, pos], dim=-1)
        h = self.decoder(dec_in)
        out = self.dec_out(h)
        return out

    def forward(self, coords: torch.Tensor, beta: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = coords.shape[0]
        mu, logvar = self.encode(coords)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, N)
        recon_loss = F.mse_loss(recon, coords)
        kl_per_res = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_mean = kl_per_res.mean()
        total_loss = recon_loss + beta * kl_mean
        return recon, total_loss, kl_mean

    @torch.no_grad()
    def generate_ensemble(self, coords: torch.Tensor, n_samples: int = 20, temp: float = 1.0) -> torch.Tensor:
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
            "model_type": "TransformerVAE",
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
        }


class MLPBlock(nn.Module):
    """MLP block with LayerNorm + ReLU + optional dropout."""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(F.relu(self.ln(self.fc(x))))


class PerResidueVAE(nn.Module):
    """Per-residue VAE for protein backbone ensemble generation.

    Each CA coordinate is encoded independently into a latent distribution.
    The decoder reconstructs from sampled latent + positional encoding.

    Args:
        latent_dim: Dimensionality of per-residue latent space.
        hidden_dim: Hidden layer width.
        num_layers: Number of MLP layers (2-4).
        dropout: Dropout probability (0 = no dropout).
    """

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # --- Encoder: (x,y,z) → μ, logvar ---
        self.enc_fc1 = nn.Linear(3, hidden_dim)
        self.enc_bn1 = nn.BatchNorm1d(hidden_dim)
        self.num_layers = num_layers

        # --- Encoder: (x,y,z) → μ, logvar ---
        enc_layers = []
        in_dim = 3
        for i in range(num_layers):
            enc_layers.append(MLPBlock(in_dim, hidden_dim, dropout))
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*enc_layers)
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        # --- Decoder: z + position encoding → (x,y,z) ---
        dec_layers = []
        in_dim = latent_dim + 1
        for i in range(num_layers):
            dec_layers.append(MLPBlock(in_dim, hidden_dim, dropout))
            in_dim = hidden_dim
        self.decoder = nn.Sequential(*dec_layers)
        self.dec_out = nn.Linear(hidden_dim, 3)

    def encode(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode each CA coordinate into a latent distribution."""
        h = self.encoder(coords)                             # (N, hidden)
        mu = self.enc_mu(h)                                   # (N, latent)
        logvar = self.enc_logvar(h)                           # (N, latent)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, n_residues: int) -> torch.Tensor:
        """Decode latents + position encoding to coordinates."""
        pos = torch.linspace(-1.0, 1.0, n_residues, device=z.device).unsqueeze(-1)  # (N, 1)
        dec_in = torch.cat([z, pos], dim=-1)  # (N, latent+1)
        h = self.decoder(dec_in)
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
            "num_layers": self.num_layers,
        }
