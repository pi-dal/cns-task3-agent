"""Training and ensemble generation for CNS Task 3.

Uses the PerResidueVAE with LayerNorm to train on PDB structures
and generate conformational ensembles.
"""

import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timezone

from .models import PerResidueVAE


def load_normalized_inputs(entry_dirs: list[str]) -> dict:
    """Load normalized.pt files from entry directories."""
    all_coords = []
    entry_ids = []
    boundaries = []
    offset = 0
    
    for entry_dir in entry_dirs:
        npt_path = os.path.join(entry_dir, "normalized.pt")
        if not os.path.exists(npt_path):
            continue
        data = torch.load(npt_path, map_location="cpu", weights_only=True)
        coords = data["coords"]
        n = coords.shape[0]
        all_coords.append(coords)
        entry_ids.append(data.get("entry_id", os.path.basename(os.path.dirname(npt_path))))
        boundaries.append((offset, offset + n))
        offset += n
    
    if not all_coords:
        return {"coords": torch.empty(0, 3), "entry_ids": [], "num_samples": 0, "entry_boundaries": []}
    
    return {
        "coords": torch.cat(all_coords, dim=0),
        "entry_ids": entry_ids,
        "num_samples": offset,
        "entry_boundaries": boundaries,
    }


def train_vae(
    entry_dirs: list[str],
    checkpoint_dir: str,
    run_name: str = "default",
    latent_dim: int = 24,
    hidden_dim: int = 192,
    num_epochs: int = 6000,
    learning_rate: float = 5e-4,
    beta: float = 0.001,
    noise_std: float = 0.05,
    ensemble_temp: float = 2.0,
    ensemble_samples: int = 20,
    device: str = "cpu",
) -> dict:
    """Train PerResidueVAE on PDB structures and generate ensembles.

    Args:
        entry_dirs: List of directories containing normalized.pt files.
        checkpoint_dir: Directory to save artifacts.
        run_name: Run name.
        latent_dim: VAE latent dimension.
        hidden_dim: VAE hidden dimension.
        num_epochs: Training epochs.
        learning_rate: Adam learning rate.
        beta: KL weight.
        noise_std: Denoising noise std.
        ensemble_temp: Temperature for ensemble generation.
        ensemble_samples: Number of ensemble members per protein.
        device: "cpu" or "cuda".

    Returns:
        metadata dict.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load data
    input_data = load_normalized_inputs(entry_dirs)
    if input_data["num_samples"] == 0:
        raise ValueError("No training data available")

    # Build model
    model = PerResidueVAE(latent_dim=latent_dim, hidden_dim=hidden_dim)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        # Process each protein
        for entry_dir in entry_dirs:
            npt_path = os.path.join(entry_dir, "normalized.pt")
            if not os.path.exists(npt_path):
                continue
            data = torch.load(npt_path, map_location="cpu", weights_only=True)
            coords = data["coords"].to(device)
            # Denoising
            if noise_std > 0:
                noisy = coords + torch.randn_like(coords) * noise_std
            else:
                noisy = coords
            _, loss, _ = model(noisy, beta=beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / max(len(entry_dirs), 1)

        if (epoch + 1) % 1000 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs} | loss={avg_loss:.6f}")

    # Save checkpoint
    ckpt_path = os.path.join(checkpoint_dir, "vae_best.pt")
    torch.save(model.state_dict(), ckpt_path)

    # Generate ensembles for each training protein
    model.eval()
    ensemble_dir = os.path.join(checkpoint_dir, "..", "ensembles")
    os.makedirs(ensemble_dir, exist_ok=True)

    ensemble_info = []
    with torch.no_grad():
        for entry_dir in entry_dirs:
            npt_path = os.path.join(entry_dir, "normalized.pt")
            if not os.path.exists(npt_path):
                continue
            data = torch.load(npt_path, map_location="cpu", weights_only=True)
            coords = data["coords"].to(device)
            entry_id = data.get("entry_id", os.path.basename(os.path.dirname(npt_path)))

            # Generate ensemble
            ensemble = model.generate_ensemble(coords, n_samples=ensemble_samples, temp=ensemble_temp)

            # Save ensemble
            out_path = os.path.join(ensemble_dir, f"{entry_id}_ensemble.pt")
            torch.save({
                "entry_id": entry_id,
                "ensemble": ensemble.cpu(),
                "n_samples": ensemble_samples,
                "temp": ensemble_temp,
                "coords_shape": ensemble.shape,
            }, out_path)

            # Compute pairwise RMSD
            pairwise_rmsd = _pairwise_rmsd(ensemble)
            ensemble_info.append({
                "entry_id": entry_id,
                "n_residues": coords.shape[0],
                "ensemble_pairwise_rmsd": pairwise_rmsd,
                "ensemble_path": out_path,
            })

    # Build metadata
    metadata = {
        "run_name": run_name,
        "checkpoint_path": ckpt_path,
        "input_entries": input_data["entry_ids"],
        "num_samples": input_data["num_samples"],
        "num_epochs": num_epochs,
        "final_loss": avg_loss,
        "ensemble_config": {
            "n_samples": ensemble_samples,
            "temp": ensemble_temp,
        },
        "model_config": {
            "model_type": "PerResidueVAE",
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
        },
        "ensembles": ensemble_info,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    meta_path = os.path.join(checkpoint_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def _pairwise_rmsd(ensemble: torch.Tensor) -> float:
    """Mean pairwise RMSD across an ensemble (S, N, 3)."""
    S = ensemble.shape[0]
    if S < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(S):
        for j in range(i + 1, S):
            diff = ensemble[i] - ensemble[j]
            mse = (diff * diff).sum(-1).mean().item()
            total += mse ** 0.5
            count += 1
    return total / count if count > 0 else 0.0
