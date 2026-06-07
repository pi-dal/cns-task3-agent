"""Evaluation benchmarks for protein conformational ensemble generation.

Provides:
- PDB dataset loading and preprocessing
- Train/validation split
- Reconstruction and ensemble diversity metrics
- Benchmark runner suitable for autoresearch
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from .models import PerResidueVAE as ProteinVAE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ONE_LETTER = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E",
    "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I",
    "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S",
    "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def parse_pdb_ca(pdb_path: str) -> tuple[list[list[float]], list[str], int]:
    """Extract CA coordinates and residue types from a PDB file.

    Returns:
        coords: list of [x, y, z] lists.
        residues: list of one-letter residue codes.
        seq_len: number of residues.
    """
    coords = []
    residues = []
    seen_residues = set()
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    res_id = (line[21:26].strip(), line[17:20].strip())
                    if res_id in seen_residues:
                        continue
                    seen_residues.add(res_id)
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    res_name = line[17:20].strip()
                    coords.append([x, y, z])
                    residues.append(ONE_LETTER.get(res_name, "X"))
                except (ValueError, IndexError):
                    continue
    return coords, residues, len(coords)


def normalize_coords(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Center and scale coordinates to have zero mean and unit variance.

    Returns:
        normalized: (N, 3) normalized coordinates.
        center: (3,) centroid.
        scale: scalar scaling factor (std).
    """
    center = coords.mean(dim=0)
    centered = coords - center
    scale = centered.std()
    if scale < 1e-8:
        scale = 1.0
    normalized = centered / scale
    return normalized, center, scale.item()


def rmsd(coords_a: torch.Tensor, coords_b: torch.Tensor) -> float:
    """Root Mean Square Deviation between two coordinate sets (N, 3)."""
    return torch.sqrt(F.mse_loss(coords_a, coords_b)).item()


def pairwise_rmsd(ensemble: torch.Tensor) -> float:
    """Mean pairwise RMSD across an ensemble (S, N, 3) -> float."""
    S = ensemble.shape[0]
    if S < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(S):
        for j in range(i + 1, S):
            total += rmsd(ensemble[i], ensemble[j])
            count += 1
    return total / count if count > 0 else 0.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PDBDataset(Dataset):
    """Dataset of PDB structures loaded as CA coordinate tensors."""

    def __init__(self, pdb_paths: list[str]):
        self.entries = []
        for path in pdb_paths:
            coords_list, residues, seq_len = parse_pdb_ca(path)
            if seq_len < 10:
                continue
            coords = torch.tensor(coords_list, dtype=torch.float32)
            self.entries.append({
                "coords": coords,
                "seq_len": seq_len,
                "entry_id": Path(path).stem,
            })

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]


def collate_pdb(batch: list[dict]) -> dict:
    """Collate function: returns a list of coords tensors (variable length)."""
    return {
        "coords": [b["coords"] for b in batch],
        "entry_ids": [b["entry_id"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_vae(
    model: ProteinVAE,
    train_data: list[dict],
    val_data: list[dict],
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    beta: float = 1.0,
    beta_anneal_epochs: int = 0,
    noise_std: float = 0.0,
    device: str = "cpu",
    verbose: bool = False,
) -> dict:
    """Train a ProteinVAE on a list of protein structures.

    Args:
        model: ProteinVAE instance.
        train_data: list of {"coords": Tensor(N,3), "entry_id": str}.
        val_data: list for validation.
        num_epochs: training epochs.
        learning_rate: Adam learning rate.
        beta: KL weight.
        beta_anneal_epochs: If >0, linearly anneal beta from 0 to beta.
        noise_std: Gaussian noise std for denoising VAE (0 = disabled).
        device: "cpu" or "cuda".
        verbose: print progress.

    Returns:
        dict with training history and final metrics.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    history = {"train_loss": [], "val_loss": [], "val_recon": [], "val_kl": [], "beta_used": []}

    for epoch in range(num_epochs):
        # KL annealing
        if beta_anneal_epochs > 0 and epoch < beta_anneal_epochs:
            current_beta = beta * (epoch + 1) / beta_anneal_epochs
        else:
            current_beta = beta

        model.train()
        epoch_loss = 0.0
        for entry in train_data:
            coords = entry["coords"].to(device)
            if noise_std > 0:
                noisy = coords + torch.randn_like(coords) * noise_std
                _, loss, _ = model(noisy, beta=current_beta)
            else:
                _, loss, _ = model(coords, beta=current_beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / max(len(train_data), 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0
        with torch.no_grad():
            for entry in val_data:
                coords = entry["coords"].to(device)
                recon, total_loss, kl = model(coords, beta=current_beta)
                recon_mse = F.mse_loss(recon, coords).item()
                val_loss += total_loss.item()
                val_recon += recon_mse
                val_kl += kl.item()
        n_val = max(len(val_data), 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss / n_val)
        history["val_recon"].append(val_recon / n_val)
        history["beta_used"].append(current_beta)
        history["val_kl"].append(val_kl / n_val)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs} | train_loss={avg_train_loss:.6f} | val_loss={val_loss/n_val:.6f}")

    return history


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

BENCHMARK_PDBS = [
    "1ubq",  # Ubiquitin (76 residues)
    "4hhb",  # Hemoglobin (574 residues)
    "2pah",  # Phenylalanine hydroxylase (651 residues)
    "1crn",  # Crambin (46 residues)
    "1bba",  # BBA fold (36 residues)
    "4ins",  # Insulin (102 residues)
    "1hoe",  # Engrailed homeodomain (74 residues)
    "2gb1",  # B1 domain of protein G (56 residues)
    "3u7y",  # Heat shock protein (768 residues)
    "4m3e",  # Engineered beta-lactamase (193 residues)
    "1pko",  # Protein kinase (123 residues)
    "1a6m",  # DNA-binding protein (151 residues)
    "2bh8",  # Beta-sandwich (171 residues)
    "2hdm",  # Designed ankyrin repeat (74 residues)
]

# Held-out proteins for validation (diverse, unseen folds)
VAL_ENTRIES = {"1crn", "1bba", "1hoe", "2gb1"}


def run_benchmark(
    pdb_dir: str = "/tmp/cns_pdb_data",
    latent_dim: int = 8,
    hidden_dim: int = 64,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    beta: float = 0.1,
    beta_anneal_epochs: int = 0,
    noise_std: float = 0.0,
    ensemble_samples: int = 20,
    ensemble_temp: float = 1.0,
    diversity_weight: float = 0.001,
    device: str = "cpu",
    verbose: bool = False,
) -> dict:
    """Run the full benchmark: load data, train VAE, evaluate.

    Coordinates are normalized (centered + scaled) before training;
    metrics are reported in original (unnormalized) scale.

    Returns a dict with all metrics suitable for autoresearch.
    """
    start_time = time.time()

    # Load all PDBs and normalize each protein
    all_entries = []
    for eid in BENCHMARK_PDBS:
        path = os.path.join(pdb_dir, f"{eid}.pdb")
        if not os.path.exists(path):
            if verbose:
                print(f"  Skipping {eid}: file not found")
            continue
        coords_list, residues, seq_len = parse_pdb_ca(path)
        if seq_len < 10:
            continue
        raw_coords = torch.tensor(coords_list, dtype=torch.float32)
        norm_coords, center, scale = normalize_coords(raw_coords)
        all_entries.append({
            "coords": norm_coords,       # normalized for training
            "raw_coords": raw_coords,     # original for metrics
            "center": center,
            "scale": scale,
            "seq_len": seq_len,
            "entry_id": eid,
        })
        if verbose:
            print(f"  {eid}: seq_len={seq_len}, scale={scale:.2f}")

    if len(all_entries) < 2:
        raise ValueError(f"Need at least 2 PDB entries, got {len(all_entries)}")

    # Split train/val
    train_data = [e for e in all_entries if e["entry_id"] not in VAL_ENTRIES]
    val_data = [e for e in all_entries if e["entry_id"] in VAL_ENTRIES]

    if not val_data:
        # Fallback: use last entry as val
        train_data = all_entries[:-1]
        val_data = [all_entries[-1]]

    if verbose:
        print(f"  Train: {[e['entry_id'] for e in train_data]}")
        print(f"  Val:   {[e['entry_id'] for e in val_data]}")

    # Create and train model
    model = ProteinVAE(latent_dim=latent_dim, hidden_dim=hidden_dim)
    history = train_vae(
        model, train_data, val_data,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        beta=beta,
        beta_anneal_epochs=beta_anneal_epochs,
        noise_std=noise_std,
        device=device,
        verbose=verbose,
    )

    # Evaluate on validation set (in original coordinate scale)
    model.eval()
    val_recon_mse = 0.0
    val_diversity = 0.0
    with torch.no_grad():
        for entry in val_data:
            norm_coords = entry["coords"].to(device)
            raw_coords = entry["raw_coords"]
            center = entry["center"].to(device)
            scale = entry["scale"]

            recon_norm, _, _ = model(norm_coords, beta=beta)

            # Unnormalize reconstruction back to original coordinate space
            recon_raw = recon_norm * scale + center

            # MSE in original coordinate space
            val_recon_mse += F.mse_loss(recon_raw, raw_coords.to(device)).item()

            # Ensemble diversity (in original space)
            ensemble_norm = model.generate_ensemble(norm_coords, n_samples=ensemble_samples, temp=ensemble_temp)
            ensemble_raw = ensemble_norm * scale + center
            val_diversity += pairwise_rmsd(ensemble_raw)

    val_recon_mse /= max(len(val_data), 1)
    val_diversity /= max(len(val_data), 1)

    # Combined metric: lower is better (good recon + high diversity)
    # scale diversity to be comparable with MSE
    combined = val_recon_mse - diversity_weight * val_diversity

    elapsed = time.time() - start_time

    # Final validation loss (last epoch, normalized space)
    final_val_loss = history["val_loss"][-1] if history["val_loss"] else float("inf")

    result = {
        "combined_score": combined,
        "val_recon_mse": val_recon_mse,
        "val_diversity": val_diversity,
        "final_val_loss": final_val_loss,
        "train_loss_last": history["train_loss"][-1] if history["train_loss"] else 0,
        "num_train": len(train_data),
        "num_val": len(val_data),
        "elapsed_s": elapsed,
        "config": {
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "beta": beta,
            "beta_anneal_epochs": beta_anneal_epochs,
            "noise_std": noise_std,
            "ensemble_samples": ensemble_samples,
            "ensemble_temp": ensemble_temp,
            "diversity_weight": diversity_weight,
        },
    }

    return result


def print_metrics(result: dict):
    """Pretty-print benchmark results."""
    print(f"  Combined score:      {result['combined_score']:.6f}  (lower is better)")
    print(f"  Val recon MSE:       {result['val_recon_mse']:.6f}")
    print(f"  Val diversity RMSD:  {result['val_diversity']:.6f}")
    print(f"  Final val loss:      {result['final_val_loss']:.6f}")
    print(f"  Elapsed:             {result['elapsed_s']:.1f}s")
    print(f"  Config:              {result['config']}")


if __name__ == "__main__":
    result = run_benchmark(verbose=True)
    print_metrics(result)
