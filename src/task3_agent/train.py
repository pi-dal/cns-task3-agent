"""Minimal baseline training for CNS Task 3.

Consumes normalized.pt artifacts and produces a checkpoint
with associated metadata. Not a real protein model — this is
a scaffold for the training path + checkpoint contract.
"""

import json
import os
import torch
import torch.nn as nn
from datetime import datetime, timezone


class MLPBaseline(nn.Module):
    """Minimal MLP for coordinate reconstruction."""
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def forward(self, x):
        return self.net(x)


def load_normalized_inputs(entry_dirs: list[str]) -> dict:
    """Load normalized.pt files from entry directories.
    
    Returns dict with keys:
        coords: torch.Tensor (N, 3) — concatenated coords from all entries
        entry_ids: list[str]
        num_samples: int — total coordinate rows
        entry_boundaries: list[tuple[int, int]] — (start, end) slices per entry
    """
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


def train_baseline(
    input_data: dict,
    checkpoint_dir: str,
    hidden_dim: int = 64,
    num_epochs: int = 200,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
) -> dict:
    """Train the MLP baseline and save checkpoint + metadata.
    
    Args:
        input_data: Dict from load_normalized_inputs.
        checkpoint_dir: Directory to save artifacts.
        hidden_dim: MLP hidden dimension.
        num_epochs: Training epochs.
        learning_rate: Learning rate.
        batch_size: Batch size.
    
    Returns:
        metadata dict.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    coords = input_data["coords"]
    num_samples = input_data["num_samples"]
    
    if num_samples == 0:
        raise ValueError("No training data available")
    
    model = MLPBaseline(input_dim=3, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    dataset = torch.utils.data.TensorDataset(coords, coords)
    loader = torch.utils.data.DataLoader(dataset, batch_size=min(batch_size, num_samples), shuffle=True)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    # Save checkpoint
    ckpt_path = os.path.join(checkpoint_dir, "baseline.pt")
    torch.save(model.state_dict(), ckpt_path)
    
    # Build metadata
    metadata = {
        "run_name": os.path.basename(os.path.dirname(checkpoint_dir)),
        "checkpoint_path": ckpt_path,
        "input_entries": input_data["entry_ids"],
        "num_samples": num_samples,
        "feature_schema": "coords: float32 (N, 3)",
        "training_config": {
            "model_type": "mlp_baseline",
            "hidden_dim": hidden_dim,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        },
        "final_loss": total_loss / len(loader),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    
    meta_path = os.path.join(checkpoint_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata
