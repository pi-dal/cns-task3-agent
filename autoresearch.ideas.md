# Autoresearch Ideas - Protein Conformational Ensemble Generation

## Status: Converged (30 experiments)
**Best: combined_score=0.077 (-99.2% from baseline 9.17)**

## Key successes
- ✅ **Per-residue VAE** (instead of global VAE)
- ✅ **LayerNorm > BatchNorm** — biggest single improvement (-43%)
- ✅ **Denoising VAE** (noise_std=0.05)
- ✅ **Temperature scaling** (temp=2.0, 3x diversity boost)
- ✅ **Near-zero KL** (beta=0.001)
- ✅ **Expanded dataset** (6 → 14 PDBs)
- ✅ **Long training** (6000 epochs with LayerNorm)
- ✅ **Bigger model** (latent=24, hidden=192)

## Exhausted (no improvement)
- ❌ Deeper networks (3+ layers overfit)
- ❌ Dropout (hurts recon with tiny dataset)
- ❌ KL annealing
- ❌ Cosine annealing LR
- ❌ AdamW weight decay (better diversity, worse combined)
- ❌ Huber loss (eval mismatch)
- ❌ Sinusoidal positional encoding (too many params)
- ❌ Gradient clipping (no benefit for simple MLP)
- ❌ Noise annealing (constant noise works better)
- ❌ Rotation data augmentation (per-residue model already rotation-robust)
- ❌ Extreme epochs (10000 > 6000 showed stochastic regression)

## Missed opportunities (would require major rework)
- **Transformer encoder** — self-attention for non-local interactions
- **VAE with graph network** — GNN captures residue connectivity
- **Flow matching / diffusion** — state-of-the-art generative approach
- **BioEmu benchmark** — proper comparison with published methods
- **Ensemble quality metrics** — TM-score, clash score, Ramachandran
- **Full CNS pipeline** — autonomous literature + data + training loop
