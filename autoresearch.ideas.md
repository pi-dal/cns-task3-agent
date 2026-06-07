# Autoresearch Ideas - Protein Conformational Ensemble Generation

## Exhausted (tried, converged)

### Architecture
- ~~**Deeper network**: 3-layer model overfits on current dataset size~~ ❌
- ~~**Dropout layers**: Dropout=0.1-0.3 hurts reconstruction, no benefit~~ ❌
- ~~**Sinusoidal positional encoding**: Too many params for dataset, no improvement~~ ❌
- ~~**Bigger latent/hidden**: Sweet spot at 24/192, 32/256 was worse~~ ✅

### Training
- ~~**Denoising VAE**: noise_std=0.05 works well~~ ✅
- ~~**KL annealing**: Added but didn't help with near-zero beta~~ ❌
- ~~**Cosine annealing**: Worse than constant LR for this dataset~~ ❌
- ~~**AdamW with weight_decay**: Higher diversity but worse combined score~~ ❌
- ~~**Huber loss**: Mismatch with MSE evaluation metric~~ ❌

### Ensemble Generation
- ~~**Temperature scaling (temp=2.0)**: 3x diversity boost, minimal recon loss~~ ✅

### Data
- ~~**More diverse PDB structures**: Expanded from 6 to 14 PDBs — reduced variance~~ ✅

## Promising directions not yet explored

### Architecture
- **Transformer encoder**: Self-attention could capture non-local residue interactions
- **LayerNorm > BatchNorm**: Better for small batch sizes, less batch dependency
- **Residual connections**: Enable deeper networks without overfitting

### Training
- **Gradient clipping**: Prevent loss spikes with denoising
- **Learning rate warmup**: Gradual LR increase helps stabilize deep VAEs

### Ensemble Generation
- **Truncation sampling**: Sample within 2σ for quality-diversity tradeoff
- **Structure-based sampling**: Kabsch alignment before RMSD
- **Tempered ensemble**: Mix temperatures within ensemble (some cold, some hot)

### Evaluation
- **Kabsch alignment**: Rotation-invariant RMSD comparison
- **Per-residue RMSD**: Residue-level error analysis
- **Physical plausibility**: Clash score, Ramachandran analysis
- **TM-score**: Structure similarity metric

### Data
- **Multi-chain handling**: Process each chain separately
- **Data augmentation**: Random rotations during training
- **CATH/SCOP fold classification**: Ensure fold-level train/test split

### Integration with CNS Task3
- **Auto-download PDBs**: Autonomous data acquisition from RCSB
- **Literature stage**: Auto-research BioEmu, STARLING, EPO papers
- **Agent decision log**: Record why each architecture/model choice was made
