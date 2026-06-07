# Autoresearch Ideas - Protein Conformational Ensemble Generation

## Promising directions not yet explored

### Architecture
- **Deeper network**: Add more hidden layers (3-4 layers) for better capacity
- **Dropout layers**: Add dropout (p=0.1-0.3) for regularization with near-zero KL
- **Sinusoidal positional encoding**: Replace linear [-1,1] with sin/cos encoding
- **Transformer encoder**: Use self-attention instead of per-residue MLP for better global context
- **Residual connections**: Add skip connections for deeper networks
- **LayerNorm vs BatchNorm**: Try LayerNorm for small batch sizes

### Training
- **Denoising VAE**: Add Gaussian noise to input coordinates during training — encourages smooth latent space AND diverse ensembles
- **Learning rate scheduling**: Cosine annealing or ReduceLROnPlateau
- **AdamW with weight decay**: Better regularization than Adam
- **Gradient clipping**: Prevent loss spikes
- **Reconstruction loss scaling**: Use RMSD instead of MSE, or Huber loss for robustness

### Ensemble Generation
- **Temperature scaling**: Multiply latent std by temperature > 1 for more diverse ensembles
- **Truncation sampling**: Sample from truncated Normal (within 2 std) for quality-diversity tradeoff
- **Structure-based sampling**: Use Kabsch alignment before RMSD computation for rotation-invariant evaluation
- **Multiple ensemble sizes**: Evaluate at different n_samples (10, 50, 100)

### Data
- **More diverse PDB structures**: Add proteins with different folds (alpha, beta, alpha/beta)
- **Data augmentation**: Random rotations, mirror transforms
- **Multi-chain handling**: Process each chain separately

### Evaluation
- **Kabsch alignment**: Align structures before RMSD computation for proper structural comparison
- **Per-residue RMSD**: Evaluate per-residue errors instead of global
- **TM-score**: Add TM-score metric for structural similarity
- **Clash score**: Evaluate physical plausibility of generated conformations
- **Ramachandran analysis**: Check backbone dihedral angle distributions
