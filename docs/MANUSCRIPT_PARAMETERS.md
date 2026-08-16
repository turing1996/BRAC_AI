# Current manuscript implementation parameters

This document summarizes parameters encoded in the checked-in scripts and `04_survival_model/config.yaml`. The YAML file remains the machine-readable source of truth for downstream-model training.

## WSI processing

- Pyramid level: 0
- Tile size: 256 x 256 pixels
- Tile stride: 256 pixels
- Minimum estimated tissue fraction: 0.10
- White-background RGB threshold: 220
- Dark-border mean-RGB threshold: 15
- Saved tile format: JPEG, quality 95 by default
- Global H&E image: valid-tissue bounding box resized to 384 x 384 pixels

## Stain/style normalization

Current CycleGAN script defaults:

- Training epochs: 400
- Batch size: 2
- Training image size: 400 x 400 pixels
- Initial learning rate: 2e-4
- Adam betas: 0.5, 0.999
- Identity-loss multiplier: 5
- Cycle-consistency multiplier: 10
- Default random seed: 42
- External cohort is domain A and TCGA-reference patches are domain B; `A2B` therefore denotes external-to-TCGA normalization.

The exact study generator checkpoint should be identified in the public release before submission.

## UNI feature extraction

- Model: UNI (ViT-L/16 pathology foundation model)
- Input: 224 x 224 transformed tile image
- Output feature dimension: 1024
- Usage: frozen feature extractor; no UNI parameter is optimized by the BRAC survival objective
- Checkpoint: obtained separately from the official MahmoodLab UNI distribution

## UMAP

- Fitted using TCGA training tile embeddings only
- `n_neighbors`: 10
- `n_components`: 2
- `metric`: Euclidean
- `min_dist`: 0.1
- No feature standardization in the current clean implementation

If strict deterministic reproduction of the UMAP fit is required, use and report a fixed integer `random_state`. The historical clean script permits `None`; the manuscript release should document the exact setting used for the reported model.

## Downstream prognostic model

Architecture:

- Global H&E input: 384 x 384 RGB
- H&E backbone: ImageNet-pretrained ViT-B/16
- Morphology input: 384 x 384 x 2 spatial UMAP map
- Morphology patch size: 16
- Morphology hidden dimension: 1024
- Morphology residual MLP depth: 2
- Morphology dropout: 0.10
- Cross-attention heads: 12
- Cross-attention dropout: 0.15
- Cross-attention gamma initialization: 0.10
- Cox-head hidden dimension: 128
- Cox-head dropout: 0.30
- Endpoints: OS and DFS

Training:

- Seed: 42
- Epochs: 30 maximum
- Training batch size: 4
- Evaluation batch size: 2
- Cox ties: Efron
- Patient-level slide-risk aggregation: mean
- Full-risk-set updates per epoch: 1
- H&E backbone learning rate: 1e-5
- Morphology branch learning rate: 1e-5
- Fusion/Cox-head learning rate: 1e-5
- Weight decay: 1e-4
- Gradient clipping: 1.0
- H&E backbone: only the final two transformer blocks are trainable; the patch embedding and earlier blocks remain frozen
- Learning-rate scheduler: ReduceLROnPlateau, patience 4, factor 0.5
- Early stopping patience: 8 epochs
- AMP: enabled on CUDA
- Deterministic algorithms: enabled
- Checkpoint selection: minimum held-out TCGA `OS Cox loss + DFS Cox loss`

## Interpretability utility

`04_survival_model/src/brac_ai/gradcam.py` generates endpoint-specific Grad-CAM maps from an H&E ViT block by backpropagating the OS or DFS risk output. Such maps are visualization/attribution aids and should not be described as mechanistic biological evidence.
