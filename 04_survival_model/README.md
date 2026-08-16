# BRAC-AI dual-input OS/DFS survival model

This package implements the downstream prognostic model used after global H&E preparation and tile-derived morphology-map construction.

## Architecture

```text
Global H&E image (384 x 384 RGB)
  -> ImageNet-pretrained ViT-B/16
  -> H&E token sequence / CLS query

2-channel morphology map (384 x 384 x 2)
  -> non-overlapping 16 x 16 patches
  -> patch-wise MLP projection
  -> 2 residual MLP blocks
  -> morphology tokens

H&E CLS query
  -> 12-head cross-attention over morphology tokens
  -> fused representation
       |-> OS Cox head
       `-> DFS Cox head
```

UNI is **not** part of this trainable package. UNI is used upstream as a frozen tile-level feature extractor in `../02_uni_feature_extraction/`.

## Data layout

The default configuration expects:

```text
data/
├── TCGA/
│   ├── train/
│   │   ├── HE/
│   │   ├── sur_time/
│   │   └── umap/
│   └── validation/
│       ├── HE/
│       ├── sur_time/
│       └── umap/
└── CBCGA/
    ├── HE/
    ├── sur_time/
    └── umap/
```

See `../docs/DATA_LAYOUT.md` for naming conventions.

## Installation

```bash
conda create -n brac_ai python=3.10 -y
conda activate brac_ai
pip install -e .
```

Place the ImageNet-pretrained ViT-B/16 checkpoint at:

```text
pretrained/B_16_imagenet1k.pth
```

A helper downloader is provided at `../scripts/download_vit_backbone.py`.

## Trainable and frozen components

At the start of training, the H&E ViT is initialized from ImageNet-pretrained weights. The implementation freezes the ViT patch embedding and all but the final two transformer blocks. The final two ViT blocks are optimized jointly with:

- the morphology patch MLP and residual MLP blocks;
- the cross-attention fusion module;
- the OS Cox head;
- the DFS Cox head.

The checked-in `config.yaml` is the machine-readable source of truth for training parameters.

## Current training configuration

Key values in `config.yaml`:

```text
maximum epochs                   30
training batch size               4
evaluation batch size             2
seed                             42
Cox ties                      Efron
patient slide-risk aggregation  mean
full-risk updates / epoch         1
H&E backbone LR                1e-5
morphology LR                  1e-5
fusion / Cox-head LR           1e-5
weight decay                   1e-4
gradient clipping               1.0
trainable H&E ViT blocks          2 (final two)
LR scheduler                   ReduceLROnPlateau
LR scheduler patience             4
LR reduction factor             0.5
early-stopping patience           8
AMP                              on (CUDA)
deterministic algorithms         on
```

## Training

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m brac_ai.train --config config.yaml
```

Training uses patient-level complete-risk-set Cox optimization with Efron ties:

```text
joint Cox loss = OS Cox loss + DFS Cox loss
```

The best checkpoint is selected using the **minimum joint Cox loss on the held-out TCGA validation set**.

Outputs include the best checkpoint, epoch history, training curves, sample-level predictions, patient-level predictions and held-out validation metrics.

## External evaluation

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m brac_ai.evaluate \
  --config config.yaml \
  --checkpoint outputs/vit_morphology_mlp_cross_attention/best.pt \
  --cohort external
```

The bundled evaluator writes OS/DFS risks and Harrell C-indices. Manuscript-specific statistical analyses such as time-dependent AUC and calibration should be released with the exact scripts used to generate the reported tables/figures if those scripts are separate from this training package.

## Grad-CAM

Endpoint-specific H&E Grad-CAM maps can be generated with:

```bash
python -m brac_ai.gradcam \
  --config config.yaml \
  --checkpoint outputs/vit_morphology_mlp_cross_attention/best.pt \
  --cohort external \
  --endpoint both
```

These visualizations are attribution aids for hypothesis generation and should not be interpreted as direct mechanistic or biological evidence.
