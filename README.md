# BRAC-AI computational pathology pipeline

This repository contains the code used to implement the H&E-based prognostic modelling workflow in the BRAC study. The repository is organized as four explicit stages so that preprocessing, frozen foundation-model feature extraction, training-set-only UMAP fitting, and downstream survival modelling can be inspected independently.

## Workflow

```text
Whole-slide H&E image
  |
  |-- tissue tiling (256 x 256 px)
  |-- optional external-cohort stain/style normalization
  v
Frozen UNI tile encoder -> 1024-D tile features
  |
  |-- UMAP fitted on TCGA training tile features only
  v
2-D tile morphology coordinates
  |
  |-- place tile coordinates back at their WSI locations
  |-- construct 2-channel spatial morphology map
  v
384 x 384 x 2 morphology map ----------------------+
                                                       |
Cropped global H&E image -> 384 x 384 RGB -> ViT-B/16 |
                                                       v
                                     cross-attention fusion
                                          /          \
                                     OS Cox head   DFS Cox head
```

## Repository layout

```text
BRAC_AI/
├── 01_stain_normalization/       # CycleGAN style normalization
├── 02_uni_feature_extraction/    # WSI tiling, UNI features, global H&E crop
├── 03_umap_projection/           # training-set UMAP + spatial morphology map
├── 04_survival_model/            # dual-input prognostic model
├── splits/                       # optional de-identified split manifests
├── weights/                      # local large-file placeholders; not committed
└── .gitignore
```

## Included functions
- WSI tissue tiling at level 0 and patch-coordinate manifests.
- CycleGAN training and inference code for stain/style normalization.
- Frozen UNI feature-extraction code.
- UMAP fitting and transformation code.
- Spatial reconstruction of the two-channel morphology map.
- Global H&E crop generation.
- ViT-B/16 + morphology-MLP + cross-attention survival architecture.
- Multi-task OS/DFS Cox training with patient-level risk aggregation and Efron ties.
- Held-out TCGA validation checkpoint selection by joint OS + DFS Cox loss.
- External-cohort inference and Harrell C-index reporting.
- Outcome-specific Grad-CAM code for the H&E ViT branch.
- Explicit survival-model configuration with the training hyperparameters used by the current implementation.

## Installation

Python 3.10 is recommended. Because PyTorch installation depends on the local CUDA environment, install an appropriate PyTorch build first, then install the dependencies for each stage as needed.

```bash
python -m pip install -r 01_stain_normalization/requirements.txt
python -m pip install -r 02_uni_feature_extraction/requirements.txt
python -m pip install -r 03_umap_projection/requirements.txt
python -m pip install -e 04_survival_model
```

## Steps

### 1. Tile WSIs

```bash
python 02_uni_feature_extraction/01_tile_wsi.py \
  --input /path/to/wsi \
  --output-dir work/patches \
  --level 0 \
  --patch-size 256 \
  --stride 256
```

### 2. Apply the external-to-TCGA style normalizer when required

```bash
python 01_stain_normalization/infer.py \
  --direction A2B \
  --input work/patches_external \
  --output work/patches_external_normalized \
  --weights /path/to/study_netG_A2B.pth \
  --recursive
```

For CycleGAN training details, see `01_stain_normalization/README.md`.

### 3. Extract frozen UNI features

The UNI checkpoint must first be obtained from the official MahmoodLab distribution under its applicable access terms.

```bash
python 02_uni_feature_extraction/02_extract_uni_features.py \
  --input-dir work/patches \
  --output-dir work/uni_features \
  --checkpoint weights/uni/pytorch_model.bin
```

### 4. Fit UMAP using TCGA training features only

```bash
python 03_umap_projection/train_umap.py \
  /path/to/tcga_training_features \
  --recursive \
  --output-model work/tcga_train_umap.pkl \
  --n-neighbors 10 \
  --n-components 2 \
  --metric euclidean \
  --min-dist 0.1
```

Apply the fitted training UMAP to held-out/external features:

```bash
python 03_umap_projection/transform_umap.py \
  /path/to/features \
  --recursive \
  --model work/tcga_train_umap.pkl \
  --output work/umap_coordinates
```

### 5. Reconstruct the two-channel spatial morphology map

```bash
python 03_umap_projection/build_spatial_map.py \
  --umap work/umap_coordinates/CASE.npz \
  --coordinates work/uni_features/CASE.csv \
  --output work/morphology_maps/CASE.npy \
  --size 384
```

### 6. Generate the global 384 x 384 H&E image

```bash
python 02_uni_feature_extraction/03_crop_wsi_to_384.py \
  --patches-dir work/patches \
  --output-dir work/global_he
```

### 7. Train the downstream OS/DFS model

Arrange H&E images, morphology maps and survival files as described in `docs/DATA_LAYOUT.md`, then:

```bash
cd 04_survival_model
CUDA_VISIBLE_DEVICES=0 python -m brac_ai.train --config config.yaml
```

The best checkpoint is selected using the minimum held-out TCGA joint Cox loss (`OS Cox loss + DFS Cox loss`).

### 8. External evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python -m brac_ai.evaluate \
  --config config.yaml \
  --checkpoint outputs/vit_morphology_mlp_cross_attention/best.pt \
  --cohort external
```

### 9. Grad-CAM visualization

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m brac_ai.gradcam \
  --config config.yaml \
  --checkpoint outputs/vit_morphology_mlp_cross_attention/best.pt \
  --cohort external \
  --endpoint both \
  --output outputs/vit_morphology_mlp_cross_attention/gradcam_cbcga
```