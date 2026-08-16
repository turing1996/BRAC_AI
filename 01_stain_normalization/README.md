# CycleGAN stain/style normalization

This module trains and applies the H&E style normalizer used to reduce acquisition/staining differences between an external cohort and the TCGA reference domain.

## Domain convention

- Domain A: external-cohort H&E patches
- Domain B: TCGA training H&E patches
- `A2B`: external -> TCGA-style normalization
- `B2A`: reverse mapping used during CycleGAN training

## Environment

```bash
python -m pip install -r requirements.txt
```

## Dataset layout

```text
dataset_root/
├── train/
│   ├── A/    # external cohort
│   └── B/    # TCGA reference patches
└── test/
    ├── A/
    └── B/
```

## Training

The current implementation defaults to 400 epochs, batch size 2, 400 x 400 crops, Adam with learning rate 2e-4 and betas (0.5, 0.999), identity-loss weight 5 and cycle-consistency weight 10.

```bash
python train.py \
  --data-root /path/to/dataset_root \
  --output-dir runs/CBCGA2TCGA_400 \
  --epochs 400 \
  --batch-size 2 \
  --image-size 400 \
  --lr 0.0002 \
  --decay-start-epoch 5 \
  --seed 42
```

## Inference

External cohort -> TCGA style:

```bash
python infer.py \
  --direction A2B \
  --input /path/to/external_patches \
  --output /path/to/normalized_patches \
  --weights /path/to/study_netG_A2B.pth \
  --recursive
```

The exact study generator checkpoint is not present in the uploaded source archive and should be added to the manuscript release, or archived separately, before claiming that trained weights are publicly available.
