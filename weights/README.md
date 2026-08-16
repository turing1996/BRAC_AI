# Model artifact locations

Large model artifacts are ignored by Git by default.

Recommended local layout:

```text
weights/
├── uni/
│   └── pytorch_model.bin
├── style_normalizer/
│   └── STUDY_netG_A2B.pth
└── survival_model/
    └── STUDY_best.pt
```

The downstream ViT-B/16 checkpoint is currently expected by `04_survival_model/config.yaml` at:

```text
04_survival_model/pretrained/B_16_imagenet1k.pth
```

For the manuscript release, document the exact filenames and immutable archive identifiers for author-generated study checkpoints.
