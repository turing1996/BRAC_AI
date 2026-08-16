# Reproducibility and release checklist

## Included in this code archive

- preprocessing and WSI tiling code
- CycleGAN architecture, training and inference code
- UNI feature-extraction code
- UMAP fitting/transform code
- spatial morphology-map construction code
- global H&E crop code
- downstream model architecture
- survival-model training and inference code
- explicit downstream hyperparameter YAML
- Harrell C-index implementation
- endpoint-specific Grad-CAM utility

## Artifacts that must be supplied separately

The uploaded source archive did not contain the following large/access-controlled study artifacts, so this repository does **not** claim that they are currently bundled:

1. the UNI pretrained checkpoint;
2. the study-specific CycleGAN generator checkpoint;
3. the final trained BRAC-AI survival checkpoint used for manuscript results;
4. the large fitted TCGA-training UMAP object;
5. study WSIs and clinical outcomes;
6. exact de-identified train/validation split manifests, if these are permitted to be shared.

Before using wording such as “the complete computational pipeline and trained weights are publicly available,” make these artifacts available where licensing, data governance and file-size constraints permit, or state clearly how they can be obtained.

## Recommended manuscript release steps

1. Create a GitHub release corresponding exactly to the revision, for example `v1.0-manuscript`.
2. Record the Git commit SHA in the Code Availability statement or Supplementary Methods.
3. Place large author-generated checkpoints in a persistent archive (for example, a DOI-backed repository) and link that archive from this README.
4. Do not redistribute the gated UNI weight file unless its license/access terms explicitly permit redistribution; instead document the official model identity and acquisition route.
5. Add the exact CycleGAN checkpoint filename and downstream BRAC-AI checkpoint filename used to generate the manuscript results.
6. Add/check exact TCGA train/validation split manifests when permissible.
7. Record the UMAP `random_state` actually used for the manuscript model. If it was `None`, state this explicitly; do not silently replace it with a new seed.
8. Cross-check `03_umap_projection/build_spatial_map.py` against the exact historical script/rules used to generate the morphology maps supplied to the trained manuscript model. The uploaded source archive did not contain that historical spatial-map script, so the included implementation is a clean reconstruction of the currently described method (tile UMAP coordinates placed back on the WSI grid, cropped to the occupied grid and resized to 384 x 384). Do not claim bitwise reproduction until this equivalence has been verified.
9. Run the syntax and smoke checks described below from a clean environment.

## Suggested pre-release checks

```bash
python -m compileall 01_stain_normalization 02_uni_feature_extraction 03_umap_projection 04_survival_model/src
pytest -q 03_umap_projection/tests
```

After adding the required weights and a small de-identified/synthetic example dataset, also run one full inference pass through all four stages.
