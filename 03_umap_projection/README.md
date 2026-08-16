# UNI features -> UMAP morphology representation

This stage fits a reusable two-dimensional UMAP transformation using **TCGA training-set tile embeddings only**, applies that fixed transformation to other cohorts, and reconstructs the resulting tile-level coordinates as a two-channel spatial morphology map.

## Environment

```bash
python -m pip install -r requirements.txt
```

## 1. Fit UMAP on TCGA training embeddings

```bash
python train_umap.py /path/to/tcga_training_features \
  --recursive \
  --output-model tcga_train_umap.pkl \
  --n-neighbors 10 \
  --n-components 2 \
  --metric euclidean \
  --min-dist 0.1
```

The current clean implementation does not standardize the UNI embeddings before UMAP. The fitted object, training embedding and software/parameter metadata are saved separately.

**Important:** the exact `random_state` used for the manuscript model must be documented. The current script permits `None` to reproduce the historical non-fixed behavior, but a fixed integer is preferable for future deterministic reruns.

## 2. Transform held-out or external embeddings

```bash
python transform_umap.py /path/to/features \
  --recursive \
  --model tcga_train_umap.pkl \
  --output /path/to/umap_coordinates
```

Each output `.npz` contains a float32 array under the key `umap` with shape `[N,2]`, together with compact provenance.

## 3. Reconstruct the two-channel spatial morphology map

Use the UMAP output and the row-aligned coordinate CSV produced during UNI extraction:

```bash
python build_spatial_map.py \
  --umap /path/to/CASE.npz \
  --coordinates /path/to/CASE.csv \
  --output /path/to/CASE.npy \
  --size 384
```

The script places the two UMAP values for each valid tile at its original tile-grid location, crops to the occupied tile bounding box and resizes the two channels to 384 x 384. Missing tile locations remain zero before resizing.

The resulting array is saved as float32 with shape `(384,384,2)`, matching the downstream model input convention.

## Verification note

The source archive supplied for repository cleanup did not include the historical spatial-map construction script used before downstream training. `build_spatial_map.py` is therefore a clean implementation of the currently described rule. Before manuscript release, verify its grid placement, missing-tile handling and resizing behavior against the exact procedure that produced the morphology maps used by the reported model.
