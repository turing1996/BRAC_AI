# WSI tiling, UNI feature extraction and global H&E preparation

This stage converts each whole-slide image into (i) level-0 tissue patches for frozen UNI feature extraction and (ii) a global 384 x 384 H&E image for the downstream ViT branch.

## 1. Install dependencies

Install an appropriate PyTorch build for the local CUDA environment, then:

```bash
python -m pip install -r requirements.txt
```

Reading `.svs` files requires the OpenSlide runtime in addition to `openslide-python`.

## 2. Tile each WSI

```bash
python 01_tile_wsi.py \
  --input /path/to/wsi \
  --output-dir /path/to/patches \
  --level 0 \
  --patch-size 256 \
  --stride 256 \
  --min-tissue-fraction 0.10
```

Default behavior:

- level-0, non-overlapping 256 x 256 patches;
- incomplete right/bottom patches are not padded;
- patches with estimated tissue fraction <10% are discarded;
- each slide receives its own output directory;
- `patches.csv` stores level-0 `x,y` coordinates and provenance.

## 3. Extract frozen UNI features

Obtain the UNI `pytorch_model.bin` checkpoint from the official MahmoodLab UNI distribution under its applicable access terms and place it at, for example:

```text
../weights/uni/pytorch_model.bin
```

Then run:

```bash
python 02_extract_uni_features.py \
  --input-dir /path/to/patches \
  --output-dir /path/to/features \
  --checkpoint ../weights/uni/pytorch_model.bin
```

The script instantiates the UNI ViT-L/16 architecture, loads the supplied pretrained checkpoint, switches the encoder to evaluation mode and extracts 1024-dimensional tile embeddings without gradient-based fine-tuning.

For each image-containing directory, the default `.pt` output contains:

- `features`: float32 tensor `[N,1024]`;
- `paths` / `filenames`: row-aligned patch identifiers;
- `coordinates`: `[N,2]` in `x,y` order when filename coordinates are available;
- encoder/checkpoint metadata.

A CSV index with `feature_index, patch_path, filename, x, y` is also written.

## 4. Generate the global H&E input

The global branch uses the valid-tissue bounding rectangle derived from `patches.csv`, resized to 384 x 384 pixels:

```bash
python 03_crop_wsi_to_384.py \
  --patches-dir /path/to/patches \
  --output-dir /path/to/global_he \
  --size 384 \
  --resize-mode stretch
```

The output `crop_summary.csv` records the source slide, bounding box, pyramid read level and final output path.
