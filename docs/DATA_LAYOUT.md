# Data layout

Study data are not included in the repository. The downstream model expects the following structure:

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

The paths are configured in `04_survival_model/config.yaml`. The historical local folder named `TCGA/test` represents the held-out TCGA **validation** set; for manuscript-facing documentation we refer to it consistently as validation.

## File correspondence

For one slide/sample, H&E and survival-time files share the same stem:

```text
HE/1_0_TCGA-XX-XXXX-..._0_0.png
sur_time/1_0_TCGA-XX-XXXX-..._0_0.csv
```

The leading two digits are interpreted as the OS-event and DFS-event indicators. The survival CSV contains at least two numeric values in the order `OS time, DFS time`.

The morphology map may use the full event-prefixed stem or the same stem with the leading `<OS>_<DFS>_` removed:

```text
umap/1_0_TCGA-XX-XXXX-..._0_0.npy
# or
umap/TCGA-XX-XXXX-..._0_0.npy
```

Each morphology file must be a finite float array with shape `(384,384,2)` or `(2,384,384)`.

## Patient-level separation

The training code checks that the TCGA training and held-out validation folders contain no overlapping TCGA patient identifiers. Multiple slides from the same patient are aggregated to a patient-level risk value before Cox-loss calculation and model evaluation.

For the manuscript release, de-identified split manifests can be placed in `splits/` when sharing is permitted. If identifiers cannot be released, document the split-generation procedure and counts in the manuscript and Data Availability statement.
