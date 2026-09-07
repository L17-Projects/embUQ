# EMB Compression

`emb/compression/` is the canonical asset root for encapsulated microbubble compression. It replaces the former root-level EMB compression layout.

- `src/`: compression-specific Mirheo source templates and generation/equilibration helpers.
- `evalkit/`: compression reference-data preparation, posterior/likelihood helpers, and small reproducibility assets.
- `surrogate/`: compression surrogate evaluation/training entrypoints and curated per-diameter surrogate artifacts.

Reusable contracts, registries, model-loading helpers, orchestration utilities, and cross-experiment APIs belong under `src/meso_uq`. Runtime outputs, logs, generated plots, and scratch Mirheo state belong under configured run roots such as `_runs/...`, not here.
