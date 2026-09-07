# EMB Indentation

`emb/indentation/` is the canonical asset root for encapsulated microbubble indentation. It replaces the former root-level EMB indentation layout.

- `src/`: indentation-specific Mirheo source templates and generation/equilibration helpers.
- `evalkit/`: indentation reference-data conversion, posterior/likelihood helpers, and small reproducibility assets.
- `surrogate/`: indentation surrogate evaluation/training entrypoints and curated per-diameter surrogate artifacts.

Reusable contracts, registries, model-loading helpers, orchestration utilities, and cross-experiment APIs belong under `src/meso_uq`. Runtime outputs, logs, generated plots, and scratch Mirheo state belong under configured run roots such as `_runs/...`, not here.
