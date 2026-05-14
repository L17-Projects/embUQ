# MesoUQ migration inventory

This inventory records the current repository surface for the scalable-architecture
migration. It is intentionally factual: current roots first, then owner decisions.

Snapshot date: 2026-05-13.

## Top-level areas

| Area | Current role |
|---|---|
| `src/meso_uq` | Shared package API and runtime helpers |
| `compression` | Compression workflow, surrogate, evaluation, and sensitivity code |
| `indentation` | Indentation workflow, surrogate, evaluation, and sensitivity code |
| `inference` | Full-model workflow scripts and configs |
| `reduced` | Reduced-model workflow scripts and configs |
| `sampling` | Lightweight design generation and related helpers |
| `propagation` | Plotting and propagation postprocessing |
| `scripts` | Shared, platform, QA, release, and workflow launchers |
| `ops/huq_emb` | Legacy shim surface for HUQ-EMB launch compatibility |
| `gv` | GV provenance/source staging roots |
| `papers/huq_emb` | Paper-facing replay and asset generation inputs |
| `examples/configs` | Example configs that are refreshed or retired, not treated as stable source of truth |
| `extern/korali` | Vendored backend patch surface |
| `docs` | Public documentation and release evidence |
| `tests` | Repo governance, workflow, and docs sanity checks |

## Current source roots

| Root | Notes |
|---|---|
| `src/meso_uq` | Shared import surface; keep `import meso_uq` stable |
| `emb/compression/src` | Compression-specific runtime and geometry helpers |
| `emb/compression/surrogate` | Compression surrogate training and evaluation |
| `emb/indentation/src` | Indentation-specific runtime and geometry helpers |
| `emb/indentation/surrogate` | Indentation surrogate training and evaluation |
| `inference/scripts` | Full-model phase entrypoints |
| `reduced/scripts` | Reduced-model phase entrypoints |
| `sampling` | Sampling and LHS generation |
| `propagation/scripts` | Propagation and plotting entrypoints |
| `scripts/shared` | Cross-platform helpers |
| `scripts/platforms` | Karolina, Vega, and HPC launcher surfaces |
| `scripts/workflows` | Higher-level workflow orchestration |
| `ops/huq_emb` | Compatibility shim for the HUQ-EMB paper workflow |
| `gv` | GV provenance/source staging tree |
| `extern/korali` | Vendored Korali subtree |
| `papers/huq_emb` | Paper replay assets and generated figure pipelines |

## Config roots

| Root | Notes |
|---|---|
| `inference/configs/production` | Canonical full-model production configs |
| `inference/configs/validation` | Full-model validation configs |
| `reduced/configs/production` | Canonical reduced-model production configs |
| `reduced/configs/validation` | Reduced-model validation configs |
| `examples/configs` | Example and archive candidate configs; refresh or retire as needed |

## Workflow roots

| Root | Notes |
|---|---|
| `scripts/platforms/hpc` | Site-neutral HPC dispatchers |
| `scripts/platforms/karolina` | Karolina-specific launchers, sbatch templates, and validation helpers |
| `scripts/platforms/vega` | Vega-specific launchers, sbatch templates, and validation helpers |
| `scripts/workflows/emb/huq_emb` | HUQ-EMB campaign orchestration |
| `scripts/workflows/gv` | GV workflow orchestration and canary surfaces |
| `scripts/qa` | Documentation and release sanity checks |
| `inference/scripts`, `reduced/scripts` | Public phase runners |
| `scripts/run_vega_acceptance.py` | Thin Vega acceptance wrapper |

## Generated roots

| Root | Current interpretation |
|---|---|
| `_runs` | Canonical run output root for site-aware validation and workflow outputs |
| `_out` | Legacy generated output tree not needed for reproduction |
| `_ci` | CI and coverage artifacts |
| `_init_compression_*` | Generated compression initial conditions |
| `out_hierarchical` | Legacy hierarchical outputs |
| `logs` | Root Slurm logs and similar transient operator output |

## Reproducible inventory commands

These read-only commands reproduce the inventory fields below from the repository root:

```bash
git status --short --ignored=matching --untracked-files=all --branch
git ls-files
git ls-files | rg '(^(_out|_runs|_ci|out_hierarchical)/|^_init_compression_|^logs/|^slurm-.*\.(out|err)$|\.coverage$|\.pytest_cache/|__pycache__/|\.egg-info/|\.(pkl|pt|pth|h5|hdf5|xmf|dat|csv|npy|npz)$)'
git ls-files -z | xargs -0 du -b 2>/dev/null | sort -nr | head -80
for p in emb/compression/evalkit/data emb/indentation/evalkit/data emb/compression/surrogate/diameters emb/indentation/surrogate/diameters gv; do
  echo "$p"
  find "$p" -type f 2>/dev/null | wc -l
  git ls-files "$p" | wc -l
  du -sh "$p" 2>/dev/null
done
find . -maxdepth 2 \( -name '_out' -o -name '_runs' -o -name '_ci' -o -name 'out_hierarchical' -o -name 'logs' -o -name '.pytest_cache' -o -name '*.egg-info' -o -name '_init_compression_*' \) -prune -print -exec du -sh {} \; 2>/dev/null
find . -type f -name 'mesouq-*.out' -o -name 'mesouq-*.err'
```

The snapshot was collected with 1507 tracked files. No tracked files were found under
the repo-root generated roots `_out`, `_runs`, `_ci`, `out_hierarchical`, `logs`,
or `_init_compression_*`.

## Artifact inventory snapshot (2026-05-13)

| Path or glob | Tracked status | Ignored status | Approximate size and count | Extension or class examples | Suspected artifact class | Proposed disposition |
|---|---|---|---|---|---|---|
| `_out` | Not tracked | Ignored | 68M / 1116 files | `.xmf`, `.h5`, `.yaml`, `.csv`, `.off`, `.sh` | Generated run output bundle | Keep out of git; Phase 6 delete/archive candidate after validation evidence |
| `_runs` | Not tracked | Ignored | 480K / 79 files | `.py`, `.sh`, `.yaml`, `.json` | Runtime staging and output tree | Keep ignored; clear stale local runs after owner-approved cleanup |
| `_ci` | Not tracked | Ignored | 3.2M / 11 files | `.json`, `.log`, `.md` | CI coverage/test artifacts | Keep ignored; disposable after checks complete |
| `out_hierarchical` | Not tracked | Ignored | 248K / 85 files | `.yaml`, `.log`, `.txt`, `.dat` | Legacy hierarchical output | Keep ignored; Phase 6 delete/archive candidate |
| `logs` | Not tracked | Ignored | 76K / 2 files | `.log` | Slurm/job logs | Keep ignored; disposable operator logs |
| `.coverage` | Not tracked | Ignored | 1.2M / 1 file | coverage database | Coverage artifact | Keep ignored; regenerated by tests |
| `.pytest_cache` | Not tracked | Ignored | 212K / 6 files | pytest cache | Test cache | Keep ignored; regenerated by pytest |
| `__pycache__/**` | Not tracked | Ignored | About 13M / 762 files in 54 dirs | `.pyc` | Python bytecode cache | Keep ignored; clean locally when needed |
| `*.egg-info` (`src/mesouq.egg-info`) | Not tracked | Ignored | 36K / 5 files in this checkout | package metadata | Packaging artifact | Keep ignored; regenerated by packaging |
| `_init_compression_*` | Not tracked | Ignored | 3 dirs x about 212K / 9 files each | `.yaml`, `.txt`, `.sbatch`, `.py`, `.off` | Generated compression initialization payloads | Keep ignored; Phase 6 delete/archive candidate |
| `runtime/` | Untracked | Not ignored in this snapshot | 88K / 14 files | `.py`, `.sh`, `.yaml`, `.json` | Local GV/runtime staging copy | Needs policy; prefer runtime staging under `_runs/...` or external scratch, not repo root |
| `build/` | Not tracked | Ignored | About 1.0M / 106 files | copied package tree | Build artifact | Keep ignored; regenerated by `python -m build` |
| `dist/` | Not tracked | Ignored | About 508K / 2 files | sdist, wheel | Build artifact | Keep ignored; regenerated by release build |
| root `mesouq-*.out` / `mesouq-*.err` | None present | Pattern policy only | 0 files found | Slurm logs | Root Slurm logs | Keep ignored by policy; no current files to classify |

## EMB data and surrogate artifact classification

| Path or glob | File count | Size | Representative files | Likely class | Proposed disposition | Confidence |
|---|---:|---:|---|---|---|---|
| `emb/compression/evalkit/data` | 12 | 56K | `data_1.csv`, `compression_data_2.1um.dat`, `plot_reference_data.py` | Raw/reference with small processed `.dat` aliases | Keep tracked; manifest as reference inputs and conversion outputs | High |
| `emb/indentation/evalkit/data` | 15 | 52K | `data_morris_3.40.csv`, `indentation_data_3.2um.dat`, `.full`, `.filtered` | Raw/reference plus small processed files | Keep tracked; mark generated/filtered files as processed in manifests | High |
| `emb/compression/surrogate/diameters/*/data` | 3 | 13M | `F_Delta.dat` | Generated or processed training input | Move external once regeneration provenance exists; keep tracked until manifest and validation cover it | Medium |
| `emb/indentation/surrogate/diameters/*/data` | 3 | 46M | `samples_all.dat` | Generated training data | Move external once regeneration provenance exists; keep tracked until manifest and validation cover it | Medium |
| `emb/compression/surrogate/diameters/*/trained` | 6 | 1.1M | `microbubble_force_BEST.pkl`, `microbubble_force_BNN.pt` | Release-critical surrogate checkpoints | Keep `*BEST.pkl` tracked with manifest; BNN checkpoints need explicit release/retrain policy | High for BEST, medium for BNN |
| `emb/indentation/surrogate/diameters/*/trained` | 4 | 431K | `microbubble_displacement_BEST.pkl`, `microbubble_displacement_BNN.pt` | Release-critical surrogate checkpoints | Keep `*BEST.pkl` tracked with manifest; owner decision for incomplete BNN coverage | High for BEST, low for BNN consistency |
| `emb/indentation/evalkit/data/data_morris_3.40_old.csv` | 1 | 983 bytes | legacy-looking CSV | Owner decision / historical reference candidate | Archive or move external only after compatibility decision | Medium |

## Large tracked artifact watchlist

The largest tracked files are mostly surrogate training tables plus vendored Korali
examples. They are not repo-root generated roots, but they should remain visible in
artifact policy decisions:

| Path | Size | Classification |
|---|---:|---|
| `emb/indentation/surrogate/diameters/3.4um/data/samples_all.dat` | 23.9M | Generated training data; externalization candidate |
| `emb/indentation/surrogate/diameters/5.8um/data/samples_all.dat` | 11.4M | Generated training data; externalization candidate |
| `emb/indentation/surrogate/diameters/3.2um/data/samples_all.dat` | 11.4M | Generated training data; externalization candidate |
| `emb/compression/surrogate/diameters/2.9um/data/F_Delta.dat` | 4.5M | Generated/processed training data; externalization candidate |
| `emb/compression/surrogate/diameters/3.0um/data/F_Delta.dat` | 4.1M | Generated/processed training data; externalization candidate |
| `emb/compression/surrogate/diameters/2.1um/data/F_Delta.dat` | 4.0M | Generated/processed training data; externalization candidate |
| `extern/korali/python/korali/profiler/examples/example_single_4096Nodes.json` | 2.5M | Vendored dependency example; keep with `extern/korali` unless vendoring policy changes |

## Disposition definitions

| Disposition | Meaning |
|---|---|
| Keep tracked | Source, small reference data, docs, or release-critical artifacts remain in git. |
| Keep ignored | Local/generated outputs remain untracked and covered by ignore rules. |
| Move external | Store outside git with manifest, checksum, source command, and retrieval or regeneration instructions. |
| Regenerate | Do not version the artifact; CI or operational workflows recreate it from source/config. |
| Archive | Preserve historical evidence outside active source paths. |
| Delete candidate | May be removed only in Phase 6 after owner confirmation, inventory record, rollback note, and validation matrix evidence. |
| Needs owner classification | Static evidence is insufficient to decide whether the item is source, fixture, reference, or generated output. |

## Decisions still required

- Decide whether top-level `runtime/` should be ignored as a local artifact root or migrated under `_runs/...`.
- Confirm the external storage and manifest policy for `emb/compression/surrogate/diameters/*/data/F_Delta.dat`.
- Confirm the external storage and manifest policy for `emb/indentation/surrogate/diameters/*/data/samples_all.dat`.
- Decide whether incomplete indentation BNN checkpoint coverage is acceptable, should be regenerated, or should be moved out of release scope.
- Decide whether `emb/indentation/evalkit/data/data_morris_3.40_old.csv` is a historical reference, an archive candidate, or removable after owner confirmation.
- Keep `extern/korali` vendored unless a separate vendoring policy changes; the large profiler examples are part of that vendored surface for now.

## Artifact classes

| Class | Owner decision |
|---|---|
| Release-critical surrogate artifacts | Keep in git only when curated and manifest-backed |
| Raw/reference data | Keep only if small, license-clear, documented, and reproducibility-critical |
| Generated training data | Move out of git |
| Serialized models and pickles | Track via manifests and compatibility tests, not by accidental directory retention |
| Reports and manifests | Keep as the primary reproducibility contract |
| Paper-facing generated artifacts | Keep `papers/` for now, then move heavier outputs out once manifests exist |
| Legacy path aliases | Keep for one release cycle with shims and warnings |

## Owner decisions

- Public import API stays `meso_uq`; CLI/project branding may be `mesouq`.
- `extern/korali` remains vendored.
- GV should stage from manifests, not own permanent `gv/<experiment>/src` state.
- `examples/configs` should be refreshed or archived instead of treated as durable source.
- Legacy imports and launch paths get one release cycle of compatibility shims and warnings.
- Supported platforms remain Karolina, Vega, and workstation.
- Generated training data and non-reproducible outputs should move out of git.
