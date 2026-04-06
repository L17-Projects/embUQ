# PR20 workflow reconciliation notes

This branch adds hardened candidate replacements for the two main PR20 execution drivers:

- `inference/scripts/run_phase_1_hardened.py`
- `inference/scripts/run_phase_3b_hardened.py`

## Why these files exist

The current public workflow spine in `MesoUQ` is close to runnable, but it still needs a workflow-reconciliation pass after the runtime-preparation import.

These hardened candidates restore and strengthen behavior that is needed for the PR20 goals:

- restart handling for Phase 1
- dry-run parameter patching for compression Phase 1
- explicit output-root normalization relative to the project root
- execution from the project root so runtime preparation and output paths stay coherent
- stronger Phase 3b verification of prerequisite Phase 1 and Phase 2 states
- stronger load-state checks and diagnostics for the Phase 3b dataset loop
- preservation of output paths compatible with MAP extraction and postprocessing

## Intended replacement mapping

Once the final PR20 patch is applied to the canonical public entrypoints, the content of these candidates should replace:

- `inference/scripts/run_phase_1.py`
- `inference/scripts/run_phase_3b.py`

## Scope alignment with the vault plan

This is meant to satisfy the workflow-reconciliation part of the release hardening plan:

- reconnect the imported runtime core to the public workflow spine
- validate path assumptions, runtime directories, temporary outputs, and config selection
- keep Phase 1 GPU-batched and Phase 3b GPU-batched paths internally coherent
- preserve compatibility with MAP extraction
