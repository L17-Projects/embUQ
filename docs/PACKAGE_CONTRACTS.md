# Package Contracts and Metadata Boundaries

This page is the Phase 2 contract map for MES-137, MES-138, MES-139, and MES-142.
It defines where new metadata belongs while the legacy workflow entry points remain in place.

## Source-of-truth modules

- `src/meso_uq/core/contracts.py`: dependency-light identities, run metadata, artifact references, surrogate identifiers, dataset source metadata, runtime requirements, and manifest records.
- `src/meso_uq/agents/registry.py`: agent-family registry, aliases, default legacy behavior, supported modalities, supported backends, inference backends, runtime requirements, and unsupported-combination errors.
- `src/meso_uq/agents/emb/` and `src/meso_uq/agents/gv/`: metadata-only family boundaries for EMB and GV constants.
- `src/meso_uq/agents/emb/workflows.py`: EMB compression/indentation generation contracts, legacy script identities, config-resolution candidates, parameter-file names, and modality-specific runtime-prep constants.
- `src/meso_uq/modalities/registry.py`: modality descriptors for EMB compression, EMB indentation, GV stretching, GV buckling, GV torsion, GV eigenmodes, and experimental GV shear flow.
- `src/meso_uq/simulation/emb_generation.py`: shared EMB parameter-sweep, command-file, and legacy Sbatch generation logic consumed by the compatibility `compression/src/generate.py` and `indentation/src/generate.py` entry points.
- `src/meso_uq/surrogate/emb_workflows.py`: shared EMB surrogate workflow contracts for deterministic NN and BNN wrapper commands, checkpoint metadata, dataset split metadata, backend resolution, and grouped holdout orchestration.
- `src/meso_uq/workflows/legacy.py`: compatibility inventory, legacy evalkit path setup, project-root/config resolution, surrogate runtime parsing, trained-directory resolution, and once-per-surface warning helpers for legacy compression, indentation, inference, reduced, and propagation entry points.
- `src/meso_uq/public_api.py`: narrow dependency-light public API for contracts, registries, and descriptor lookup.

These modules must remain importable without Torch, Pyro, Matplotlib, MPI, Mirheo, Korali, Slurm helpers, checkpoints, generated data, or HPC runtime directories.

## What belongs where

Core contracts:
- stable string identifiers used by configs, manifests, CLI flags, reports, and tests;
- serializable metadata such as `RunMetadata`, `ManifestMetadata`, `ArtifactReference`, `DatasetSourceMetadata`, and `SurrogateIdentifier`;
- runtime requirement records that describe optional, external, required, or unsupported dependencies;
- no workflow execution, filesystem probing, checkpoint loading, plotting, simulator execution, or HPC module handling.

Agent packages:
- family identity and aliases, such as `emb`, `elastic_microbubble`, `gv`, and `gas_vesicle`;
- which modalities, backends, inference backends, platforms, artifact classes, and runtime requirements a family supports;
- whether a family is the legacy default.

Modality descriptors:
- controls, observables, surrogate input/output conventions, config schema names, artifact manifest expectations, capability flags, and runtime requirements;
- family-specific meanings. A modality name alone is not enough to infer parameter schema, observables, or runtime behavior.

Workflow code:
- remains under the existing compatibility paths until wrapper retirement is explicitly scheduled;
- should consume contracts rather than redefine identities, path labels, or support matrices.
- legacy EMB generation wrappers may delegate shared parameter-sweep and command-file behavior to `src/meso_uq/simulation/emb_generation.py`, but they must keep the existing CLI flags, default object names, generated file names, and Sbatch text unless a dedicated compatibility issue changes them.
- legacy EMB surrogate training, BNN training, and grouped holdout wrappers may delegate parser and orchestration behavior to `src/meso_uq/surrogate/emb_workflows.py`; new surrogate backends should add metadata and backend resolution there first, then provide backend-specific runtime code behind optional imports.
- legacy evalkit and propagation wrappers should use `src/meso_uq/workflows/legacy.py` for compatibility-only path/config/backend mechanics rather than duplicating project-root probing, `sys.path` setup, surrogate backend parsing, or trained-artifact path construction.

## Compatibility rules

- `meso_uq` remains the public import namespace.
- The root `import meso_uq` must stay minimal and expose only the existing version contract.
- Legacy compression, indentation, inference, reduced, propagation, script, and Slurm entry points remain compatibility surfaces for the migration window.
- Compatibility wrappers that are still user/HPC-facing should be listed in `meso_uq.workflows.legacy.LEGACY_WORKFLOW_SURFACES` with the replacement package API before behavior is moved.
- Serialized artifact compatibility takes precedence over import cleanup. Do not remove legacy modules used by pickle artifacts unless a shim is present and tested.
- GV runtime source layout must remain manifest-driven; contract metadata must not assume permanent checked-in runtime source ownership.

## Import-safety guards

`tests/unit/test_import_safety_contracts.py` locks down metadata import behavior for:

- `meso_uq`
- `meso_uq.public_api`
- `meso_uq.agents`
- `meso_uq.modalities`
- `meso_uq.structures`
- `meso_uq.structures.gv`
- `meso_uq.surrogate.catalogs`
- `meso_uq.surrogate.emb_catalog`
- `meso_uq.surrogate.gv_catalog`
- `meso_uq.workflows`
- `meso_uq.workflows.legacy`

The tests also exercise cycle-sensitive import orders for the GV structure registry and surrogate catalog modules. Heavy optional dependencies must be imported only when dependent functionality is invoked.
