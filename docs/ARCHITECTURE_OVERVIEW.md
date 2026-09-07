# Architecture Overview (Wave 0/1)

This page is the current architecture map for the migrated slice. It is intentionally limited to what is actually in use after Wave 0/1, with legacy paths called out where they still carry compatibility obligations.

## Migration framing

- **Current migration spine:** the documented workflow and governance surface in `src/meso_uq`, workflow scripts, validation configs, and policy docs.
- **Legacy workflows remain:** compatibility wrappers, old launch entrypoints, and transitional artifacts are kept intentionally for one release cycle.
- **Future extraction:** selected GV and workflow surfaces are explicitly flagged for later extraction once manifest-first handoffs are complete.

## Core contract surfaces

| Concern | Current home | Status |
|---|---|---|
| Shared runtime/API package | `src/meso_uq` | stable |
| Core workflow orchestration | `scripts/workflows/`, `scripts/platforms/`, `scripts/shared/` | active |
| Public configs (examples, schema-first) | `configs/` (`agents`, `modalities`, `datasets`, `surrogates`, `inference`, `noise`, `platforms`, `reports`, `artifacts`) | active |
| Legacy production configs | `inference/configs/*`, `reduced/configs/*` | active legacy + compatibility |
| Validation profiles | `inference/configs/validation/*`, `reduced/configs/validation/*`, plus workflow guides | active |
| Output roots and generated state | `_runs`, `_out`, `_ci`, `_init_*`, `_vega`, `logs` | transitional/non-reproducible roots |

## Agents and modalities

- Agents and modality contracts are represented by schema-example configs under `configs/agents` and `configs/modalities`.
- Canonical examples include:
  - `configs/agents/default_controller.example.yaml`
  - `configs/modalities/emb_compression.example.yaml`
- `docs/CONFIGURATION_POLICY.md` is the owning contract for schema-versioned examples and migration status; it is explicitly scoped as a foundation layer, not a wholesale replacement of legacy runtime trees.

## Config landscape

- `docs/CONFIGURATION_POLICY.md` describes config schema minimums and migration posture.
- `configs/platforms/*.example.yaml` carries policy-first platform config examples.
- Legacy `inference/configs` and `reduced/configs` remain the operator-facing production/validation config roots for current workflows, with compatibility continuity and explicit archive/refresh expectations documented in `docs/MESO_UQ_MIGRATION_INVENTORY.md`.
- `docs/VALIDATION_CONFIGS.md` records the validation profile intent and execution context.

## Artifact governance

- Canonical artifact contract is `docs/ARTIFACT_POLICY.md`.
- Initial manifest example: `configs/artifacts/artifact_manifest.example.json`.
- Practical interpretation: manifests are treated as first-class reproducibility contracts; generated large/bulk outputs remain outside git unless curated as release artifacts.
- `extern/korali` is not an artifact root and remains vendored source.

## Platform policy

- Platform policy for the migration slice is in `docs/PLATFORM_POLICY.md`.
- `configs/platforms/*.example.yaml` is the new placeholder-first policy surface.
- `scripts/platforms/*` remains the active operator entrypoint surface for Karolina, Vega, and workstation.
- `docs/KAROLINA_FULL_PLATFORM.md` and `docs/VEGA_BOOTSTRAP.md` document the site-specific operations that still govern current cluster execution.

## GPU validation

- The closeout gate is defined in `docs/MESO_UQ_GPU_VALIDATION_GATE.md`.
- Supporting operational command/doc coverage is split across:
  - `docs/VALIDATION_MATRIX.md`
  - `docs/VEGA_VALIDATION_MATRIX.md`
  - `docs/VEGA_ACCEPTANCE_COMMAND.md`
  - `docs/VEGA_ACCEPTANCE_CHECKLIST.md`
- Wave 0/1 guidance explicitly requires command, platform, output root, failure summary, and rerun evidence for gate closeout.

## Compatibility surface

- Governed by `docs/MESO_UQ_COMPATIBILITY_SURFACE.md`.
- Includes retained stability points such as:
  - `import meso_uq`
  - legacy matrix/acceptance command compatibility layers
  - GV compatibility staging paths under `gv/` during manifest migration
- This is a measured compatibility layer, not a permanent API architecture target.

## Risk register

- Current Wave 0/1 risk posture is in `docs/MESO_UQ_MIGRATION_RISK_REGISTER.md`.
- It tracks reproducibility, API stability, HPC operator drift, artifact movement, and validation continuity with explicit controls and evidence pointers.

## Migration repository map

- Source layout and canonical root mapping: `docs/MESO_UQ_MIGRATION_INVENTORY.md`
- Governance history and linked evidence index: `docs/architecture/MES-127_131_WAVE0_GOVERNANCE_HUB.md`

## High-level contribution map

- Start with this document and the migration inventory.
- Use the policy documents (`CONFIGURATION_POLICY`, `ARTIFACT_POLICY`, `PLATFORM_POLICY`, `MESO_UQ_COMPATIBILITY_SURFACE`) before changing workflow behavior.
- Validate any architecture claim against `WORKFLOWS.md`, `RELEASE_SCOPE.md`, and the risk register.
- Any GPU path change must include the closeout evidence workflow before merge-readiness status is asserted.
