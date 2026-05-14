# Generated Artifact Relocation Plan

This document defines the planning slice for generated-data relocation in MES-149.
It is deliberately non-destructive: it classifies paths, records relocation/archive/delete
options, and keeps all destructive actions behind explicit owner confirmation.

## Scope

The policy slice covers:

- owner-approved disposable generated roots: `_out`, `_runs`, `_ci`, `out_hierarchical`
- `_init_compression_*` generated roots
- root-level Slurm logs: `*.out` and `*.err` at the repository root
- protected roots and investigation-only paths
- forbidden private Karolina paths covered by `FORBIDDEN_PRIVATE_PATHS`

It does not move files and it does not delete files.

## Path Inventory

### Approved generated roots

These are the disposable generated roots that can enter archive/delete planning once the
owner confirms the path:

- `_out`
- `_runs`
- `_ci`
- `out_hierarchical`
- `_init_compression_*`

Typical examples:

- `_runs/<site>/<workflow>/<run-id>`
- `_ci/<job-id>`
- `_init_compression_<name>/...`

### Root-level Slurm logs

Files matching `*.out` or `*.err` in the repository root are treated as root-level Slurm
logs. They are relocation candidates, but they still require owner confirmation before a
delete is scheduled.

### Ambiguous generated paths

Nested `.out` and `.err` files are not treated as safe root logs. They are flagged for
investigation because the file may be a workflow trace, a retained run artifact, or a
curated log.

### Protected roots

These paths are protected and should not be marked safe to delete:

- `extern/korali`
- `src`
- `configs`
- `docs`
- `tests`
- `examples`

### Checkpoint-like paths

Compression and indentation trained checkpoints are owner-decision-only paths. They are
not safe-delete candidates and should stay intact until the owner confirms the lineage.

### Forbidden private paths

Any path containing the forbidden private literal is rejected. The Karolina private home prefix
represented in `FORBIDDEN_PRIVATE_PATHS` must be treated as forbidden.

## Validation Rules

1. Normalize candidate paths before classification.
2. Reject any path containing the forbidden private literal.
3. Treat protected roots as investigation-only unless the owner issues a separate override.
4. Treat nested `.out` and `.err` files as ambiguous until provenance is confirmed.
5. Never mark compression or indentation checkpoints as safe to delete.
6. Require an owner decision before any archive/delete action is finalized.

## Rollback Notes

- Archive before delete for approved generated roots and root-level Slurm logs.
- Preserve the original path mapping so an artifact can be restored later.
- Do not attempt rollback on protected roots or forbidden private paths; record the decision
  trail instead.
- For checkpoint-like paths, snapshot the directory before any relocation plan is executed.
- For ambiguous nested logs, keep the file until the parent workflow provenance is clear.

## No-Delete-Until-Owner-Confirmation Policy

No generated artifact is scheduled for deletion until the owner explicitly confirms the
decision. The planning layer may recommend archive, delete, hold, investigate, or reject,
but the actual deletion step is always gated by owner confirmation and path-class safety.

## Operational Checklist

### Before planning

- Confirm the path is in inventory.
- Confirm whether the path is generated, protected, or forbidden.
- Record the owner decision when one exists.

### Before archive or delete

- Confirm the path is an approved generated root or root-level Slurm log.
- Ensure the path is not protected and not forbidden.
- Verify the rollback archive has been staged.
- Capture the decision record with the original and normalized path.

### After plan generation

- Review any `investigate` or `hold` entries.
- Escalate forbidden private paths immediately.
- Keep protected roots and checkpoint-like paths untouched unless the owner provides a
  separate exception.
