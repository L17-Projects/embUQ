# Release evidence checklist for `v0.1.0`

This document is historical and release-specific. It describes the archived `v0.1.0` evidence
contract only; it does not define the current HUQ-EMB rebuild branch contract.

This checklist defines the minimum evidence bundle required before tagging `MesoUQ` as `v0.1.0`.

The goal is to make the release bar explicit, reviewable, and machine-checkable.

## Required evidence

Before tagging, archive all of the following for the candidate commit:

1. one green GitHub CI run
2. one green GitHub Release Smoke run
3. one archived Vega validation matrix report
4. one archived Vega acceptance report
5. one archived Vega production-sanity report
6. one archived workstation acceptance report
7. finalized release notes and release-facing docs
8. one machine-readable release evidence manifest

## Mandatory non-claim for `v0.1.0`

For `v0.1.0`, the release evidence must state explicitly that:

- native-CUDA Phase 2 is **not** part of the claimed public support contract

If that field is flipped to `true`, the release evidence is invalid for `v0.1.0`.

## Canonical manifest

The canonical manifest filename is:

- `release_evidence_manifest.json`

An example manifest is shipped at:

- `examples/reports/release_evidence_manifest.example.json`

Validate a completed manifest with:

```bash
python scripts/qa/release/validate_release_evidence.py \
  --report /path/to/release_evidence_manifest.json \
  --must-exist
```

## Minimum manifest contents

The manifest must include:

- `release`
- final `status`
- the candidate Git commit
- GitHub CI and Release Smoke workflow URLs
- archived report paths for:
  - Vega validation matrix
  - Vega acceptance
  - Vega production sanity
  - workstation acceptance
- core release-facing document paths
- the explicit `native_cuda_phase2_public_claim: false` field

## Practical release gate

Treat the release as blocked until:

- the candidate commit is on `main`
- the required GitHub checks are green for that commit
- the required hardware-validated reports exist and are archived
- the release manifest validates cleanly

## Recommended archive layout

Keep the evidence bundle together under one release root, for example:

- `release_evidence_manifest.json`
- `logs/`
- `github_ci/`
- `vega/validation_matrix/`
- `vega/acceptance/`
- `vega/production_sanity/`
- `workstation/`

## Related documents

- `RELEASE_NOTES_v0.1.0.md`
- `VALIDATION_MATRIX.md`
- `WORKSTATION_ACCEPTANCE_CHECKLIST.md`
- `VEGA_ACCEPTANCE_CHECKLIST.md`
- `VEGA_PRODUCTION_SANITY.md`
