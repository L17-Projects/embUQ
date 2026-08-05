# UQ_EMB paper reproduction

This is the canonical paper-specific entrypoint for the UQ-driven EMB
manuscript. Reusable scientific and HPC behavior remains under `src/meso_uq`
and `scripts/workflows`; this directory owns frozen editorial sources,
paper-specific manifests, and reproduction instructions.

## Immutable editor submission

`editor_submission/review2_v1/` is the exact review-v2 package sent to the
editor. It is a scoped exception to the general external-artifact policy: the
complete 16 MiB package is retained in Git because it is the immutable review
baseline. Do not edit files inside that snapshot.

The frozen exception is limited to 20 MiB in total and 8 MiB per file. Activate
a supported Python 3.10+ environment and set `MESOUQ_PYTHON` to that
interpreter before running the commands below. On Karolina, use
`export MESOUQ_PYTHON=/usr/bin/python3.11`.

Verify it from the repository root:

```bash
"${MESOUQ_PYTHON:?set MESOUQ_PYTHON to Python 3.10+}" \
  scripts/workflows/emb/uq_emb/verify_frozen_submission.py verify \
  --root papers/UQ_EMB/editor_submission/review2_v1 \
  --manifest papers/UQ_EMB/manifests/editor_submission_review2_v1.json
```

The snapshot was staged with the safety-gated command below. The source hint is
documentary; verification depends only on paths, sizes, and SHA-256 values in
the tracked manifest.

```bash
"${MESOUQ_PYTHON:?set MESOUQ_PYTHON to Python 3.10+}" \
  scripts/workflows/emb/uq_emb/verify_frozen_submission.py snapshot \
  --source "${MESOUQ_WORKSPACE}/paperWeek27Jul/bundle_review2_v1" \
  --destination papers/UQ_EMB/editor_submission/review2_v1 \
  --manifest papers/UQ_EMB/manifests/editor_submission_review2_v1.json \
  --snapshot-id review2_v1 \
  --source-hint '${MESOUQ_WORKSPACE}/paperWeek27Jul/bundle_review2_v1'
```

## External artifacts

Heavy HBI states, trajectories, fitted-frequency inputs, and logs remain under
the persistent artifact root:

```text
/scratch/project/eu-26-17/eubrieucb/mesouq/papers/UQ_EMB/artifacts/
```

They will be addressed through manifests and explicit staging commands. No
paper workflow may depend on symlinks into working directories.

## Planned command surface

The closeout adds separate, frozen commands for data preparation, DNN training,
acoustic-surrogate fitting, 10k/50k HBI, direct DPD validation, figures, and
manuscript compilation. Those commands will delegate shared behavior to neutral
MesoUQ workflow code and preserve both `--site karolina` and `--site vega`.
