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

The locked accepted-output set contains only the successful July 24 SonoVue
and July 27 Definity 50k states, their accepted direct-DPD checks, the
paper-figure exports, and their audit records. Interrupted and cancelled HBI
states, failed materializations, and Python caches are excluded. The accepted
Definity `d2` acoustic check is retained with its exact near-MAP provenance
(0.1482% relative offset in `k_a`) in the machine-readable collection
manifest.

Define the two site-independent roots before using the staging commands:

```bash
export MESOUQ_SCRATCH_ROOT=/scratch/project/eu-26-17/eubrieucb/mesouq
export MESOUQ_UQ_EMB_ARTIFACT_ROOT="${MESOUQ_SCRATCH_ROOT}/papers/UQ_EMB/artifacts"
```

Plan the copy and inspect its size without writing any artifact:

```bash
/usr/bin/python3.11 scripts/workflows/emb/uq_emb/stage_external_artifacts.py plan \
  --spec papers/UQ_EMB/manifests/accepted_production_outputs_202607.staging.json
```

Stage the immutable set. The command copies into a temporary directory,
preserves internal hardlinks, computes SHA-256 values, verifies the temporary
copy, and only then publishes it atomically:

```bash
/usr/bin/python3.11 scripts/workflows/emb/uq_emb/stage_external_artifacts.py stage \
  --spec papers/UQ_EMB/manifests/accepted_production_outputs_202607.staging.json \
  --manifest papers/UQ_EMB/manifests/accepted_production_outputs_202607.files.json
```

Verify an existing staged set without modifying it:

```bash
/usr/bin/python3.11 scripts/workflows/emb/uq_emb/stage_external_artifacts.py verify \
  --root "${MESOUQ_UQ_EMB_ARTIFACT_ROOT}/accepted_production_outputs_202607" \
  --manifest papers/UQ_EMB/manifests/accepted_production_outputs_202607.files.json
```

The DNN weights, force-spectroscopy inputs, acoustic observations, 736 physical
frequency labels, polynomial coefficients and banks, mass-scaling evidence,
and surrogate audit records form a separate immutable dependency set. Stage
and verify it with:

```bash
export MESOUQ_WORKSPACE_ROOT=/home/it4i-bbenvegnen/workspace
/usr/bin/python3.11 scripts/workflows/emb/uq_emb/stage_external_artifacts.py stage \
  --spec papers/UQ_EMB/manifests/frozen_runtime_dependencies_202607.staging.json \
  --manifest papers/UQ_EMB/manifests/frozen_runtime_dependencies_202607.files.json
/usr/bin/python3.11 scripts/workflows/emb/uq_emb/stage_external_artifacts.py verify \
  --root "${MESOUQ_UQ_EMB_ARTIFACT_ROOT}/frozen_runtime_dependencies_202607" \
  --manifest papers/UQ_EMB/manifests/frozen_runtime_dependencies_202607.files.json
```

## HBI configuration materialization

The accepted SonoVue and grouped Definity 50k configurations are immutable
inputs in `accepted_production_outputs_202607`. Materialize a 10k replay config
against the verified external dependencies without changing its scientific
settings:

```bash
/usr/bin/python3.11 scripts/workflows/emb/uq_emb/materialize_hbi_config.py \
  --agent definity \
  --artifact-root "${MESOUQ_UQ_EMB_ARTIFACT_ROOT}" \
  --manifest-root papers/UQ_EMB/manifests \
  --output-dir "${MESOUQ_SCRATCH_ROOT}/papers/UQ_EMB/replay/configs" \
  --run-root "${MESOUQ_SCRATCH_ROOT}/papers/UQ_EMB/replay/definity_10k" \
  --population 10000
```

Use `--agent sonovue` for SonoVue and `--population 50000` for an exact-size
production replay. The command verifies the frozen source config, polynomial
bank, bank report, independent-audit receipt, and promotion contract by SHA-256.
It rewrites only external runtime paths, output location, and the three
population fields. A JSON sidecar records every rewrite. The grouped Definity
`source3` configuration is authoritative and remains grouped during
materialization.

## Remaining command surface

The closeout will add separate frozen commands for data preparation, DNN
training, acoustic-surrogate fitting, HBI execution, direct DPD validation,
figures, and manuscript compilation. Those commands delegate shared behavior to
neutral MesoUQ workflow code and preserve both `--site karolina` and
`--site vega`.
