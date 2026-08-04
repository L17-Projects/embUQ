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

The tracked file manifest is the acceptance decision. For a new artifact set,
create it once with the explicit `snapshot` command, review it, and commit it
before staging. `snapshot` refuses to overwrite an existing manifest:

```bash
/usr/bin/python3.11 scripts/workflows/emb/uq_emb/stage_external_artifacts.py snapshot \
  --spec papers/UQ_EMB/manifests/accepted_production_outputs_202607.staging.json \
  --manifest papers/UQ_EMB/manifests/accepted_production_outputs_202607.files.json
```

Stage the immutable set against that existing locked manifest. The command
copies into a temporary directory, preserves internal hardlinks, verifies every
path, size, and SHA-256 value, and only then publishes the copy atomically. It
never creates or overwrites the manifest:

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
bank, bank build tool, bank report, independent-audit receipt, and promotion
contract by SHA-256.
It rewrites only external runtime paths, output location, and the three
population fields. A JSON sidecar records every rewrite. The grouped Definity
`source3` configuration is authoritative and remains grouped during
materialization. The sidecar also records a location-independent semantic
configuration digest, the two locked-manifest hashes, the Git commit, and the
runtime identity. Forward canaries and HBI replays refuse configs without this
matching sidecar.

## Forward-artifact canary

Before starting HBI, exercise all three mechanical DNNs and every configured
acoustic likelihood lane for one agent. The command evaluates a three-row batch
at each diameter, runs the complete acoustic preflight, and records model hashes,
array shapes, finite ranges, positive observation uncertainties, source grouping,
and wall time:

```bash
${MESOUQ_ENV_ROOT}/bin/python \
  scripts/workflows/emb/uq_emb/run_forward_canary.py \
  --site karolina \
  --device cuda \
  --config "${MESOUQ_SCRATCH_ROOT}/papers/UQ_EMB/replay/configs/definity_hbi_10000.yaml" \
  --output "${MESOUQ_SCRATCH_ROOT}/papers/UQ_EMB/replay/canaries/karolina_definity.json"
```

Use `--site vega` on Vega after staging and verifying the same immutable
dependency set below that site's `MESOUQ_UQ_EMB_ARTIFACT_ROOT`. The neutral
launcher requires an explicit site and submits the selected site wrapper so its
scheduler directives are honored:

```bash
REPO_ROOT="$PWD" \
MESOUQ_SITE_RUNTIME_ROOT="${MESOUQ_SITE_RUNTIME_ROOT}" \
CONFIG_PATH="${MESOUQ_SCRATCH_ROOT}/papers/UQ_EMB/replay/configs/definity_hbi_10000.yaml" \
OUTPUT_PATH="${MESOUQ_SCRATCH_ROOT}/papers/UQ_EMB/replay/canaries/karolina_definity.json" \
bash scripts/platforms/hpc/sbatch/uq_emb_forward_canary.sbatch --site karolina
```

Replace `definity` with `sonovue` for the second agent. The Definity receipt must
show one grouped acoustic dataset with 14 reference rows. SonoVue must show three
separate acoustic datasets with one row each.

After copying the two Vega receipts to Karolina, compare the scientific payload
while ignoring only location, scheduler, interpreter-path, and timing fields.
The comparison requires the same semantic configuration digest and Git commit:

```bash
/usr/bin/python3.11 scripts/workflows/emb/uq_emb/compare_forward_canaries.py \
  --agent definity \
  --karolina "${MESOUQ_SCRATCH_ROOT}/papers/UQ_EMB/replay/canaries/karolina_definity.json" \
  --vega "${MESOUQ_SCRATCH_ROOT}/papers/UQ_EMB/replay/canaries/vega_definity.json" \
  --output "${MESOUQ_SCRATCH_ROOT}/papers/UQ_EMB/replay/canaries/definity_cross_site.json"
```

## HBI replay

Print and record the exact accepted Phase 1, native-CUDA Phase 2, and Phase 3b
commands without running them:

```bash
${MESOUQ_ENV_ROOT}/bin/python \
  scripts/workflows/emb/uq_emb/run_hbi_replay.py \
  --site karolina \
  --config "${MESOUQ_SCRATCH_ROOT}/papers/UQ_EMB/replay/configs/definity_hbi_10000.yaml" \
  --output-root "${MESOUQ_SCRATCH_ROOT}/papers/UQ_EMB/replay/definity_10k" \
  --python-bin "${MESOUQ_ENV_ROOT}/bin/python"
```

Add `--execute` only inside an appropriate GPU allocation. The runner rejects
an existing Phase 1 tree instead of overwriting it and updates a JSON receipt
after every completed stage. The neutral production launcher is:

```bash
REPO_ROOT="$PWD" \
MESOUQ_SITE_RUNTIME_ROOT="${MESOUQ_SITE_RUNTIME_ROOT}" \
CONFIG_PATH="${MESOUQ_SCRATCH_ROOT}/papers/UQ_EMB/replay/configs/definity_hbi_10000.yaml" \
OUTPUT_ROOT="${MESOUQ_SCRATCH_ROOT}/papers/UQ_EMB/replay/definity_10k" \
bash scripts/platforms/hpc/sbatch/uq_emb_hbi_replay.sbatch --site karolina
```

The equivalent Vega wrappers are under `scripts/platforms/vega/sbatch`. Both
sites delegate to the same neutral Python runners and retain identical
scientific configuration.

## Manuscript replay

Compile the immutable editor submission into a fresh scratch directory with
the frozen TinyTeX toolchain. The command verifies the submission manifest,
preserves the submitted clean and marked PDFs as baselines, resolves the
bibliographies and cross-references, and requires matching page counts and
extracted text for the two main-manuscript PDFs:

```bash
/usr/bin/python3.11 scripts/workflows/emb/uq_emb/compile_manuscript.py \
  --build-root "${MESOUQ_SCRATCH_ROOT}/papers/UQ_EMB/replay/manuscript_$(date +%Y%m%dT%H%M%S)" \
  --pdflatex "${MESOUQ_SCRATCH_ROOT}/runs/jcp_june02_repro_audit_20260628/tinytex-local/bin/x86_64-linux/pdflatex" \
  --bibtex "${MESOUQ_SCRATCH_ROOT}/runs/jcp_june02_repro_audit_20260628/tinytex-local/bin/x86_64-linux/bibtex"
```

The build directory must be absent or empty. A machine-readable
`uq_emb_manuscript_replay_receipt.json` records every command, output hash,
page count, baseline comparison, and wall time.

## Accepted figure replay

Figures 6, 7, and 9 are replayed from immutable accepted outputs and a separate
plotting-dependency set. Stage and verify the plotting inputs and the small
historical conversion runtime once:

```bash
/usr/bin/python3.11 scripts/workflows/emb/uq_emb/stage_external_artifacts.py stage \
  --spec papers/UQ_EMB/manifests/frozen_plotting_dependencies_202607.staging.json \
  --manifest papers/UQ_EMB/manifests/frozen_plotting_dependencies_202607.files.json
/usr/bin/python3.11 scripts/workflows/emb/uq_emb/stage_external_artifacts.py verify \
  --root "${MESOUQ_UQ_EMB_ARTIFACT_ROOT}/frozen_plotting_dependencies_202607" \
  --manifest papers/UQ_EMB/manifests/frozen_plotting_dependencies_202607.files.json
/usr/bin/python3.11 scripts/workflows/emb/uq_emb/stage_external_artifacts.py stage \
  --spec papers/UQ_EMB/manifests/frozen_legacy_paper_runtime_complete_202606.staging.json \
  --manifest papers/UQ_EMB/manifests/frozen_legacy_paper_runtime_complete_202606.files.json
/usr/bin/python3.11 scripts/workflows/emb/uq_emb/stage_external_artifacts.py verify \
  --root "${MESOUQ_UQ_EMB_ARTIFACT_ROOT}/frozen_legacy_paper_runtime_complete_202606" \
  --manifest papers/UQ_EMB/manifests/frozen_legacy_paper_runtime_complete_202606.files.json
```

The three indentation conversion constants used by Figure 9 are tracked in
`manifests/figure9_indentation_conversion_constants_202606.json`. They equal
twice the median of column 8 in each historical surrogate sample table, with
the source hashes and row counts recorded alongside each value. This avoids
retaining 45 MiB of training tables solely for three deterministic scalars.

Define the frozen paths and render into new, empty scratch directories:

```bash
PLOT_ROOT="${MESOUQ_UQ_EMB_ARTIFACT_ROOT}/frozen_plotting_dependencies_202607"
ACCEPTED_ROOT="${MESOUQ_UQ_EMB_ARTIFACT_ROOT}/accepted_production_outputs_202607"
LEGACY_ROOT="${MESOUQ_UQ_EMB_ARTIFACT_ROOT}/frozen_legacy_paper_runtime_complete_202606"
TEX_BIN="${MESOUQ_SCRATCH_ROOT}/runs/jcp_june02_repro_audit_20260628/tinytex-local/bin/x86_64-linux"

/usr/bin/python3.11 scripts/workflows/emb/uq_emb/render_figure6_replay.py \
  --renderer "${PLOT_ROOT}/code/figure6/render_figure6.py" \
  --paper-style "${PLOT_ROOT}/code/paper_style/uqdpd_generate_reduced_story_assets.py" \
  --paper-style-source "${PLOT_ROOT}/code/paper_style" \
  --rows "${PLOT_ROOT}/inputs/figure6/leave_one_out_rows.csv" \
  --tex-bin-dir "${TEX_BIN}" --texdeps-dir "${PLOT_ROOT}/texdeps" \
  --output-dir "${MESOUQ_SCRATCH_ROOT}/papers/UQ_EMB/replay/figure6" \
  --baseline-pdf papers/UQ_EMB/editor_submission/review2_v1/Figure_6.pdf

/usr/bin/python3.11 scripts/workflows/emb/uq_emb/render_figure7_replay.py \
  --code-root "${PLOT_ROOT}/code/figure7" \
  --renderer-inputs "${PLOT_ROOT}/inputs/figure7/renderer_inputs" \
  --phase1-overlay "${PLOT_ROOT}/inputs/figure7/definity_phase1_overlay_780k.csv.gz" \
  --phase1-manifest "${PLOT_ROOT}/inputs/figure7/definity_phase1_overlay_780k.manifest.json" \
  --tex-bin-dir "${TEX_BIN}" --texdeps-dir "${PLOT_ROOT}/texdeps" \
  --output-dir "${MESOUQ_SCRATCH_ROOT}/papers/UQ_EMB/replay/figure7" \
  --baseline-pdf papers/UQ_EMB/editor_submission/review2_v1/Figure_7.pdf

/usr/bin/python3.11 scripts/workflows/emb/uq_emb/render_figure9_replay.py \
  --renderer "${PLOT_ROOT}/code/figure9/render_figure9.py" \
  --accepted-root "${ACCEPTED_ROOT}" --plotting-root "${PLOT_ROOT}" \
  --old-generator "${LEGACY_ROOT}/legacy_tree/_paper/v3/scripts/generate_reduced_story_assets.py" \
  --conversion-constants papers/UQ_EMB/manifests/figure9_indentation_conversion_constants_202606.json \
  --tex-bin-dir "${TEX_BIN}" --texdeps-dir "${PLOT_ROOT}/texdeps" \
  --output-root "${MESOUQ_SCRATCH_ROOT}/papers/UQ_EMB/replay/figure9" \
  --baseline-pdf papers/UQ_EMB/editor_submission/review2_v1/Figure_9.pdf
```

Each renderer refuses a non-empty output directory and writes a receipt. With
the frozen toolchain, the rasterized result must match the submitted editor
asset exactly.

## Mechanical DNN provenance

The accepted DNN training tables and weights are immutable runtime dependencies.
Verify their hashes and row counts, record the recovered network architectures,
and emit prospective refresh commands with explicit new seeds:

```bash
RUNTIME_ROOT="${MESOUQ_UQ_EMB_ARTIFACT_ROOT}/frozen_runtime_dependencies_202607"
/usr/bin/python3.11 scripts/workflows/emb/uq_emb/audit_dnn_surrogate_provenance.py \
  --dependency-root "${RUNTIME_ROOT}" \
  --manifest papers/UQ_EMB/manifests/frozen_runtime_dependencies_202607.files.json \
  --repo-root "$PWD" \
  --output-root "${MESOUQ_SCRATCH_ROOT}/papers/UQ_EMB/replay/dnn_refresh" \
  --python-bin "${MESOUQ_ENV_ROOT}/bin/python" \
  --seed 202607 --max-epoch 100 \
  --receipt "${MESOUQ_SCRATCH_ROOT}/papers/UQ_EMB/replay/dnn_provenance.json"
```

The receipt deliberately reports `exact_retraining: false`: the accepted
architecture-sweep receipts and random seeds were not preserved. The accepted
weights are verified exactly; the emitted commands are a documented statistical
refresh baseline, not a byte-identical retraining claim.

## Acoustic polynomial replay

Recompute every exact-diameter squared-frequency polynomial from the frozen
physical-frequency table and compare its coefficients and in-support
predictions with the accepted banks:

```bash
RUNTIME_ROOT="${MESOUQ_UQ_EMB_ARTIFACT_ROOT}/frozen_runtime_dependencies_202607"
${MESOUQ_ENV_ROOT}/bin/python \
  scripts/workflows/emb/uq_emb/replay_acoustic_polynomial_surrogates.py \
  --artifact-root "${RUNTIME_ROOT}/acoustic_surrogates" \
  --output-dir "${MESOUQ_SCRATCH_ROOT}/papers/UQ_EMB/replay/acoustic_polynomials_$(date +%Y%m%dT%H%M%S)"
```

The replay enforces the accepted policy: one free-intercept quadratic in
`f_res^2` per exact diameter, fitted to the highest 28 of 32 labels with
unweighted frequency-space residuals. It rejects fallback, diameter
interpolation, and evaluation outside the declared support.

## Direct-DPD replay readiness

Materialize all six mechanical and all six acoustic validation commands without
submitting or executing DPD:

```bash
ACCEPTED_ROOT="${MESOUQ_UQ_EMB_ARTIFACT_ROOT}/accepted_production_outputs_202607"
DIRECT_ROOT="${MESOUQ_SCRATCH_ROOT}/papers/UQ_EMB/replay/direct_dpd_$(date +%Y%m%dT%H%M%S)"
GV_PYTHON="${MESOUQ_SITE_RUNTIME_ROOT}/gv_venv/bin/python"
/usr/bin/python3.11 scripts/workflows/emb/uq_emb/materialize_direct_dpd_replay.py \
  --accepted-root "${ACCEPTED_ROOT}" \
  --accepted-manifest papers/UQ_EMB/manifests/accepted_production_outputs_202607.files.json \
  --output-root "${DIRECT_ROOT}" \
  --site "${MESOUQ_SITE}" \
  --python-bin "${GV_PYTHON}"
```

Before verifying or executing the generated commands, enter the Mirheo/OpenMPI/
CUDA module environment provided by the selected site's scheduler wrapper, then
activate the shared runtime:

```bash
export MESOUQ_REPO_ROOT="$PWD"
source scripts/platforms/hpc/site_env.sh
mesouq_activate_site_env "${MESOUQ_SITE}" "${MESOUQ_REPO_ROOT}"
source "${MESOUQ_SITE_RUNTIME_ROOT}/gv_venv/env.sh"
```

Run the sub-minute readiness verifier in that activated environment. It checks
each accepted source against the locked manifest, reconstructs and compares all
12 canonical commands, checks the materialized hashes, and loads each acoustic
input through the frozen breathing runner. It never launches Mirheo:

```bash
"${GV_PYTHON}" scripts/workflows/emb/uq_emb/verify_direct_dpd_replay_plan.py \
  --plan "${DIRECT_ROOT}/direct_dpd_replay_plan.json" \
  --receipt "${DIRECT_ROOT}/direct_dpd_replay_verification.json"
```

The Definity `d2` acoustic command intentionally retains the user-approved
`k_a=17812.5` same-protocol point, 0.1482% from the inferred MAP. The other five
acoustic launch commands are reconstructed from retained MAP and setup
manifests using the byte-identical frozen runner and protocol values. The plan
records that the historical shell launch wrapper was not uniformly preserved.
