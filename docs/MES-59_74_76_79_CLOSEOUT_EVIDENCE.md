# MES-59 / MES-74 / MES-76 / MES-79 closeout evidence

This note records the repo-visible decisions and validation surfaces for the
MES-59, MES-74, MES-76, and MES-79 closeout PR.

## MES-59 merge readiness

Merge readiness is governed by
`docs/MES-59_MERGE_READINESS_CHECKLIST.md`. The checklist requires both an
immediate connector review sweep after the latest push and a delayed,
thread-aware sweep shortly before merge. Flat top-level PR comments are not
sufficient; unresolved inline connector threads must be inspected, fixed, or
recorded with owner/rationale/follow-up before the PR can be called
merge-ready.

## MES-74 bottom-row Phase 1 range policy

The old bottom-row x-range adjustment for the Phase 1 reduced representative
figure is incorporated as explicit code rather than discarded. The behavior is
captured by `_phase1_representative_histogram_xlim()` and the bottom-row
constants in `papers/huq_emb/uqdpd_generate_reduced_story_assets.py`.

Regression coverage:

```bash
PYTHONPATH=src python3.11 -m pytest -q tests/test_uqdpd_map_phase3b_adapter.py
```

The MES-74 regression test checks that the bottom-row preview keeps the broader
`1.25` / `0.30` span policy and remains broader than the older unmerged
`0.85` / `0.18` tweak.

## MES-76 MAP Mirheo smoke initialization

MAP Mirheo smoke runs prepare dataset-specific scratch init directories under
the lane output tree. Operators do not pre-create repo-root `_init_*_map`
directories for the smoke path.

Documented surfaces:

- `docs/WORKFLOWS.md`
- `docs/VEGA_BOOTSTRAP.md`

Executable/reporting surfaces:

- `scripts/platforms/vega/run_map_mirheo.py`
- `scripts/platforms/vega/run_map_mirheo_sanity.py`

Regression coverage:

```bash
PYTHONPATH=src python3.11 -m pytest -q \
  tests/test_map_mirheo_sanity_runner.py \
  tests/unit/test_map_mirheo_orchestration.py \
  tests/unit/test_map_mirheo_scripts.py
```

Missing Phase 3b MAP manifests, missing MAP init templates, and missing Mirheo
bootstrap inputs are hard failures. They are not treated as silent skips.

## MES-79 Karolina full-suite evidence

The Karolina full pytest suite is launched through:

```bash
RUN_TAG=<tag> \
OUTPUT_ROOT=/scratch/project/eu-26-17/eubrieucb/mesouq/runs/full_test_suite/<tag> \
PYTHON_BIN=${MESOUQ_SITE_RUNTIME_ROOT}/env/bin/python \
sbatch --export=ALL,REPO_ROOT="$PWD",RUN_TAG="$RUN_TAG",OUTPUT_ROOT="$OUTPUT_ROOT",PYTHON_BIN="$PYTHON_BIN" \
  scripts/platforms/karolina/sbatch/full_test_suite.sbatch -q
```

The wrapper requests one A100 GPU on `qgpu_exp`, loads the Karolina runtime
libraries, and records Slurm metadata, git state, command, pytest result, and
artifact paths in `full_test_suite_summary.json`.

The wrapper intentionally hides Slurm job identity variables from the pytest
process while retaining them in the summary. This keeps unit tests that assert
repo-local defaults independent of the enclosing Slurm allocation.

MPI/GPU applicability: the full-suite command is a single-task repository
pytest suite. It does not launch MPI ranks. The GPU allocation proves Karolina
compute-node/runtime parity and CUDA visibility for the pytest suite, not a
dedicated MPI or GPU workflow.

Latest full-suite evidence:

- Slurm job: `4302486`
- Node/partition/GPU: `acn32`, `qgpu_exp`, one A100 allocation
- Summary artifact:
  `/scratch/project/eu-26-17/eubrieucb/mesouq/runs/full_test_suite/mes79_full_pytest_main_1df732a_summaryfix_20260513_142720/full_test_suite_summary.json`
- Result: `1799 passed, 1 skipped, 5 warnings in 53.37s`

Strict merge-ready coverage evidence:

- Artifact root:
  `/scratch/project/eu-26-17/eubrieucb/mesouq/reports/merge_ready_coverage/mes59_74_76_79_final_20260513_143814`
- Base: `1779 passed, 9 skipped`
- PR: `1788 passed, 9 skipped`
- Strict coverage delta: `+0.066` percentage points
