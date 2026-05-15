# NativeCuda Phase 2 validation evidence

This page is the repo-side evidence index for the Linear project
`Korali Native-CUDA Phase 2 Migration`. It records what was run, where the
external artifacts live, and which support claims are still blocked.

The public operator switch remains `--phase2-backend native-cuda`. No Korali
C++ interface or MesoUQ config key was intentionally changed by the Phase 2
optimization pass; the optimization stayed inside vendored
`extern/korali/source/modules/problem/hierarchical/psi/`.

## Support status

NativeCuda Phase 2 remains an internal hardening target until the Karolina full
EMB lane and Vega/platform-delta evidence are both closed. CPU-MPI remains the
maintained fallback and parity reference.

## Implementation and post-merge gate

Merged implementation commit:

- PR: `#161`
- commit: `97f8a6a71f16515929b540d1e17bca22bffa9424`
- Linear: `MES-195`, `MES-196`
- changed surface: vendored Korali PSI source and headers only

Post-merge build evidence:

- isolated runtime root:
  `/scratch/project/eu-26-17/eubrieucb/mesouq/runtime/nativecuda_main_97f8a6a_20260515_161912`
- bootstrap log:
  `/scratch/project/eu-26-17/eubrieucb/mesouq/runtime/nativecuda_main_97f8a6a_20260515_161912/logs/bootstrap_korali.log`
- Meson metadata:
  `/scratch/project/eu-26-17/eubrieucb/mesouq/runtime/nativecuda_main_97f8a6a_20260515_161912/korali/build/meson-info/intro-buildoptions.json`
- verified build option: `native_cuda_batch=True`

Post-merge GPU evidence:

- Karolina validation matrix job: `4316740`
- GPU node: `acn15`
- report:
  `/scratch/project/eu-26-17/eubrieucb/mesouq/runs/validation_matrix/cycle-11-after-mes-195-20260515_162320/workflow_matrix_report.json`
- status: `passed`
- targeted CUDA runtime harness job: `4316741`
- harness output:
  `/scratch/project/eu-26-17/eubrieucb/mesouq/runs/native_cuda_phase2/nativecuda-main-97f8a6a-runtime-20260515_162331/pytest_cuda_4316741.out`
- status: `2 passed`

## MES-197 interface propagation

Closeout decision: no public Korali/MesoUQ interface propagation was required
for the optimization commit because no workflow key changed. The repo still
uses:

- `--phase2-backend native-cuda` for operator selection,
- `phase2_backend: native-cuda` in production EMB configs,
- `native_cuda_batch` as the Meson build option,
- `Batch Evaluation Backend = NativeCuda` inside the Korali problem contract.

Focused checks for this no-interface-change closeout should include:

```bash
rg -n "phase2_backend|phase2-backend|native-cuda|NativeCuda|native_cuda_batch" \
  docs scripts src inference reduced tests extern/korali/meson_options.txt
python -m pytest -q \
  tests/unit/test_phase2_backend_selection.py \
  tests/test_vega_operator_helpers.py \
  tests/test_vega_production_sbatch.py \
  tests/test_native_cuda_phase2_baseline_docs.py
```

## MES-199 Karolina full EMB lane

Target lane:

- selection: `compression:full-model:production`
- config:
  `inference/configs/production/inference_config_compression.yaml`
- acceptance stop:
  valid `results_phase_2/latest` plus posterior sanity
- backend: `native-cuda`
- Phase 2 CPU ranks: `1`

Failed attempt retained for audit:

- Slurm job: `4316750`
- output root:
  `/scratch/project/eu-26-17/eubrieucb/mesouq/runs/native_cuda_phase2/mes-199-karolina-compression-full-nativecuda-20260515_162937`
- failure:
  the production compression setup shells out through `python3`; the launcher
  did not put the runtime venv first on `PATH`, so the child process could not
  import `trimesh`.
- fix for rerun:
  export `PATH="$(dirname "$PYTHON_BIN"):$PATH"` before running Phase 1.

Successful rerun:

- Slurm job: `4316753`
- GPU node: `acn15`
- output root:
  `/scratch/project/eu-26-17/eubrieucb/mesouq/runs/native_cuda_phase2/mes-199-karolina-compression-full-nativecuda-rerun-20260515_163151`
- evidence manifest:
  `/scratch/project/eu-26-17/eubrieucb/mesouq/runs/native_cuda_phase2/mes-199-karolina-compression-full-nativecuda-rerun-20260515_163151/manifests/mes199_nativecuda_karolina_full_emb.json`
- `results_phase_2/latest`:
  `/scratch/project/eu-26-17/eubrieucb/mesouq/runs/native_cuda_phase2/mes-199-karolina-compression-full-nativecuda-rerun-20260515_163151/runs/compression/full-model/production/results_phase_2/latest`
- posterior sanity:
  `/scratch/project/eu-26-17/eubrieucb/mesouq/runs/native_cuda_phase2/mes-199-karolina-compression-full-nativecuda-rerun-20260515_163151/runs/compression/full-model/production/results_phase_2/native_cuda_posterior_summary.json`
- sample count: `50000`
- finite log-posterior ratio: `1.0`
- status: `passed`

## MES-200 Vega/platform delta

Status on 2026-05-15: **blocked by Vega platform maintenance/access**.

Scope: this is a scoped platform blocker for Vega NativeCuda validation only.
It is not runtime success evidence, not a Korali correctness failure, and not a
public production-support claim. Karolina evidence above cannot substitute for
Vega GPU visibility, Vega build metadata, or Vega runtime output.

Missing Vega evidence:

- an allocated Vega GPU node with visible NVIDIA devices,
- Vega Korali bootstrap/build metadata with `native_cuda_batch=True`,
- exact Vega module and runtime environment logs,
- a completed Vega `compression:full-model:production` run using
  `--phase2-backend native-cuda` and `--phase2-cpu-ranks 1`,
- Vega `results_phase_2/latest` output and posterior sanity summary,
- Phase 3b consumption of the Vega Phase 2 output, or a recorded failure log if
  that consumption fails after Phase 2 succeeds.

Release-gate implication: keep `native_cuda_phase2_public_claim=false` until
Vega produces either the successful evidence above or a newer release decision
explicitly changes the public support scope.

When Vega access returns, use the same support contract:

```bash
export REPO_ROOT="${PWD}"
module load Python/3.10.8-GCCcore-12.2.0 openmpi/4.1.2.1 CUDA/12.2.2 \
  GSL/2.7-GCC-12.2.0 Eigen/3.4.0-GCCcore-12.2.0
source "${REPO_ROOT}/_vega/venv/bin/activate"
source "${REPO_ROOT}/_vega/korali/env.sh"
export PYTHON_BIN="${REPO_ROOT}/_vega/venv/bin/python"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

"${PYTHON_BIN}" scripts/platforms/vega/run_workflow_matrix.py \
  --selection compression:full-model:production \
  --output-root "${MESOUQ_RUNS_ROOT}/native_cuda_phase2/<run-tag>" \
  --site vega \
  --python-bin "${PYTHON_BIN}" \
  --phase2-backend native-cuda \
  --phase2-cpu-ranks 1 \
  --inference-device gpu \
  --propagation-device gpu \
  --skip-release-manifest
```

Required checks before MES-200 can move from blocked to passed:

```bash
nvidia-smi
"${PYTHON_BIN}" - <<'PY'
import korali
print(korali.__file__)
PY
buildoptions="${REPO_ROOT}/_vega/korali/build/meson-info/intro-buildoptions.json"
"${PYTHON_BIN}" - "${buildoptions}" <<'PY'
import json
import sys
from pathlib import Path

options = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
enabled = any(
    option.get("name") == "native_cuda_batch" and option.get("value") is True
    for option in options
)
if not enabled:
    raise SystemExit("native_cuda_batch Meson option is not true")
PY
test -e "${MESOUQ_RUNS_ROOT}/native_cuda_phase2/<run-tag>/runs/compression/full-model/production/results_phase_2/latest"
test -s "${MESOUQ_RUNS_ROOT}/native_cuda_phase2/<run-tag>/runs/compression/full-model/production/results_phase_2/native_cuda_posterior_summary.json"
```

If Vega remains in maintenance, archive the scheduler or access notice with
this section as the platform-delta artifact instead of treating the absence of a
Vega run as a NativeCuda runtime failure.

## MES-201 evidence bundle rules

Every accepted bundle must include:

- branch and commit,
- platform, node, GPU context, and Slurm job id,
- exact build command and runtime command,
- config path and output root,
- Korali build metadata path,
- `results_phase_2/latest`,
- posterior sanity output,
- failure and rerun evidence when a first attempt fails,
- Linear comments that point to the same external paths.
