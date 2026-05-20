# NativeCuda Phase 2 baseline

This note records the MES-189/MES-190 K1 baseline for the vendored Korali
`Hierarchical/Psi` NativeCuda path. It is a recovery and strategy record, not a production-support claim.

Native-CUDA Phase 2 remains an internal hardening target until validation
passes. CPU-MPI remains the maintained fallback and must keep parity coverage.

## Current call path

The public MesoUQ entrypoint is
[`inference/scripts/run_phase_2.py`](../inference/scripts/run_phase_2.py). It
accepts `--phase2-backend cpu-mpi` and `--phase2-backend native-cuda`. Without
an explicit override, production configs resolve to `native-cuda`; validation
configs resolve to `cpu-mpi`.

For `phase2_backend=native-cuda`, the driver:

- requires one MPI rank,
- configures a Sequential Korali conduit,
- sets `Problem/Use Batch Evaluation = true`,
- sets `Problem/Batch Evaluation Backend = NativeCuda`,
- checks repo-local Meson metadata for `native_cuda_batch` when available.

For `phase2_backend=cpu-mpi`, the driver keeps the Distributed/MPI Korali path
available. The reduced wrapper delegates to the same driver, and the Vega/HPC
helper surface forwards the same `--phase2-backend` contract.

Public production helpers must therefore pair `phase2_backend=native-cuda` with
one Phase 2 CPU rank. Multi-rank Phase 2 belongs to the maintained
`phase2_backend=cpu-mpi` fallback.

## Korali baseline

The implementation lives under
[`extern/korali/source/modules/problem/hierarchical/psi/`](../extern/korali/source/modules/problem/hierarchical/psi/).
The current Korali configuration exposes three batch backends:

- `External`
- `NativeCpu`
- `NativeCuda`

The native batch cache supports `Univariate/Normal` and `Univariate/Uniform`
conditional priors. Constant conditional-prior terms are folded into cached
sub-problem base log weights. Dynamic conditional-prior metadata is evaluated
per candidate.

The current NativeCuda path is guarded by `_KORALI_USE_CUDA_BATCH`, which is set
from the Meson `native_cuda_batch` option. If Korali is built without that
option, selecting `NativeCuda` is a hard runtime error.

Today the CUDA kernel is embedded in `kPsiNativeCudaKernelSource` inside
`psi.cpp`. During `Psi::initializeNativeCudaBatch()`, Korali:

- initializes the CUDA driver API and retains the primary context,
- queries the active device compute capability,
- compiles the embedded kernel with NVRTC through `nvrtcCreateProgram` and
  `nvrtcCompileProgram`,
- loads the generated PTX with `cuModuleLoadData`,
- extracts the kernel entry and stores a `CUfunction`,
- allocates persistent device buffers for sub-problem coordinates, base log
  weights, sample counts, and dynamic-prior metadata.

During `Psi::evaluateBatchNativeCuda()`, Korali still performs host-side prior
evaluation and host-side final reduction. It reuses persistent per-batch device
buffers for flattened batch parameters and per-subproblem likelihood output,
growing capacity when a larger batch requires it. The path copies flattened
batch parameters to the device, launches the kernel, copies per-subproblem
likelihoods back, and writes `Batch logPrior` plus `Batch logLikelihood`.
`releaseNativeCudaBatch()` releases the retained device buffers during teardown.

`HUQ_PSI_NATIVE_CUDA_PROFILE_JSONL` enables JSONL records for setup and batch
timing. For Phase 2 migration this profiling output is for optimization runs,
not for every validation run.

`meso_uq.inference.native_cuda_performance` and
`scripts/platforms/hpc/check_native_cuda_profile.py` are the fixed-threshold
profile harness for those optimization runs. They parse setup and batch records,
report maximum and mean timing buckets, and fail with the exact exceeded bucket
when a threshold is violated. The default thresholds are intentionally broad
synthetic-harness gates; Karolina or Vega validation evidence must record any
platform-specific overrides used for a campaign.

## Build surface

The current build gate is `-Dnative_cuda_batch=true` in
[`extern/korali/meson_options.txt`](../extern/korali/meson_options.txt). When
enabled, [`extern/korali/meson.build`](../extern/korali/meson.build) expects
CUDA headers under `CUDA_HOME` or `/opt/cuda`, checks `cuda.h` and `nvrtc.h`,
links the CUDA driver library or stubs, links NVRTC, and defines
`_KORALI_USE_CUDA_BATCH`.

The site-neutral bootstrap entrypoint is
[`scripts/platforms/hpc/bootstrap_korali.sh`](../scripts/platforms/hpc/bootstrap_korali.sh).
It dispatches to the selected site. The Vega bootstrap accepts
`--native-cuda-batch`; Karolina delegates through the same bootstrap path after
setting the Karolina site environment. Existing CPU-MPI builds continue to
require MPI support because CPU-MPI is still the fallback.

## Strategy decision

For production hardening, the selected target is a Meson-built CUDA
object/module delivery path, not the current long-term NVRTC runtime-compiled
kernel string.

The K1 baseline does not implement that migration. It documents that the
current implementation is NVRTC-based and chooses the next delivery direction:

- move the kernel into source-controlled CUDA build input, such as a `.cu`
  translation unit or generated module owned by vendored Korali,
- compile it during Korali bootstrap through Meson and the site CUDA toolkit,
- record the chosen GPU architecture/fatbin settings in build logs and Meson
  metadata,
- load or call the built artifact at runtime without requiring NVRTC compilation
  in every Phase 2 job,
- keep the public MesoUQ interface centered on `phase2_backend=native-cuda`
  unless a later cleaner interface/config key is propagated across MesoUQ.

Rationale for Karolina/Vega-style HPC stacks:

- Reproducibility: compile diagnostics and toolkit selection belong in the
  bootstrap log instead of changing at every runtime initialization.
- Operator burden: batch jobs should not need to diagnose NVRTC availability or
  runtime compiler/library mismatches.
- Performance: startup should not pay repeated kernel compilation cost before
  useful Phase 2 work begins.
- Debuggability: Meson/Ninja failures are easier to archive and compare than
  runtime PTX generation failures inside Korali sampling.
- Public interface: the MesoUQ workflow should keep exposing backend intent,
  while Korali owns the kernel-delivery details.

Until that migration lands, `--native-cuda-batch` still means the current
NVRTC-backed implementation. Validation evidence must state which delivery mode
was actually used.

## Posterior parity contract

CPU-MPI remains the maintained fallback while NativeCuda hardening proceeds. The
Phase 2 validation contract is statistical equivalence, not byte-for-byte TMCMC
trajectory identity.

The package helper
`meso_uq.inference.posterior_equivalence` defines the shared report shape used
for CPU-MPI vs NativeCuda comparisons:

- both backends are read through the same Korali posterior state reader,
  `load_phase2_posterior_samples(...)`,
- summaries compare parameter means, sample standard deviations, configured
  quantiles, finite log-posterior ratio, sample count, variable names, and
  maximum finite log-posterior,
- failures name the backend side and statistic that violated the threshold,
- reports serialize through `PosteriorEquivalenceReport.to_manifest()` for
  later operational validation records.

The default thresholds are intentionally small synthetic-harness defaults:
`min_sample_count=2`, `min_finite_logposterior_ratio=1.0`,
`max_mean_abs_delta=5e-2`, `max_std_scaled_delta=2e-1`,
`max_quantile_abs_delta=1e-1`, and
`max_logposterior_max_abs_delta=1.0`. Full EMB lane validation may tighten or
loosen these explicitly in the validation manifest, but it must record the
actual threshold object used.

## Non-goals for K1

- no Meson-built CUDA object/module delivery migration,
- no public production-support claim,
- no required Korali C++ interface or config-key change,
- no requirement that JSONL profiling be enabled on every validation run,
- no vault-derived old-implementation scope beyond what current repo files and
  MES-189/MES-190 already state.

The current repo does not contain a concrete old UQ_DPD/Korali commit reference
or an old NativeCuda harness under the allowed K1 source-of-truth boundary. That
missing old-reference evidence remains a follow-up discovery item.

## Validation gates

Before NativeCuda Phase 2 can be treated as production-supported:

- update Korali build/bootstrap/doctor coverage for the Meson-built CUDA
  delivery path on both Karolina and Vega,
- add tests that parse and compile the kernel build input,
- compare NativeCuda results against both native CPU batch and scalar
  `Hierarchical/Psi` references,
- cover Normal priors, Uniform priors, mixed priors, invalid sigma/range,
  non-finite priors, empty batches, multiple subproblems, and 50k-like
  populations,
- run `pytest -m cuda` runtime coverage on target hardware,
- add performance harnesses with fixed thresholds,
- complete one full EMB lane with valid `results_phase_2/latest` output and
  posterior sanity,
- archive evidence in repo manifests, external campaign storage, and Linear.

The Vega-facing checklist remains
[`docs/VEGA_PHASE2_NATIVE_CUDA_CHECKLIST.md`](VEGA_PHASE2_NATIVE_CUDA_CHECKLIST.md).
Karolina platform evidence is governed by
[`docs/KAROLINA_FULL_PLATFORM.md`](KAROLINA_FULL_PLATFORM.md) and the GPU gate in
[`docs/MESO_UQ_GPU_VALIDATION_GATE.md`](MESO_UQ_GPU_VALIDATION_GATE.md).

## CUDA runtime test invocation

The CUDA runtime harness is intentionally outside the default test path. Run it
only on a GPU allocation with CUDA driver and NVRTC libraries visible:

```bash
source scripts/platforms/karolina/env_karolina.sh
unset MESOUQ_SITE MESOUQ_SITE MESOUQ_RUNS_ROOT MESOUQ_SCRATCH_ROOT MESOUQ_SITE_RUNTIME_ROOT MESOUQ_PROVENANCE_ROOT MESOUQ_GV_ENV_SCRIPT GV_SCALE_SPACE_BINARY GV_CGAL_TOOLS_ROOT SLURM_JOB_ID SLURM_ARRAY_JOB_ID
export PYTHONPATH="${PWD}/src:${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
python -m pytest -q -m cuda tests/integration/test_native_cuda_psi_runtime.py
```

Use the same command after sourcing the documented Vega environment on Vega.
If CUDA driver devices or NVRTC are absent, the test skips with an actionable
message; an executed test failure is a NativeCuda correctness or runtime issue.

## Performance profile harness

Enable the Korali profile writer during an optimization run, then check the
resulting JSONL file with explicit thresholds:

```bash
export HUQ_PSI_NATIVE_CUDA_PROFILE_JSONL="$PWD/_runs/native_cuda/phase2_native_cuda.jsonl"
# run the targeted NativeCuda Phase 2 or synthetic CUDA harness here
python scripts/platforms/hpc/check_native_cuda_profile.py \
  "$HUQ_PSI_NATIVE_CUDA_PROFILE_JSONL" \
  --output-json "$PWD/_runs/native_cuda/native_cuda_profile_report.json" \
  --max-batch-total-seconds 10 \
  --max-batch-alloc-seconds 1 \
  --max-batch-h2d-seconds 1 \
  --max-batch-compute-seconds 5 \
  --max-batch-d2h-seconds 1 \
  --max-batch-host-reduce-seconds 1
```

The report includes setup and batch counts, max/mean setup buckets, max/mean
batch buckets, maximum batch size, parameter count, sub-problem count, dynamic
prior count, the thresholds used, and any exceeded bucket names.
