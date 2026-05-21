# Target-platform Phase 2 native-CUDA validation checklist

This checklist is the minimum acceptance layer needed before `MesoUQ` can honestly claim a supported **Phase 2 native-CUDA** path.

## Goal

Validate whether the vendored Korali native-CUDA backend can support the public `Hierarchical/Psi` Phase 2 workflow on the real target environment. The current target platforms are Karolina and Vega; archive platform-specific deltas separately.

## Preconditions

Before attempting this checklist, verify:

- the desired `MesoUQ` branch is checked out,
- the vendored `extern/korali/` subtree is present and internally coherent,
- a compatible CUDA module stack is loaded,
- a matching MPI toolchain is loaded,
- `mpi4py` imports correctly in the runtime environment,
- the runtime Python environment is first on `PATH` so child `python3` calls use
  the same packages as the launcher,
- successful Phase 1 outputs already exist for the intended dataset set.
- the NativeCuda Phase 2 command uses one CPU/MPI rank. Multi-rank Phase 2 is the CPU-MPI fallback path, not native-CUDA.

## Build-side checks

1. Configure vendored Korali with the intended Meson options.
2. Record whether `-Dnative_cuda_batch=true` is enabled.
3. Record the CUDA toolkit path used by Meson.
4. Confirm that the configuration step resolves:
   - `cuda.h`
   - `nvrtc.h`
   - driver stub/library
   - NVRTC library
5. Save the exact Meson summary in the validation notes.

## Runtime-side checks

1. Run a small public Phase 2 dataset set on the target platform.
2. Record the exact config used.
3. Record whether the run completes without Korali load/state errors.
4. Record whether the run writes a valid `results_phase_2/latest` output.
5. Record whether downstream Phase 3b can load that Phase 2 result.

For Karolina full-lane validation, prefer the current workflow-matrix entrypoint
with a single production lane:

```bash
python scripts/platforms/vega/run_workflow_matrix.py \
  --selection compression:full-model:production \
  --output-root "${MESOUQ_RUNS_ROOT}/native_cuda_phase2/<run-tag>" \
  --site karolina \
  --python-bin "${MESOUQ_SITE_RUNTIME_ROOT}/env/bin/python" \
  --phase2-backend native-cuda \
  --phase2-cpu-ranks 1 \
  --inference-device gpu \
  --propagation-device gpu \
  --skip-release-manifest
```

If the lane prepares compression runtime assets, set:

```bash
export PYTHON_BIN="${MESOUQ_SITE_RUNTIME_ROOT}/env/bin/python"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
```

before launching. Some experiment setup helpers invoke `python3` from shell
commands, so `PATH` must resolve to the same runtime environment as
`--python-bin`.

## Performance / behavior checks

1. Confirm whether the native-CUDA path is actually exercised, not merely compiled.
2. Record GPU visibility and allocation during Phase 2.
3. Record whether runtime behavior differs from the documented CPU-MPI baseline.
4. Record any correctness mismatches relative to the standard public Phase 2 path.

## Minimum evidence to archive

For a successful acceptance result, archive at least:

- the exact branch and commit SHA,
- the Meson configure command,
- the Meson summary,
- the runtime launch command,
- stdout/stderr logs,
- the resulting `results_phase_2/latest` artifact path,
- the posterior sanity summary generated from `results_phase_2/latest`,
- a short statement about whether Phase 3b successfully consumed the Phase 2 output.

## Honest pass/fail rule

Do **not** mark Phase 2 native-CUDA as supported unless all of the following are true:

- build configuration succeeds,
- runtime execution succeeds,
- valid Phase 2 outputs are written,
- posterior sanity passes for `results_phase_2/latest`,
- Phase 3b can consume those outputs,
- the evidence listed above is archived.
