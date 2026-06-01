# Shared shell helpers for DPD production preflight in Slurm wrappers.

mesouq_sanitize_nested_srun_cpu_env() {
  # Some GPU jobs expose both values; nested srun treats the pair as fatal.
  if [[ -n "${SLURM_CPUS_PER_TASK:-}" && -n "${SLURM_TRES_PER_TASK:-}" ]]; then
    echo "Sanitizing nested srun CPU task environment: SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}, SLURM_TRES_PER_TASK=${SLURM_TRES_PER_TASK}"
    unset SLURM_TRES_PER_TASK
  fi
}

mesouq_run_dpd_production_preflight() {
  PREFLIGHT_SCRIPT="${PREFLIGHT_SCRIPT:-${REPO_ROOT}/scripts/workflows/dpd/run_dpd_production_preflight.py}"
  DPD_PREFLIGHT="${DPD_PREFLIGHT:-1}"
  DPD_PREFLIGHT_MIN_FREE_BYTES="${DPD_PREFLIGHT_MIN_FREE_BYTES:-1000000000}"
  DPD_PREFLIGHT_HDF5_DRIVER="${DPD_PREFLIGHT_HDF5_DRIVER:-serial}"
  DPD_PREFLIGHT_HDF5_SMOKE_TEST="${DPD_PREFLIGHT_HDF5_SMOKE_TEST:-1}"

  if [[ "${DPD_PREFLIGHT}" != "1" || "${EXECUTION_MODE:-execute}" != "execute" ]]; then
    return 0
  fi
  if [[ ! -f "${PREFLIGHT_SCRIPT}" ]]; then
    echo "Preflight script missing: ${PREFLIGHT_SCRIPT}" >&2
    exit 2
  fi
  if [[ -z "${PYTHON_EXECUTABLE:-}" || ! -x "${PYTHON_EXECUTABLE}" ]]; then
    echo "PYTHON_EXECUTABLE must resolve to an executable before DPD preflight." >&2
    exit 2
  fi
  if [[ -z "${REPO_ROOT:-}" || -z "${CANDIDATE_MANIFEST:-}" ]]; then
    echo "REPO_ROOT and CANDIDATE_MANIFEST are required before DPD preflight." >&2
    exit 2
  fi

  local preflight_args=(
    --repo-root "${REPO_ROOT}"
    --candidate-manifest "${CANDIDATE_MANIFEST}"
    --min-free-bytes "${DPD_PREFLIGHT_MIN_FREE_BYTES}"
  )
  if [[ -n "${SCRATCH_ROOT:-}" ]]; then
    preflight_args+=(--scratch-root "${SCRATCH_ROOT}")
  fi
  if [[ -n "${MESOUQ_SCRATCH_ROOT:-}" && "${MESOUQ_SCRATCH_ROOT}" != "${SCRATCH_ROOT:-}" ]]; then
    preflight_args+=(--scratch-root "${MESOUQ_SCRATCH_ROOT}")
  fi
  if [[ "${DPD_PREFLIGHT_HDF5_SMOKE_TEST}" == "1" ]]; then
    preflight_args+=(--hdf5-smoke-test --hdf5-driver "${DPD_PREFLIGHT_HDF5_DRIVER}")
  fi

  "${PYTHON_EXECUTABLE}" "${PREFLIGHT_SCRIPT}" "${preflight_args[@]}"
}
