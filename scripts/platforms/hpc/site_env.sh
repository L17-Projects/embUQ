#!/usr/bin/env bash
# Shared activation guard for HPC entrypoints that must use the canonical
# ${MESOUQ_SITE_RUNTIME_ROOT}/env environment.

mesouq_activate_site_env() {
  local expected_site="$1"
  local repo_root="${2:-${REPO_ROOT:-}}"
  local legacy_name="HPC""_SITE"

  if [[ ${!legacy_name+x} ]]; then
    echo "${legacy_name} is no longer supported. Use MESOUQ_SITE." >&2
    return 2
  fi
  if [[ -n "${SITE:-}" && "${SITE}" != "${expected_site}" ]]; then
    echo "This site template is fixed to ${expected_site}; SITE=${SITE} conflicts." >&2
    return 2
  fi
  if [[ -n "${MESOUQ_SITE:-}" && "${MESOUQ_SITE}" != "${expected_site}" ]]; then
    echo "This site template is fixed to ${expected_site}; MESOUQ_SITE=${MESOUQ_SITE} conflicts." >&2
    return 2
  fi
  export MESOUQ_SITE="${expected_site}"

  if [[ -z "${MESOUQ_SITE_RUNTIME_ROOT:-}" ]]; then
    echo "MESOUQ_SITE_RUNTIME_ROOT must be set; expected canonical env at \${MESOUQ_SITE_RUNTIME_ROOT}/env/env.sh." >&2
    return 2
  fi

  local canonical_env_root="${MESOUQ_SITE_RUNTIME_ROOT%/}/env"
  if [[ -n "${MESOUQ_ENV_ROOT:-}" && "${MESOUQ_ENV_ROOT%/}" != "${canonical_env_root}" ]]; then
    echo "MESOUQ_ENV_ROOT=${MESOUQ_ENV_ROOT} conflicts with canonical ${canonical_env_root}." >&2
    return 2
  fi
  export MESOUQ_ENV_ROOT="${canonical_env_root}"

  local canonical_env_script="${MESOUQ_ENV_ROOT}/env.sh"
  if [[ -n "${MESOUQ_ENV_SCRIPT:-}" && "${MESOUQ_ENV_SCRIPT}" != "${canonical_env_script}" ]]; then
    echo "MESOUQ_ENV_SCRIPT=${MESOUQ_ENV_SCRIPT} conflicts with canonical ${canonical_env_script}." >&2
    return 2
  fi
  if [[ -n "${MESOUQ_GV_ENV_SCRIPT:-}" && "${MESOUQ_GV_ENV_SCRIPT}" != "${canonical_env_script}" ]]; then
    echo "MESOUQ_GV_ENV_SCRIPT=${MESOUQ_GV_ENV_SCRIPT} conflicts with canonical ${canonical_env_script}." >&2
    return 2
  fi
  export MESOUQ_ENV_SCRIPT="${canonical_env_script}"
  export MESOUQ_GV_ENV_SCRIPT="${canonical_env_script}"

  if [[ ! -f "${MESOUQ_ENV_SCRIPT}" ]]; then
    echo "Missing canonical MesoUQ environment: ${MESOUQ_ENV_SCRIPT}" >&2
    return 2
  fi

  source "${MESOUQ_ENV_SCRIPT}"

  ENV_ROOT="${MESOUQ_ENV_ROOT}"
  ENV_SCRIPT="${MESOUQ_ENV_SCRIPT}"
  if [[ -z "${PYTHON_BIN:-}" || "${PYTHON_BIN}" == "python" ]]; then
    PYTHON_BIN="${MESOUQ_ENV_ROOT}/bin/python"
  fi
  export ENV_ROOT ENV_SCRIPT PYTHON_BIN

  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing canonical MesoUQ Python: ${PYTHON_BIN}" >&2
    return 2
  fi

  if [[ -n "${repo_root}" ]]; then
    case ":${PYTHONPATH:-}:" in
      *":${repo_root}/src:"*) ;;
      *) export PYTHONPATH="${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" ;;
    esac
  fi
}
