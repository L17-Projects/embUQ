# Korali vendoring policy

`extern/korali` is the protected vendored Korali source root for this repository. It remains checked in because the current MesoUQ release line depends on a repo-local Korali patch surface, build-entrypoint alignment, and validation fixtures that are not yet represented by a clean upstream dependency boundary.

## What lives where

- source tree: `extern/korali/`
- Korali patch/build notes: `docs/PR21_KORALI_COMPILEABILITY_NOTES.md`
- Karolina runtime contract and acceptance notes: `docs/KAROLINA_FULL_PLATFORM.md`
- bootstrap and validation helpers: `docs/VEGA_BOOTSTRAP.md`, `docs/VEGA_ACCEPTANCE_COMMAND.md`, `docs/PHASE2_BACKEND_STATUS.md`
- protected-source cleanup policy: `docs/ARTIFACT_POLICY.md`

The vendored subtree is not generated output. Cleanup, archive, and inventory tooling must treat `extern/korali` as curated source and must not delete it as if it were a build artifact root.

## Supported platforms

This policy slice covers the Karolina/Slurm runtime contract. The intent is to validate that a checkout can still satisfy the repo-local Korali bootstrap rules on Karolina or another Slurm site that follows the same source-vs-runtime separation.

The validator is intentionally dependency-light and does not import `korali`, `torch`, `pyro`, `mpi4py`, or `matplotlib` at import time.

## Validation command shape

The runtime helper exposes a CLI-ready contract through `meso_uq.platforms.korali_runtime`.

Typical command shape:

```bash
python -m meso_uq.platforms.korali_runtime \
  --repo-root /path/to/MesoUQ \
  --vendor-root /path/to/MesoUQ/extern/korali \
  --build-root /path/to/korali/build-or-install-prefix \
  --library-path /path/to/runtime/lib \
  --pythonpath "/path/to/MesoUQ/src:/path/to/MesoUQ" \
  --path "/usr/bin:/bin" \
  --strict \
  --json
```

The helper also accepts environment snapshot flags for machine checks. Those flags are useful when a test harness wants to mirror the current shell state without importing heavy runtime code.

## Expected environment inputs

The validator understands these runtime hints:

- `MESOUQ_SITE`
- `MESOUQ_SITE_RUNTIME_ROOT`
- `MESOUQ_KORALI_BUILD_ROOT`
- `MESOUQ_KORALI_LIBRARY_PATH`
- `PYTHONPATH`
- `PATH`

Expected behavior:

- `MESOUQ_SITE` should match the validation platform.
- `MESOUQ_SITE_RUNTIME_ROOT` is required and must point at the canonical per-site runtime root used for Korali bootstrap state.
- `MESOUQ_KORALI_BUILD_ROOT` and `MESOUQ_KORALI_LIBRARY_PATH` are optional, but when present they should resolve to real paths and must not point into private-path prefixes.
- `PYTHONPATH` should include repo-local `src/` and the repository root when performing a repo-local Korali bootstrap.
- `PATH` should retain the operator launch helpers needed for the runtime environment.

## Risks

- private absolute paths can leak into runtime hints and silently break portability
- generated-artifact cleanup can accidentally delete vendored source if the source root is not explicitly protected
- missing `PYTHONPATH` or library-path hints can hide bootstrap regressions until the first real job submission
- treating `extern/korali` as disposable build output would erase the local patch surface needed by the current release line

## Migration policy

`extern/korali` stays vendored until the project has an explicit, reviewed replacement that preserves the same runtime behavior, tests, and operator validation story. Any migration away from vendoring must update:

- the vendoring policy
- the runtime validator
- cleanup/archival exclusions
- the Korali bootstrap and acceptance docs

Until then, `extern/korali` remains a protected source root and must not be moved into generated-output handling.
