# Phase 6 Generated Cleanup Record

MES-165 records the owner-approved closeout scope for generated repository-root outputs left by local, workstation, CI, and scheduler runs. The approved local cleanup was executed in the working tree with explicit paths, not with broad `git clean` flags.

## Approved Local Outputs

The following repository-root paths are owner-approved generated outputs. They are ignored by repository policy and should remain untracked local artifacts:

- `_out/`
- `_runs/`
- `_init_compression_*`
- `out_hierarchical/`
- `_ci/`
- `logs/`
- root Slurm scheduler logs such as `mesouq-validation-matrix-4304444.out`, `mesouq-validation-matrix-4304444.err`, and `%x-%j` style `job-name-12345.out` / `job-name-12345.err`
- root `runtime/` staging output from local dry runs and GV staging experiments

Current `.gitignore` coverage is intentional:

- `_init*/` and `_*` cover `_init_compression_*`, `_out/`, `_runs/`, and `_ci/`.
- `out_*` covers `out_hierarchical/`.
- `logs/` covers repository-local log directories.
- `/runtime/` covers generated root runtime staging output without ignoring source runtime modules.
- `/*-*.out`, `/*-*.err`, `/mesouq-*.out`, and `/mesouq-*.err` cover root scheduler logs without ignoring nested source, fixture, or documentation files.

## Protected Paths

Cleanup must not delete or rewrite curated repository content, including:

- `extern/korali/`
- curated surrogate manifests, fixtures, trained baseline artifacts, and example reports
- source, configuration, documentation, and test paths: `src/`, `compression/`, `indentation/`, `inference/`, `examples/`, `scripts/`, `docs/`, and `tests/`
- paper-specific source scripts and documentation
- compatibility-shim documentation and tests

If a path is ambiguous, keep it and escalate to the owner rather than classifying it as generated output.

## Cleanup Execution

The closeout cleanup removed ignored local outputs matching the approved scope from the active working tree:

```bash
rm -rf _out _runs _ci out_hierarchical _init_compression_* logs build dist .coverage .pytest_cache mesouq-validation-matrix-*.out mesouq-validation-matrix-*.err
```

The command intentionally did not remove `extern/korali`, source trees, configs, docs, tests, curated surrogate artifacts, or paper source. It also did not use `git clean`.

## Validation Expectations

Closeout validation must verify that approved generated roots and root scheduler log patterns are ignored by Git, and that none of the approved generated roots are tracked in the index. The full branch closeout also requires the local validation suite and Karolina GPU validation matrix.

Minimal local validation for this record:

```bash
pytest tests/test_phase6_closeout_policy.py
```
