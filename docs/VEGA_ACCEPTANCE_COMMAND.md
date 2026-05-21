# Vega acceptance command

This document describes the single Vega-first acceptance command shipped by `MesoUQ`.

## Purpose

The acceptance command is a thin wrapper around the richer operator runner:

- wrapper: `scripts/run_vega_acceptance.py`
- operator runner: `scripts/platforms/vega/run_validation_suite.py`

The wrapper exists so a fresh clone on Vega has one obvious command to run and one machine-readable report to inspect.

Before running acceptance from a fresh clone, bootstrap the repo-local Korali runtime and verify it:

- `bash scripts/platforms/vega/bootstrap_korali.sh --jobs 8`
- `source ${MESOUQ_SITE_RUNTIME_ROOT}/env/env.sh`
- `python scripts/platforms/vega/doctor_vega.py --strict`

For a canned Vega batch submission, the repo also ships:

- `scripts/platforms/vega/sbatch/acceptance.sbatch`

## Command

```bash
python scripts/run_vega_acceptance.py \
  --output-root ../vega_acceptance \
  --cpu-ranks 1
```

Optional arguments:

- `--python-bin`
- `--korali-pythonpath`
- `--population-size`
- `--workflows`
- `--config-override workflow=/abs/path/config.yaml`

By default, the wrapped validation runner requests the canonical reduced-model validation selections:

- `compression:reduced-model:validation`
- `indentation:reduced-model:validation`

The wrapped validation runner uses the committed validation configs for the selected workflows and preserves the population settings encoded in those YAML files. `--population-size` is only for an explicit operator override.

`--workflows` accepts the canonical `experiment:model-family:profile` selections. Legacy aliases remain accepted only as compatibility shims inside the underlying validation runner.

## Outputs

The command writes:

- `vega_acceptance_report.json`
- `logs/validation_suite.stdout.log`
- `logs/validation_suite.stderr.log`
- `validation_suite/workflow_suite_summary.json`
- the full workflow artifacts produced by the richer validation runner

## Report contents

The JSON report records at least:

- target environment (`vega`)
- environment snapshot
- commands executed
- step return codes and timings
- log paths
- workflow summary artifact paths
- final acceptance status

## Status model

- `passed`: runner succeeded and workflow summary loaded
- `partial`: runner succeeded but the workflow summary could not be parsed fully
- `failed`: runner failed or expected artifacts were missing

## Notes

This command is intentionally Vega-first for now. It should be treated as the public acceptance entrypoint for the cluster validation story, while the richer validation runner remains available for more operator-style usage.
