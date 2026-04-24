# Vega production sanity

This page documents the public production-sanity smoke command for Vega.

## Goal

The production-sanity command proves that one production-like workflow lane can run end to end from a clean clone without using the full production populations or generation counts.

It is intentionally narrower than the validation matrix:

- it defaults to one production lane
- it keeps shipped production semantics
- it lowers only the workflow cost knobs
- it emits standardized repo-native plots and a machine-readable report

## Default lane

If no lane is specified, the command runs:

- `compression:full-model:production`

Additional production lanes can be selected with repeated `--selection` flags, or all four production lanes can be requested with `--all-lanes`.

## Smoke path

For each selected lane, the command runs:

- Phase 1
- Phase 2
- Phase 3b
- propagation Phase 3b
- MAP extraction from Phase 3b outputs

The command intentionally skips Phase 1 MAP extraction for this smoke surface.

## Reduced-cost override

The command derives a temporary override config from the shipped production config for each selected lane.

The current reduced-cost defaults are:

- `pop_size: 10`
- `max_gen: 1`
- `hbi_pop_size: 10`
- `phase3b_pop_size: 10`
- `phase3b_max_gen: 1`
- `map_n_displacements: 1`

All other production semantics remain unchanged, including reduced-model fixed parameters and enabled diameters.

## Command

Run it inside an allocated Vega job:

```bash
python scripts/platforms/vega/run_production_sanity.py \
  --output-root _vega/production_sanity \
  --phase2-cpu-ranks 2 \
  --python-bin python
```

To run a non-default lane:

```bash
python scripts/platforms/vega/run_production_sanity.py \
  --selection indentation:reduced-model:production \
  --output-root _vega/production_sanity/indentation_reduced_model
```

To run multiple lanes explicitly:

```bash
python scripts/platforms/vega/run_production_sanity.py \
  --selection compression:full-model:production \
  --selection compression:reduced-model:production \
  --output-root _vega/production_sanity/compression_pair
```

## Outputs

The command writes:

- one machine-readable report:
  - `_vega/production_sanity/<label>/production_sanity_report.json`
- one generated override config per selected lane:
  - `_vega/production_sanity/<label>/configs/*.yaml`
- the nested workflow matrix outputs:
  - `_vega/production_sanity/<label>/matrix/...`
- captured workflow-matrix logs:
  - `_vega/production_sanity/<label>/logs/...`
- standardized plots:
  - `_vega/production_sanity/<label>/plots/...`

The standardized plot bundle includes:

- Phase 1 Korali posterior marginals for every enabled dataset
- Phase 2 Korali posterior marginals for the selected lane
- Phase 3b Korali posterior marginals for every enabled dataset
- propagation Phase 3b summary plots for every enabled dataset
- `MAP vs surrogate vs ref` plots for every enabled dataset

## Korali backend statement

This command records the current repo-local Korali build state in the report.

It does not claim support for native-CUDA Phase 2 unless the current Korali build and workflow validation prove that separately.
