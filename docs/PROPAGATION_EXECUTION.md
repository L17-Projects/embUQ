# Propagation execution

This note documents the lightweight public propagation execution surface added for the required public Phase 3b path.

## Scope

The current public propagation layer evaluates already-sampled posterior parameter sets through the public surrogate evaluators and writes summary CSV files.

It is intentionally lightweight:

- it uses the current public posterior sample outputs,
- it uses the current public surrogate evaluators,
- it writes summary predictions that the plotting layer can consume.

## Public entrypoints

Phase 1 propagation:

```bash
python propagation/scripts/run_phase1_propagation.py \
  --config inference/configs/production/inference_config_compression.yaml \
  --output-dir _setup
```

Phase 3b propagation:

```bash
python propagation/scripts/run_phase3b_propagation.py \
  --config inference/configs/production/inference_config_compression.yaml \
  --output-dir _setup
```

## Outputs

For each enabled dataset, the scripts write:

- `propagation_phase1/<dataset>/summary.csv` or `propagation_phase3b/<dataset>/summary.csv`
- `references/<dataset>.csv`

The summary CSV contains:

- `x`
- `mean`
- `median`
- `q05`
- `q95`

## Plotting path

The public plotting layer can then consume the reference CSV and propagation summary CSV using the existing validation-overlay plotting utility.

## Honest status

This is a lightweight public propagation execution surface for the current surrogate-based public workflow.

It should not be interpreted as a verbatim import of every historical propagation script from upstream.
