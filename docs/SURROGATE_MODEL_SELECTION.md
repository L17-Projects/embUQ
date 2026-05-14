# Surrogate model selection

This note documents the lightweight public model-selection surface added on top of the existing surrogate retraining utilities.

## Scope

The current public model-selection surface performs a simple grid search over:

- network width
- network depth

for the existing tabular MLP surrogate training path.

It does **not** currently claim to expose every architecture-search utility that may exist upstream.

## Public entrypoints

Compression:

```bash
python emb/compression/surrogate/scripts/emb_model_select.py TRAINING_TABLE.dat \
  --output-dir trained/model_selection
```

By default, compression now runs the 12-architecture paper sweep:

`32x2,32x3,32x4,64x2,64x3,64x4,64x5,128x2,128x3,128x4,256x2,256x3`

You can override this with `--architectures WIDTHxDEPTH,...`.

Indentation:

```bash
python emb/indentation/surrogate/scripts/emb_model_select.py TRAINING_TABLE.dat \
  --output-dir trained/model_selection \
  --widths 32,64,128 \
  --depths 2,3,4
```

Paper-facing full sweep + BEST promotion:

```bash
python emb/compression/surrogate/scripts/train_multi_arch.py --diameter 2.1
python emb/indentation/surrogate/scripts/train_multi_arch.py --diameter 3.2
```

## Outputs

For each width/depth combination, the model-selection helper writes:

- a trained model pickle
- `leaderboard.csv` with validation and training losses
- `best_model.json` describing the best configuration in the tested grid

## Honest status

This is a lightweight public model-selection layer centered on the shared `meso_uq.surrogate` core.

It should be understood as a practical public selection surface, not as a claim that the full upstream non-paper architecture-search machinery has already been imported verbatim.

## Rebuild contract

For the HUQ-EMB paper rebuild, the canonical Vega path is not the lightweight `emb_model_select.py` utility.
Use the multi-architecture trainers, or the six-diameter Vega matrix runner:

```bash
python scripts/platforms/vega/run_dnn_surrogate_training.py
```

That path promotes the selected model into the canonical `diameters/*/trained/*_BEST.pkl` artifacts consumed by inference and postprocess.
It also writes one per-spec provenance manifest at:

- `<output-root>/<spec>/dnn_training_manifest.json`

including the effective config, SLURM array job id, log locations, promoted BEST artifact checksum, and final training-report checksum when available.
