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
python compression/surrogate/scripts/emb_model_select.py TRAINING_TABLE.dat \
  --output-dir trained/model_selection
```

By default, compression now runs the 12-architecture paper sweep:

`32x2,32x3,32x4,64x2,64x3,64x4,64x5,128x2,128x3,128x4,256x2,256x3`

You can override this with `--architectures WIDTHxDEPTH,...`.

Indentation:

```bash
python indentation/surrogate/scripts/emb_model_select.py TRAINING_TABLE.dat \
  --output-dir trained/model_selection \
  --widths 32,64,128 \
  --depths 2,3,4
```

## Outputs

For each width/depth combination, the model-selection helper writes:

- a trained model pickle
- `leaderboard.csv` with validation and training losses
- `best_model.json` describing the best configuration in the tested grid

## Honest status

This is a lightweight public model-selection layer centered on the shared `meso_uq.surrogate` core.

It should be understood as a practical public selection surface, not as a claim that the full upstream non-paper architecture-search machinery has already been imported verbatim.
