# Promoted dev / operator utilities

This note summarizes the higher-value `_dev` / operator-facing utilities that are now publicly visible in `MesoUQ`.

## Purpose

These utilities are not the narrow paper-facing core. They exist because they improve:

- runnability,
- diagnosis,
- operator convenience during common scientific loops.

## Key promoted utilities

### 1. Workflow validation suite

`inference/scripts/run_gpu_validation_suite.py`

Use this when you want to run a packaged reduced/full validation sequence and collect workflow artifacts.

### 2. Richer indentation multi-architecture trainer

`indentation/surrogate/scripts/train_multi_arch.py`

Use this when you want a more diagnostic architecture-comparison loop than the lightweight public model-selection wrapper. This script keeps:

- curve cleaning heuristics,
- rupture filtering,
- multi-architecture summary outputs,
- automatic BEST-model deployment.

## Relation to the lighter public surfaces

These promoted utilities complement, rather than replace, the simpler public interfaces added earlier:

- the lightweight surrogate model-selection wrappers,
- the lightweight workflow/propagation execution surfaces,
- the MAP and plotting utilities.

## Honest status

These utilities are public because they are useful and improve operator experience, but they are still closer to an operator / analysis layer than to the minimal paper-facing core.
