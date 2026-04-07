# Validation config bundle

This directory contains the dedicated tiny validation configs used for acceptance, smoke workflows, and reduced end-to-end testing.

## Purpose

These configs are derived from the canonical production configs already shipped in `MesoUQ`, but with much smaller population settings and shorter generation limits so that:

- real public workflow scripts can be exercised in reduced mode,
- Vega/operator acceptance can run from a clean clone without mutating the production configs,
- future pytest integration tests can target real public entrypoints using these same validation YAMLs.

## Files

### Full workflows

- `inference/configs/validation/validation_config_compression.yaml`
- `inference/configs/validation/validation_config_indentation.yaml`

### Reduced workflows

- `reduced/configs/validation/validation_config_compression.yaml`
- `reduced/configs/validation/validation_config_indentation.yaml`

## Design rules

These configs are intentionally conservative:

- they preserve the public workflow semantics,
- they keep the same data paths and experiment structure,
- they only reduce expensive population/generation knobs,
- they remain readable enough to be edited manually during acceptance/debug loops.

## Current usage

These configs are the default config layer used by the Vega acceptance command and by the GPU validation runner unless an explicit `--config-override` is provided.

If an operator passes `--population-size`, that explicit override still rewrites the workflow population keys in the derived config. Otherwise the committed validation YAML values are preserved as-is.
