# Configuration Policy

This slice introduces a central `configs/` skeleton for additive, schema-versioned examples only. It does not migrate historical runtime configs under `inference/`, `reduced/`, or other legacy trees.

Each example config in `configs/` follows the same minimum identity contract:

- `schema_version`
- `kind`
- `metadata.id`
- `metadata.name`

Platform configs additionally carry a `platform_key` and `path_policy`. The policy is placeholder-first:

- use `${MESOUQ_REPO_ROOT}`, `${MESOUQ_RUNS_ROOT}`, `${MESOUQ_SCRATCH_ROOT}`, and similar env-backed placeholders
- avoid embedding user-specific absolute paths in checked-in config examples
- keep platform-specific overrides additive rather than rewriting historical operator configs

Examples/configs refresh-or-archive policy:

- refresh an example when it still represents an active workflow contract
- archive an example when the workflow or schema is no longer current, rather than silently mutating it into a different meaning
- keep archived examples schema-versioned and clearly named so old evidence remains explainable

The initial skeleton covers representative stubs for:

- agents
- modalities
- datasets
- surrogates
- inference
- noise
- active learning
- platforms
- reports
