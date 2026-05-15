# EMB Numerical Experiment Assets

`emb/` contains elastic microbubble numerical experiment assets. It is not the reusable package layer; shared package behavior belongs under `src/meso_uq`.

Canonical shape:

```text
emb/compression/{src,evalkit,surrogate}
emb/indentation/{src,evalkit,surrogate}
```

Use `src/` for experiment-specific Mirheo generation/equilibration templates, `evalkit/` for reference-data preparation and likelihood/evaluation helpers, and `surrogate/` for experiment-specific surrogate training, evaluation, curated small artifacts, and descriptors.

Do not add new root-level EMB experiment trees. Generated outputs, logs, temporary Mirheo state, and scratch data belong under `_runs/...`, external scratch/data roots, or documented artifact roots. Local `dir.md` placement guides are ignored by `**/dir.md` and are not source.
