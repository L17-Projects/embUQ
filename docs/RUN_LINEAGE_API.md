# Run Lineage API

`src/meso_uq/orchestration/lineage.py` defines the first reusable run-lineage
contract for MesoUQ workflows.

Each orchestrated run should record:

- a stable run ID derived from a normalized config payload;
- code version and platform;
- stages with status, command, inputs, outputs, reuse, and failure state;
- artifact records for configs, checkpoints, posteriors, figures, reports, and logs;
- validation evidence, including GPU validation matrix report paths and Slurm job IDs.

The API is intentionally a manifest layer, not a scheduler. It supports local,
Karolina, Vega, generic Slurm, dry-run, resumed, and failed workflows without
requiring a database or job submission from default tests.

Absolute operational report paths may appear in validation evidence, but they
are marked as external path-policy metadata and should not become source-tree
artifact paths.
