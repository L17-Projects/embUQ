# Paper Reproduction Artifact Policy

MES-167 separates paper reproduction source material from generated research payloads. The repository keeps scripts, documentation, config templates, and tiny fixtures reviewable in git; heavy or generated paper artifacts live under external artifact roots and are referenced through manifests.

## Keep In Git

Track paper material that explains or recreates a result without embedding a large run product:

- source scripts and importable helpers
- plotting and replay configuration files
- small manifest examples and schema fixtures
- documentation that explains commands, expected inputs, environment assumptions, and provenance
- tiny test fixtures intentionally sized for unit tests

For the HUQ EMB paper surface, `papers/huq_emb/` may contain source scripts, config templates, manifest examples, and documentation. It must not become a storage location for completed campaigns.

## Keep Outside Git

Do not track generated or heavyweight paper outputs:

- generated figures, tables, and rendered reports
- stdout/stderr logs and scheduler logs
- posterior samples, chains, traces, checkpoints, and model weights
- MAP extraction outputs, Mirheo run directories, replay result trees, and copied external datasets
- local caches, scratch directories, and temporary campaign outputs

Those payloads belong under external `PAPER_DATA_ROOT`, under the configured MesoUQ runs root, on artifact storage, or in another explicitly managed data location. Reproduction docs may reference them through a manifest, but the payload itself should remain outside the repository.

## Manifest Handoff

Paper replay commands should take inputs from manifests rather than assuming generated data lives next to source scripts. A manifest entry should record enough information to reproduce or retrieve an artifact:

- artifact kind, such as `map_manifest`, `posterior_samples`, `figure`, `run_log`, `checkpoint`, or `report`
- path relative to `PAPER_DATA_ROOT`, or an external artifact URI
- producing command or workflow name
- source commit or release tag
- modality and dataset identifiers where applicable
- checksum when the artifact is intended to be reused verbatim

Use absolute local paths only in private run records. Shared manifests should prefer `PAPER_DATA_ROOT`-relative paths, `${MESOUQ_RUNS_ROOT}`-relative paths, or stable artifact URIs.

## Reproduction Workflow

1. Clone the repository and install the documented package extras.
2. Set `PAPER_DATA_ROOT` to a local checkout, mounted filesystem, or restored artifact bundle containing the heavy paper data.
3. Select the manifest for the intended replay.
4. Run the paper source script with explicit manifest and output arguments.
5. Write new figures, logs, samples, checkpoints, and temporary outputs under `PAPER_DATA_ROOT` or another ignored run directory, not under git-tracked source paths.

Small fixtures in source control should exercise parsing and plotting contracts only. They are not substitutes for the full paper data bundle.

## Frozen Editorial Exceptions

An editor-submitted package may be retained verbatim in Git when all of the
following hold:

- the user explicitly designates it as an immutable editorial baseline;
- its total size remains reviewable and below the agreed file-size limits;
- a tracked manifest binds every file path, size, and SHA-256 value;
- generated reruns continue to use external artifact storage; and
- the exception is limited to the frozen submission, not its working or build
  directories.

The current scoped exception is `papers/UQ_EMB/editor_submission/review2_v1/`.
