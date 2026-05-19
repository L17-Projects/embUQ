# Active Learning Engine Contracts and Dry-Run Boundary

This module introduces a lightweight active-learning orchestration layer for experimentation,
with no direct dependency on HPC, GPU, Mirheo, Korali, or Pyro.

- **Orchestration (this package)** manages only:
  - candidate generation contract objects
  - acquisition scoring contracts
  - simulation request/result contracts
  - retraining request/result contracts
  - loop state and stopping criteria
  - deterministic dry-run execution of injected callbacks
  - failure capture and JSON serialisation for resumability

- **Platform execution is external** and owns:
  - simulation runtimes / queues
  - ML model training infrastructure
  - experiment storage backends
  - any GPU/HPC scheduling or container execution

The engine only calls caller-provided callables:

- `candidate_generator(state) -> Sequence[candidate]`
- `acquisition_policy(candidates, state) -> Sequence[score]`
- `simulator(requests, state) -> Sequence[result]`
- `retrainer(request, state) -> result`

Nothing in the orchestration layer imports or invokes heavyweight
compute backends, and no subprocesses are launched.

## Serialization contract

All loop state is intentionally JSON-compatible:

- primitives (`str`, `int`, `float`, `bool`, `None`)
- nested mappings/lists with string keys

`LoopState` (plus candidate and request/result contracts) provide `as_dict`,
`from_dict`, `to_json`, and `from_json` helpers so state can be written to disk,
restored, and resumed in a fresh process.

## Validation and failure capture

- Candidates are validated before simulation.
- Validation failures create `FailureRecord` entries with `stage="validation"`.
- Simulation/retraining errors are captured as result status `"failed"` or explicit
  `FailureRecord` entries where appropriate.

The dry-run engine keeps a full execution trail in state (`simulation_requests`,
`simulation_results`, `retraining_results`, and `failures`) for post-run inspection and resumability.

## GV selected-candidates handoff

GV numerical data generation uses a loose boundary between active learning and
the DPD launch stack:

- active learning owns candidate generation, acquisition scoring, selection,
  and loop state
- GV owns conversion from selected candidates into validated launch requests,
  scheduler-ready scripts, and expected HDF5 dataset paths
- operators remain responsible for submitting rendered scheduler scripts

Use `meso_uq.structures.gv.build_gv_active_learning_launch_handoff` for this
boundary. It accepts selected `Candidate` objects or candidate dictionaries and
builds `GVLaunchRequest` objects through the existing GV launch validator. The
candidate payload may contain only scientific launch fields:

- `experiment`
- `material_parameters`
- `geometry` or `radGV` plus `height`
- `controls`

Defaults may provide shared scientific fields. `controls` defaults are merged
with candidate controls, so common fixed controls can live at the batch boundary
while each candidate supplies its own sweep axis. Candidate control names take
precedence if a key is present in both places.

Scheduler and provenance fields (`platform`, `output_root`, `walltime`,
`gpu_count`, and `provenance_tags`) are adapter-owned and must be passed to the
handoff builder. This keeps acquisition code independent from platform launch
policy.

The optional render step is
`meso_uq.structures.gv.render_gv_active_learning_launch_handoff`. It delegates
to the GV launch renderer and writes manifests/scripts only; it does not submit
jobs. See `configs/active_learning/gv_selected_candidates.example.yaml` for a
two-candidate GV input example and
`configs/active_learning/gv_selected_candidates_handoff_manifest.example.json`
for the derived, non-submitting handoff manifest with expected HDF5 dataset
paths.

## Final-gate specification

The active-learning final-gate target for the current workstream is documented in
`ACTIVE_LEARNING_FINAL_GATE.md`.

That final-gate document is the single source of truth for:

- run target and scope
- AL rounds and per-round budget
- comparator and sample-seeding rules
- production runtime fingerprint
- per-step plot checks and final pass/fail rules
- Linear/vault-ready evidence sequencing
