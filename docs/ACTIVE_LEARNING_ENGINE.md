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
