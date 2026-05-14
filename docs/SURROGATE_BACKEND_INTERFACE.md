# Surrogate Backend Interface

`meso_uq.surrogates` is a lightweight, backend-neutral contract layer introduced for MES-157.
It defines only metadata structures and resolution behavior; it does not load models or depend on torch/pyro.

## Core contract types

- `SurrogateBackendMetadata`
  - `backend`: `ModelBackend` identifier (for example `dnn`, `bnn`, `pyro_bnn`)
  - `supported_agent_families`: `emb`/`gv`
  - `supported_modalities`: `compression`, `indentation`, ...
  - `runtime_requirements`: `RuntimeRequirement` declarations
  - `supports_epistemic`, `supports_aleatoric`, `supports_surrogate_error`
- `SurrogatePrediction`
  - Always provides deterministic `mean`
  - Optionally includes `epistemic`, `aleatoric`, and `surrogate_error`
- `SurrogateTrainingRequest`
- `SurrogateLoadRequest`
- `SurrogateCheckpoint` and `SurrogateCheckpointMetadata`

## Registry contract

`register_surrogate_backend` stores backend implementations in an in-memory registry.
`resolve_surrogate_backend`:
- resolves by backend id
- optionally checks family/modality support
- enforces required dependencies

Dependency semantics:
- `RequirementState.REQUIRED` and `RequirementState.EXTERNAL` must be available
- `RequirementState.OPTIONAL` is reported but does not block resolution

The resolver accepts an optional `availability` map so callers can run deterministic checks in tests
(for example `{"torch": True}`).

## Example usage

```python
from meso_uq.surrogates import (
    SurrogateBackendMetadata, register_surrogate_backend, resolve_surrogate_backend
)

backend = MySurrogateBackend(...)
register_surrogate_backend(backend)

resolved = resolve_surrogate_backend(
    "dnn",
    agent_family="emb",
    modality="compression",
    availability={"torch": True},
)
```

No heavy dependencies are required to import the package or run this resolution flow.
