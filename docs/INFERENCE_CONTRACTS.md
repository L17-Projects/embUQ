# Inference Contracts

MesoUQ inference code now has a dependency-light contract layer in
`src/meso_uq/inference/contracts.py`.

The contract separates these concerns:

- agent family and modality selection;
- dataset and surrogate backend identity;
- prior declarations for population, individual, measurement, discrepancy, and surrogate-error layers;
- likelihood components and their required surrogate-uncertainty fields;
- sampler backend configuration and optional runtime requirements;
- posterior artifact manifests.

GV inference support that is not scientifically ready should be represented as
`unsupported`, with a concrete reason. This is preferable to letting a workflow
fail later through a missing script, hidden working-directory assumption, or
optional dependency import.

Sampler-specific requirements such as Korali or Pyro belong to
`SamplerBackendContract.requirements`. They must not be imported by the contract
layer. Execution adapters decide whether to skip, fail, or bootstrap those
runtimes.
