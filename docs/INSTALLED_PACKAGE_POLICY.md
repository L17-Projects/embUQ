# Installed Package Policy

This slice checks the package as an installed wheel, not as a repo-local import.
The smoke helper builds or accepts a wheel, installs it into a temporary virtual
environment, and runs imports from outside the repository root.

## Local command

Typical local usage:

```bash
python scripts/qa/ci/run_installed_package_smoke.py
```

To reuse an existing wheel:

```bash
python scripts/qa/ci/run_installed_package_smoke.py --wheel dist/mesouq-0.1.0-py3-none-any.whl
```

The probe prunes repository-root and `src/` entries from `sys.path` before any
package import. Failures should report the concrete leaked path when an import
origin resolves back into the checkout.

The temporary environment installs `mesouq` from the wheel, not from an editable
checkout, while reusing the dependency environment prepared by CI or local
validation. Run it after the usual test-extra install and package build steps.

## Coverage

The smoke covers dependency-light package surfaces that are expected to remain
available in a wheel install:

- `meso_uq.public_api`
- registry modules under `meso_uq.agents`, `meso_uq.modalities`, and `meso_uq.structures`
- config loaders and aliases under `meso_uq.config` and `meso_uq.configs.policy`
- compatibility shims in `meso_uq.surrogate.compat`
- policy modules in `meso_uq.artifacts.policy`, `meso_uq.artifacts.relocation`, and `meso_uq.platforms.policy`
- Korali runtime policy in `meso_uq.platforms.korali_runtime`

## Compatibility exceptions

The smoke intentionally excludes modules whose import-time behavior is supposed to
pull optional heavy dependencies or runtime-only stacks. That includes the main
Torch-backed surrogate implementation, Pyro-backed BNN paths, and plotting entry
points such as:

- `meso_uq.surrogate.model`
- `meso_uq.surrogate.training`
- `meso_uq.surrogate.cli`
- `meso_uq.surrogate.bnn`
- `meso_uq.surrogate.group_holdout`
- `meso_uq.postprocess.plots`
- the plot-heavy GV replay lane modules

Those modules are still expected to work in their own feature-specific tests, but
they are not part of this installed-package lightweight import gate.
