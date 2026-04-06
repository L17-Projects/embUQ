# PR21 Korali compileability notes

This PR21 slice improves the vendored Korali build surface by importing several files that the current top-level Meson build already expects but that were still missing from `extern/korali/`.

## Imported in this slice

- `extern/korali/source/meson.build`
- `extern/korali/source/engine.hpp`
- `extern/korali/source/korali.hpp`
- `extern/korali/python/korali/meson.build`
- `extern/korali/python/korali/__init__.py`
- `extern/korali/tools/build/build.py`
- `extern/korali/tools/build/get_header_directory.py`

## Why these matter

Before this slice, the vendored top-level `extern/korali/meson.build` already called into `source/` and `python/korali/`, but those sub-build files were absent in `MesoUQ`.

That meant the vendored subtree was not even structurally aligned with its own Meson entrypoint.

This PR restores that basic alignment.

## What is still missing for a credible compileable vendored backend

This slice does **not** yet prove that the vendored subtree is compileable.

Important remaining gaps include at least:

- the core `extern/korali/source/engine.cpp` implementation
- the remaining source subtrees referenced by `source/meson.build`, such as `auxiliar`, `sample`, `variable`, and the broader module build glue
- the Python subdirectories referenced by `python/korali/meson.build`, such as `plot`, `profiler`, `rlview`, and `cxx`
- a documented and manually validated supported build recipe executed against the vendored subtree as it exists in `MesoUQ`

## Honest status after this slice

After this PR, the vendored Korali subtree is more internally coherent at the build-entrypoint level, but it should still be treated as an **incomplete compileability pass** rather than as a finished compile-validated backend.
