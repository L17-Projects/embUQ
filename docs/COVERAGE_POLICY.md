# Coverage Policy

This policy defines the repository coverage contract for the package-centered
architecture.

## Coverage scope

- `src/meso_uq` is the canonical package source tree.
- `compression`, `indentation`, `inference`, `reduced`, `propagation`, and
  `scripts` are compatibility roots that remain in scope because they still
  carry public or operational entrypoints.
- `tests`, `docs`, `extern`, `papers`, notebook checkpoints, generated output
  trees, and runtime artifacts are excluded from line coverage accounting.

## Test layers

- Unit tests cover package contracts, pure helpers, and import-light logic.
- Integration tests cover cross-module behavior, legacy shims, and workflow
  assembly that spans multiple roots.
- Operational tests cover platform launchers, HPC-facing wrappers, and site
  routing helpers.
- Regression tests cover fixed behavior in coverage gates, release evidence,
  and compatibility surfaces that are intentionally kept stable.

Coverage reporting should measure the product code exercised by those tests,
not the execution of the tests themselves or of generated artifacts.

## Codecov policy

Codecov stays relaxed in this slice:

- project status target remains `auto`
- project threshold remains `0%`
- patch status target remains `auto`
- patch threshold remains `0%`
- patch status is informational

That keeps the feedback loop usable while the package and its compatibility
surfaces are still evolving, and it avoids turning GPU or HPC execution into a
hard coverage gate.

## Future tightening path

When the package surface stabilizes further, tighten coverage in this order:

1. keep the canonical `src/meso_uq` package fully visible in coverage reports
2. trim compatibility-root exceptions only after their wrappers are retired or
   converted to thin shims
3. raise Codecov thresholds after the regular CI path reliably exercises the
   same package and compatibility roots on CPU-only runners
4. move any remaining platform-specific or generated helpers into dedicated
   operational checks rather than broad line-coverage policy

The intent is to ratchet coverage policy without forcing GPU or HPC resources
into every feedback cycle.
