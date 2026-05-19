# GV Eigenmodes Paper Profile

This note records the operational policy for reproducing GV Figure 8 eigenmodes
from fresh MesoUQ runtime outputs.

## Current Failure

The fresh shortened runtime writes raw covariance eigenpairs in descending
variance order from `gv/eigenmodes/src/analysis/all_analysis.py`. The previous
`trim_svd.sh` then copied the first 30 raw eigenpairs directly to
`eigvalues_new.txt`/`eigvectors_new.txt`.

That made postprocessing use low-frequency high-variance modes as Figure 8
paper modes. In the 2026-05-18 shortened run, raw mode 0 is about
`2.16 tau^-1`, while the dropped-archive/digitized Figure 8(g) spectrum starts
near `23.09 tau^-1`. Raw mode 10 is near that first paper value, but a fixed
`skip=10` is not sufficient: later archive modes map to non-contiguous raw
indices in the shortened run. The issue is therefore an explicit paper-mode
windowing problem, not a plotting-only problem.

## Runtime Profiles

`MESOUQ_GV_EIGENMODES_PROFILE=paper` is the full paper-grade profile:

- `numsteps=40000000`
- `numsteps_eq=500000`
- `stslik=200000`

`MESOUQ_GV_EIGENMODES_PROFILE=canary` is the short scaling profile:

- `numsteps=4000000`
- `numsteps_eq=50000`
- `stslik=20000`

`MESOUQ_GV_PAPER_EXACT=1` defaults to the paper profile unless
`MESOUQ_GV_EIGENMODES_PROFILE` is set explicitly. Both profiles are manifested
in `parameter/eigenmodes_runtime_profile.json`, including the configured
domain ranks, mode-window policy, frequency floor, and final 30 paper-mode
indices.

## Mode Window

The production trim step is `analysis/trim_eigenmodes.py`, called through
`trim_svd.sh`. It writes:

- `analysis/output/eigvalues_new.txt`
- `analysis/output/eigvectors_new.txt`
- `analysis/output/mode_window_manifest.json`

The mode-window manifest records the raw eigenpair count, selected raw
eigenpair indices, final mode indices, selected paper-mode indices, final mode
count, policy, and frequency floor.

Default paper-window policy:

- `MESOUQ_GV_EIGENMODES_MODE_WINDOW_POLICY=frequency-min`
- `MESOUQ_GV_EIGENMODES_MODE_MIN_FREQUENCY=22.5`

This keeps raw eigenpairs available while excluding slow global/large-scale
fluctuation modes below the paper shell-mode frequency window. For forensic
work only, `raw-head` and `explicit-indices` are available and are recorded in
the manifest.

## Acceptance

Paper-exact postprocessing records Figure 8(g) acceptance when runtime
mode-window metadata are present. The comparison target is the digitized
Figure 8(g) spectrum stored at:

`src/meso_uq/structures/gv/references/eigenmodes_fig8g_digitized.csv`

Default tolerances:

- mean absolute error <= `1.0 tau^-1`
- maximum absolute error <= `5.0 tau^-1`

The archive-backed replay was much tighter, with mean absolute error about
`0.067 tau^-1` and maximum absolute error about `0.143 tau^-1`; the operational
tolerance is wider to allow stochastic reruns while still rejecting wrong mode
windows.
