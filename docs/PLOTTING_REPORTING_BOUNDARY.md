# Plotting and Reporting Boundary

Reusable plotting contracts live under `src/meso_uq/plotting/`; reusable report
manifest contracts live under `src/meso_uq/reporting/`.

The boundary rules are:

- basic package import must not import matplotlib or any rendering backend;
- plotting requests consume structured series and write manifestable figure paths;
- reports consume artifact IDs, run IDs, config digests, and section metadata;
- paper-specific layout and narrative scripts can remain in paper or script
  folders until explicitly migrated;
- generated figures and reports follow the artifact policy rather than becoming
  implicit source files.

Headless HPC execution should set `MPLBACKEND=Agg` before rendering. The
contract helper configures that environment value without importing matplotlib.
