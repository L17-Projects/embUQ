# Table S7 PCA replay

This directory restores the missing PCA analysis surface for the
`review2_v1` supplementary Table S7. The tracked input tables and the five
analysis scripts are byte-identical copies from the accepted reproduction
bundle; their hashes and bundle-relative origins are recorded in
[`SOURCE_MANIFEST.json`](../../../scripts/workflows/emb/uq_emb/pca_table_s7/SOURCE_MANIFEST.json).

## Required data layout

Never analyze inside the read-only bundle. Assemble each `d1`--`d6` case in a
fresh scratch directory with this layout:

```text
cases/dN/
├── selected_map_row.json          # optional; generated from the tracked map table
└── case/
    ├── xyz0.xyz
    ├── mesh/emb00001.off
    ├── parameter/
    │   ├── parameters-default00001.yaml
    │   └── parameters00001.yaml
    └── output/positions.xyz
```

The compact mesh, parameters, and accepted first-300-mode products come from:

```text
recovery-v2/vega_pca_table_s7/latest_map_campaign/cases/dN/
```

The complete 40,000-frame trajectory, `xyz0.xyz`, default parameters, and full
eigenvectors come from:

```text
recovery-v4/vega_pca_table_s7/accepted_complete_pca/cases/dN/case/
```

The two recovery layers together are the complete PCA input contract.

## Run

Install `.[hpc,plot]` in a Python 3.10+ MesoUQ environment. On Karolina, use a
verified environment created from `/usr/bin/python3.11`:

```bash
export MESOUQ_PYTHON="${MESOUQ_ENV_ROOT}/bin/python"
export PCA_CASE="${MESOUQ_SCRATCH_ROOT}/papers/UQ_EMB/replay/pca/cases/d1/case"

"${MESOUQ_PYTHON}" scripts/workflows/emb/uq_emb/pca_table_s7/run_replay.py \
  --case "${PCA_CASE}"
```

That command only validates the case and prints the site-neutral plan. Run the
expensive trajectory analysis in a suitable CPU allocation by adding
`--execute`. Add `--cross-check` to repeat the independent breathing-subspace
calculation. The runner refuses to overwrite existing outputs and writes a JSON
receipt below `case/provenance/`.

The accepted zero-based modes for `d1`--`d6` are `72, 56, 104, 20, 41, 21`;
their Table S7 frequencies are `3.188, 3.178, 2.974, 2.161, 2.006, 1.231` MHz.

Recreating the trajectories themselves additionally requires Mirheo, CUDA,
OpenMPI, and a site scheduler. Mirheo is an external dependency configured by
`MESOUQ_MIRHEO_SRC`; no Karolina-specific PCA scheduler wrapper is bundled.
