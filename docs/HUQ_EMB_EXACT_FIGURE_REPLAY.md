# HUQ-EMB Exact Figure Replay

Use this workflow to recreate the paper-facing HUQ-EMB figures and tables from a stored `paper_data` campaign in one command.

## Command

```bash
python scripts/workflows/emb/huq_emb/run_exact_uqdpd_asset_port.py \
  --paper-data-root /path/to/paper_data \
  --campaign-id <campaign_id>
```

If `paper_data/runs/` contains exactly one campaign directory, `--campaign-id` is optional and the runner will infer it.

## What it does

The command:

1. reuses the stored workflow outputs under `paper_data/runs/<campaign_id>/workflow_matrix/`
2. stages the DNN grouped-holdout and Sobol figure inputs automatically
3. runs the exact paper generators under `papers/huq_emb/`
4. copies publishable outputs into:
   - `paper_data/figures/main/`
   - `paper_data/figures/supplementary/`
   - `paper_data/tables/`
5. writes a replay report to:
   - `paper_data/runs/<campaign_id>/paper_exact_stage/run_exact_uqdpd_asset_port.report.json`

The report records the staging manifest path, rendered output roots, copied assets, logs, and any hard failures caused by missing required figures or tables.

## TeX behavior

The wrapper defaults to non-TeX matplotlib rendering unless one of these is provided:

- `--texdeps-dir /path/to/texdeps`
- `MESOUQ_PAPER_TEXDEPS_DIR=/path/to/texdeps`
- a repo-local `papers/huq_emb/_texdeps/` directory

This keeps the replay command portable on machines that do not have the original Vega/UQ_DPD TeX environment available.

## Useful flags

- `--force`: rerun DNN staging and rebuild the exact replay stage from scratch
- `--staging-device cpu|cuda`: choose the device used for grouped-holdout/Sobol staging
- `--skip-supplementary`: development-only partial mode; skips supplementary figure generation and validates only the main-figure subset plus non-supplementary tables
