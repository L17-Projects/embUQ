Purpose: Expose safe HUQ-EMB operator entrypoints that preserve compatibility while the repo layout evolves.
What belongs here: Thin wrappers for non-live HUQ-EMB paper-data orchestration that delegate into `papers/huq_emb/`.
What must not appear here: Vega full-rebuild launchers, active production sbatch wrappers, Mirheo runtime edits, or campaign outputs.
Key rules: Keep wrappers compatibility-only; delegate to maintained scripts under `papers/huq_emb`; do not add live Vega orchestration here without explicit governance and approval.
