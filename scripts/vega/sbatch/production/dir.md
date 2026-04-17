# production

Production-scale multi-node Vega orchestration scripts for end-to-end
`phase1 -> phase2 -> phase3b -> propagation phase3b` runs.

This directory contains:
- complete workflow submitters for each experiment/model-family lane
- per-phase child sbatch scripts used by the complete submitters
- validation-profile variants for `dev` partition smoke workflows
