# MesoUQ GPU validation gate

This page defines the closeout gate for MES-160.

The rule is simple: after each issue in this slice, the full local validation matrix must run
on a GPU-backed platform, failures must be debugged or adapted, and the evidence must record
the command, commit, platform/GPU context, failures, fixes, and rerun.

Do not run the GPU matrix as part of the documentation task. Document the gate, then execute
it separately when the issue owner is ready.

## Required evidence fields

Record all of the following:

- Linear issue id
- branch name
- commit SHA before the run and after any fix
- platform (`karolina` or `vega`)
- GPU type, node name, partition/QOS, and allocation details
- Python/runtime path used by the run
- exact command line
- run tag
- output root
- start and end time
- pass/fail result
- failure summary, if any
- fix summary, if any
- rerun command, if any
- report paths and log paths

## Karolina commands

Use the Karolina sbatch launch for the site-default GPU gate:

```bash
RUN_TAG=<tag> OUTPUT_ROOT=/scratch/project/eu-26-17/eubrieucb/mesouq/runs/validation_matrix/<tag> PYTHON_BIN=/scratch/project/eu-26-17/eubrieucb/mesouq/runtime/venv/bin/python sbatch --export=ALL,REPO_ROOT="$PWD",RUN_TAG="$RUN_TAG",OUTPUT_ROOT="$OUTPUT_ROOT",PYTHON_BIN="$PYTHON_BIN" scripts/platforms/karolina/sbatch/validation_matrix.sbatch
```

If you are already inside an allocation, run the matrix directly:

```bash
python scripts/platforms/karolina/run_validation_matrix.py --experiments compression indentation --model-families full-model reduced-model --output-root /scratch/project/eu-26-17/eubrieucb/mesouq/runs/validation_matrix/<tag> --site karolina --run-tag <tag> --phase2-cpu-ranks 4 --python-bin /scratch/project/eu-26-17/eubrieucb/mesouq/runtime/venv/bin/python --skip-release-manifest
```

## Vega equivalent

The Vega validation matrix is documented in `docs/VEGA_VALIDATION_MATRIX.md`. Its public
command is:

```bash
python scripts/platforms/vega/run_validation_matrix.py \
  --experiments compression indentation \
  --model-families full-model reduced-model \
  --output-root _runs/vega/validation_matrix/<tag> \
  --phase2-cpu-ranks 4
```

Use the Vega sbatch template from the same doc when the matrix should be submitted rather than
run interactively.

## Failure handling

If a run fails:

1. keep the failing command and the raw logs
2. identify whether the failure is launcher, environment, workflow, or artifact related
3. adapt the code or the launcher only after the failure mode is understood
4. rerun the same matrix from the fixed commit or fixed environment
5. archive both the failure evidence and the rerun evidence

The closeout is not complete until the rerun passes and the evidence bundle is updated with
the corrected command, the corrected commit, and the final report path.

## Evidence references

- `docs/VALIDATION_MATRIX.md`
- `docs/VEGA_VALIDATION_MATRIX.md`
- `docs/KAROLINA_FULL_PLATFORM.md`
- `docs/VEGA_ACCEPTANCE_CHECKLIST.md`
- `tests/test_validation_matrix.py`
- `tests/test_karolina_validation_matrix.py`
