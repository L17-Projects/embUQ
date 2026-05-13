# MES-125 Karolina acceptance closeout

This page records the acceptance-evidence structure for the MES-125 Karolina closeout slice.
It is intentionally narrow: it tracks the evidence bundle, the open input gaps, and the
machine-readable fields that still need to be filled in by the other closeout agents.

## Platform boundary

Karolina acceptance for MES-125 does not depend on mounting `/ceph/hpc/home/eubrieucb`.
That path is treated as a non-goal for Karolina and must not be recorded as an expected mount
or a required platform capability.

## Paper-data boundary

The exact HUQ-EMB `paper_data` used for paper replay is not present on Karolina.
For this closeout, that missing dataset is an unavailable input to replay evidence, not a
platform blocker. Record it as an input gap and not as a Karolina defect.

## GV paper evidence now available

The workspace contains the source PDFs:

- `/home/it4i-bbenvegnen/workspace/EMB_GV_DPD.pdf`
- `/home/it4i-bbenvegnen/workspace/an5c02783_si_001.pdf`

They have also been staged into Karolina scratch provenance:

- `/scratch/project/eu-26-17/eubrieucb/mesouq/provenance/gv_paper/EMB_GV_DPD.pdf`
- `/scratch/project/eu-26-17/eubrieucb/mesouq/provenance/gv_paper/an5c02783_si_001.pdf`

## Required closeout fields

The evidence manifest and review notes should carry placeholders or final values for:

- doctor
- EMB evidence
- GV evidence
- tests
- Linear issue closure
- PR / CI / Codecov / review sweep

## Local validation captured on 2026-05-13

Local validation used the Karolina scratch runtime Python after sourcing
`scripts/platforms/karolina/env_karolina.sh`. For default/Vega-compatibility tests,
MesoUQ runtime-site variables were unset after loading the module/library environment.

- Focused changed-surface suite with default site variables unset: `177 passed`.
- Karolina-env focused suite: `5 passed`.
- Full repository suite with default site variables unset: `1748 passed, 9 skipped`.
- Docs/manifest workflow checks: `8 passed`.
- Shell syntax: `bash -n` passed for the Karolina env script and GV paper-replay submit/sbatch scripts.
- JSON syntax: `python3.11 -m json.tool` passed for the MES-125 example manifest.

Known local skips are optional-dependency/runtime skips for `pyro` and import-time `mirheo`
unit checks; they are not introduced by this closeout slice.

## Suggested evidence package

Use one release-root bundle containing at least:

- a machine-readable manifest
- a doctor report
- EMB acceptance or replay evidence
- GV evidence
- test outputs or test references
- Linear issue closeout references
- PR, CI, Codecov, and review-thread sweep references

## Status notes

- Karolina platform availability is the target; the `/ceph/hpc/home/eubrieucb` mount is not.
- HUQ-EMB `paper_data` exact replay remains blocked by unavailable input, not by the platform.
- The PDFs above are present and can be linked from the evidence bundle.
