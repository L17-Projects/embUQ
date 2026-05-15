# GV paper replay closeout

This page is the closeout matrix for the GV-only replay work in the Linear
project `GV Numerical Data Generation`. It covers regenerated DPD evidence for
the GV figures in `/home/it4i-bbenvegnen/workspace/EMB_GV_DPD.pdf`.

EMB figures are excluded from this project. GV `shear_flow` / Figure 9 is also
excluded from runtime replay and remains deferred in `GV_SHEAR_FLOW_DEFERRAL.md`.

## Source provenance

Paper and SI PDFs are staged both in the workspace and Karolina scratch
provenance area:

| Artifact | Path | SHA-256 |
| --- | --- | --- |
| Main paper PDF | `/home/it4i-bbenvegnen/workspace/EMB_GV_DPD.pdf` | `20eb25ba752f5daea116ffcf9a382d0ea6c0e9286ed7744d9a8544b6fcfae182` |
| SI PDF | `/home/it4i-bbenvegnen/workspace/an5c02783_si_001.pdf` | `6559522c2d7e2d428b5f190815447b8e421fb5190f29bf3312c1f8db61d8f97e` |
| Main paper PDF, Karolina scratch copy | `/scratch/project/eu-26-17/eubrieucb/mesouq/provenance/gv_paper/EMB_GV_DPD.pdf` | `20eb25ba752f5daea116ffcf9a382d0ea6c0e9286ed7744d9a8544b6fcfae182` |
| SI PDF, Karolina scratch copy | `/scratch/project/eu-26-17/eubrieucb/mesouq/provenance/gv_paper/an5c02783_si_001.pdf` | `6559522c2d7e2d428b5f190815447b8e421fb5190f29bf3312c1f8db61d8f97e` |
| Current canonical runtime default source | `gv/stretching/src/parameters-default.gv.yaml` | `7a7ff7e98b1ce03cd39ce25c06a7ce4769e26a8a7d24fb6958da95a0b1630715` |

The replay profile currently remains `runtime_default`, not
`paper_confirmed`. The blocker is that the exact paper-confirmed GV geometry and
nine-material-parameter tuple have not been machine-extracted from the paper/SI
or staged dropped scripts. Replay packets must therefore report the profile
status and canonical runtime source instead of silently treating defaults as
paper constants.

Dropped GV paper scripts are not part of the checked-in source tree. When a
dropped-script protocol is referenced, the replay packet must state whether it
is only a protocol reference or whether imported runtime behavior is being used.

## Figure inventory and acceptance matrix

| Target | MesoUQ lane | Observable packet | Controls / sweep | Output pattern | Acceptance gate |
| --- | --- | --- | --- | --- | --- |
| Figure 3, GV stretching | `stretching` | Stress-strain packet: `sigma_zz` versus `epsilon_zz`, `minus_epsilon_phi`; auxiliary force-displacement channels when present | `tot_force` sweep with fixed `bpress=-91.0` for the paper replay profile | `_runs/gv/figure_replay/<campaign>/lanes/stretching/plots/*` and `outputs/stretching_summary.json` | Finite packet ready for qualitative user comparison against Figure 3; not an automated scientific pass |
| Figure 7, GV pressure buckling | `buckling` | Relative volume `V/V0` versus pressure difference `Delta_p`, plus morphology/lobe-transition notes when available | `buck` sweep with fixed `bpress=-91.0`; pressure mapping `Delta_p = 90.9 * buck` | `_runs/gv/figure_replay/<campaign>/lanes/buckling/plots/buckling_relative_volume*.png` and `outputs/buckling_summary.json` | Corrected paper-winding and dropped-fluid protocol must be recorded; packet must identify lobe-transition evidence or document divergence |
| Figure 8, GV eigenmodes | `eigenmodes` | First-30 eigenfrequency spectrum plus selected mode-shape and axial-profile evidence when eigenvectors/reference mesh are available | fixed `bpress=-91.0`; `mode_count=30` for paper-exact replay | `_runs/gv/figure_replay/<campaign>/lanes/eigenmodes/plots/eigenmode_spectrum.png` and `outputs/eigenmodes_summary.json` | Packet must be sufficient for user qualitative review; empty spectral data, non-finite data, and ambiguous missing mode ordering fail |
| SI-backed GV torsion diagnostic | `torsion` | `sigma_phi_r` versus `gamma = R0 * theta / H0_cyl` | `theta` sweep; paper-exact lane uses the SI-backed torsion parameter tuple | `_runs/gv/figure_replay/<campaign>/lanes/torsion/plots/*` and `outputs/torsion_summary.json` | Finite diagnostic packet ready for SI qualitative review |

All generated replay data stays under ignored `_runs/` or explicitly configured
scratch roots. Do not commit generated plots, runtime work directories, or
campaign outputs.

## Current evidence

The original four-lane paper replay campaign from PR `#125` produced four
comparison packets under `_runs/gv/figure_replay/gv-paper-replay-20260505T204439Z`.
That campaign is operational evidence, not final scientific acceptance:

- `stretching`: finite Figure 3 packet.
- `buckling`: finite but initially flat Figure 7 packet.
- `torsion`: finite SI diagnostic packet.
- `eigenmodes`: finite but too small for a complete Figure 8 first-30 review.

The stronger Figure 7 correction evidence is Linear `MES-119`:

- job `31861460`, partition `dev`, exit `0:0`, elapsed `00:28:55`;
- campaign `_runs/gv/figure_replay/gv-buckling-winding-dropped-12pt-20260507`;
- corrected pressure mapping `Delta_p = 90.9 * buck`;
- paper-winding normalization and dropped-fluid coupling requirement recorded;
- corrected 12-point curve reached `V/V0≈0.984` at `Delta_p≈30.989`,
  `0.859` at `43.384`, and `0.594` at `68.175`.

## Closeout state

The replay implementation is on current `main`, and the qualitative comparison
packet remains the final user acceptance gate. Current unresolved scientific
items are:

- profile promotion is blocked until the exact paper-confirmed constants are
  extracted or staged;
- Figure 7 must preserve the corrected winding/dropped-fluid protocol in the
  reproducible lane;
- Figure 8 still needs reviewable first-30 spectrum and mode-shape/axial-profile
  evidence.

Linear mapping:

- `MES-172`: this inventory and matrix.
- `MES-173`: profile status and provenance blocker.
- `MES-174` / `MES-113`: Figure 7 buckling hardening.
- `MES-175` / `MES-115`: Figure 8 eigenmode closeout.
- `MES-176` / `MES-116` / `MES-118`: current-main landing, qualitative gate, and
  lane-aware submitter policy.
