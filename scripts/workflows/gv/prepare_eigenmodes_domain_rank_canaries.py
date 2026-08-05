#!/usr/bin/env python3
"""Render scheduler-ready GV eigenmodes domain-rank canary scripts.

The generated scripts are approval-only artifacts. This workflow never submits
Slurm jobs.
"""

from __future__ import annotations

import argparse
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "_runs" / "gv" / "eigenmodes_domain_rank_canary"
FIGURE_REPLAY_ROOT = REPO_ROOT / "_runs" / "gv" / "figure_replay"
DEFAULT_CASES = (
    ("gpu1_ranks_1x1x1", 1, "1,1,1", "02:00:00"),
    ("gpu2_ranks_2x1x1", 2, "2,1,1", "02:00:00"),
    ("gpu4_ranks_2x2x1", 4, "2,2,1", "03:00:00"),
)

_SITE_POLICY: dict[str, dict[str, Any]] = {
    "karolina": {
        "account": "#SBATCH --account=eu-26-17",
        "partition": "qgpu",
        "cpus_per_task": 8,
        "gpus_line": "#SBATCH --gpus={gpu_count}",
        "module_preamble_lines": [
            "if ! type module >/dev/null 2>&1; then",
            "  source /apps/all/Lmod/8.7.37/lmod/lmod/init/bash",
            "fi",
        ],
        "module_lines": [
            "module --force purge || module purge || true",
            "module load NVHPC/24.1-CUDA-12.4.0",
            "module load OpenMPI/4.1.6-NVHPC-24.1-CUDA-12.4.0",
            "module load HDF5/1.14.3-NVHPC-24.1-CUDA-12.4.0",
            "module load UCC-CUDA/1.3.0-GCCcore-12.2.0-CUDA-12.4.0",
            "module load Python/3.10.8-GCCcore-12.2.0",
            "module load CMake/3.24.3-GCCcore-12.2.0",
        ],
        "extra_lines": [],
    },
    "vega": {
        "account": "",
        "partition": "gpu",
        "cpus_per_task": 4,
        "gpus_line": "#SBATCH --gres=gpu:{gpu_count}",
        "module_preamble_lines": [],
        "module_lines": [
            "module purge",
            "module load \\",
            "  Python/3.10.8-GCCcore-12.2.0 \\",
            "  OpenMPI/4.1.4-GCC-12.2.0 \\",
            "  CUDA/12.2.2 \\",
            "  GSL/2.7-GCC-12.2.0 \\",
            "  Eigen/3.4.0-GCCcore-12.2.0 \\",
            "  HDF5/1.14.0-gompi-2022b \\",
            "  MPFR/4.2.0-GCCcore-12.2.0 \\",
            "  GMP/6.2.1-GCCcore-12.2.0",
        ],
        "extra_lines": ["#SBATCH --exclude=gn10"],
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--site", choices=("karolina", "vega"), default="karolina")
    parser.add_argument(
        "--include-4gpu",
        action="store_true",
        default=False,
        help="Include optional 4 GPU / ranks=(2,2,1) canary.",
    )
    return parser


def _campaign_id() -> str:
    return "gv-eigenmodes-domain-rank-canary-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _case_selection(include_4gpu: bool) -> tuple[tuple[str, int, str, str], ...]:
    return DEFAULT_CASES if include_4gpu else DEFAULT_CASES[:2]


def _domain_rank_product(domain_ranks: str) -> int:
    product = 1
    for token in domain_ranks.replace("x", ",").split(","):
        value = int(token.strip())
        if value <= 0:
            raise ValueError("Domain ranks must be positive.")
        product *= value
    return product


def _mpi_rank_count(domain_ranks: str) -> int:
    return 2 * _domain_rank_product(domain_ranks)


def _write_sbatch(
    *,
    path: Path,
    campaign_id: str,
    case_id: str,
    output_root: Path,
    site: str,
    gpu_count: int,
    domain_ranks: str,
    walltime: str,
) -> None:
    mpi_ranks = _mpi_rank_count(domain_ranks)
    replay_campaign_id = campaign_id + "-" + case_id
    lane_root = FIGURE_REPLAY_ROOT / replay_campaign_id
    policy = _SITE_POLICY[site]

    lines: list[str] = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name=gv-eig-{case_id}",
    ]
    account = str(policy["account"])
    if account:
        lines.append(account)

    lines.extend(
        [
            f"#SBATCH --partition={policy['partition']}",
            "#SBATCH --nodes=1",
            f"#SBATCH --ntasks={mpi_ranks}",
            f"#SBATCH --cpus-per-task={policy['cpus_per_task']}",
            str(policy["gpus_line"]).format(gpu_count=gpu_count),
            "#SBATCH --mem=128G",
        ]
    )
    for extra_line in policy["extra_lines"]:
        lines.append(str(extra_line))
    lines.extend(
        [
            f"#SBATCH --time={walltime}",
            f"#SBATCH --output={shlex.quote(str(output_root / case_id / 'slurm-%j.out'))}",
            f"#SBATCH --error={shlex.quote(str(output_root / case_id / 'slurm-%j.err'))}",
            "",
            "set -euo pipefail",
            "",
            f'REPO_ROOT="${{REPO_ROOT:-{REPO_ROOT}}}"',
            'cd "${REPO_ROOT}"',
            'if [[ -z "${MESOUQ_SITE_RUNTIME_ROOT:-}" ]]; then',
            '  echo "MESOUQ_SITE_RUNTIME_ROOT must be set before rendering this canary script." >&2',
            "  exit 2",
            "fi",
            "",
        ]
    )
    lines.extend([str(line) for line in policy["module_preamble_lines"]])
    lines.extend([str(line) for line in policy["module_lines"]])
    lines.extend(
        [
            "",
            'source "${REPO_ROOT}/scripts/platforms/hpc/site_env.sh"',
            f'mesouq_activate_site_env {site} "${{REPO_ROOT}}"',
            "",
            "export MESOUQ_GV_EIGENMODES_PROFILE=canary",
            f"export MESOUQ_GV_EIGENMODES_DOMAIN_RANKS={shlex.quote(domain_ranks)}",
            f"export MESOUQ_GV_EIGENMODES_MPI_RANKS={mpi_ranks}",
            "export MESOUQ_GV_EIGENMODES_MODE_WINDOW_POLICY=frequency-min",
            "export MESOUQ_GV_EIGENMODES_MODE_MIN_FREQUENCY=22.5",
            "export MESOUQ_GV_EIGENMODES_TIMEOUT_SECONDS=7200",
            "",
            '"${PYTHON_BIN}" scripts/workflows/gv/run_paper_figure_replay.py \\',
            f"  --campaign-id {shlex.quote(replay_campaign_id)} \\",
            f"  --output-root {shlex.quote(str(lane_root))} \\",
            "  --lane eigenmodes \\",
            "  --paper-exact \\",
            "  --eigenmodes-profile canary",
        ]
    )

    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o750)


def _manifest(
    *,
    campaign_id: str,
    output_root: Path,
    site: str,
    cases: tuple[tuple[str, int, str, str], ...],
) -> dict[str, Any]:
    case_payloads: list[dict[str, Any]] = []
    for case_id, gpu_count, domain_ranks, walltime in cases:
        mpi_ranks = _mpi_rank_count(domain_ranks)
        case_root = output_root / case_id
        replay_campaign_id = campaign_id + "-" + case_id
        submission_command = f"sbatch {shlex.quote(str(case_root / f'{case_id}.sbatch'))}"
        case_payloads.append(
            {
                "case_id": case_id,
                "site": site,
                "gpu_count": gpu_count,
                "domain_ranks": domain_ranks,
                "mpi_ranks": mpi_ranks,
                "runtime_profile": "canary",
                "paper_replay_runtime_path": True,
                "paper_replay_campaign_id": replay_campaign_id,
                "expected_walltime": walltime,
                "output_root": str(FIGURE_REPLAY_ROOT / replay_campaign_id),
                "scheduler_script": str(case_root / f"{case_id}.sbatch"),
                "submission_command": submission_command,
                "submission_commands": [submission_command],
            }
        )
    return {
        "schema": "mesouq.gv.eigenmodes.domain_rank_canary.v1",
        "campaign_id": campaign_id,
        "site": site,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "campaign_root": str(output_root),
        "submitted": False,
        "submission_policy": "Approval required; scripts are rendered only.",
        "cases": case_payloads,
        "acceptance_criteria": {
            "scheduler": "Slurm job exits successfully with no non-finite log scan failures.",
            "runtime_profile": "parameter/eigenmodes_runtime_profile.json records profile=canary.",
            "domain_rank_trace": "stdout and runtime profile record requested MESOUQ_GV_EIGENMODES_DOMAIN_RANKS.",
            "mode_window": {
                "manifest": "analysis/output/mode_window_manifest.json exists.",
                "policy": "frequency-min",
                "final_mode_count": 30,
                "raw_eigenpair_count_min": 30,
                "min_final_frequency_tau_inv": 22.5,
                "raw_head_low_frequency_modes_not_used": True,
            },
            "figure8g_diagnostic": (
                "Figure 8(g) acceptance metrics are computed and recorded by paper-exact "
                "postprocess; short canaries use this diagnostically, while full paper "
                "runs must satisfy the documented tolerances."
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    campaign_id = args.campaign_id or _campaign_id()
    root = Path(args.output_root).expanduser().resolve() if args.output_root else (DEFAULT_OUTPUT_ROOT / campaign_id).resolve()
    site = args.site
    cases = _case_selection(bool(args.include_4gpu))
    root.mkdir(parents=True, exist_ok=True)
    for case_id, gpu_count, domain_ranks, walltime in cases:
        _write_sbatch(
            path=root / case_id / f"{case_id}.sbatch",
            campaign_id=campaign_id,
            case_id=case_id,
            output_root=root,
            site=site,
            gpu_count=gpu_count,
            domain_ranks=domain_ranks,
            walltime=walltime,
        )
    manifest = _manifest(campaign_id=campaign_id, output_root=root, site=site, cases=cases)
    manifest_path = root / "gv_eigenmodes_domain_rank_canary_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"GV eigenmodes domain-rank canary manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
