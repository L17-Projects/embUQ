#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.campaign_manifests import (  # noqa: E402
    MANDATORY_MAIN_FIGURES,
    MANDATORY_SUPPLEMENTARY_FIGURES,
    MANDATORY_TABLES,
    build_required_asset_entries,
    utc_now_iso,
    write_manifest,
)
from meso_uq.vega import build_runtime_pythonpath, get_vega_paths  # noqa: E402

MAIN_SCRIPT = REPO_ROOT / "papers" / "huq_emb" / "uqdpd_generate_reduced_story_assets.py"
SUPP_SCRIPT = REPO_ROOT / "papers" / "huq_emb" / "uqdpd_generate_supplementary_map_figures.py"
STAGING_SCRIPT = REPO_ROOT / "scripts" / "workflows" / "emb" / "huq_emb" / "stage_dnn_figure_inputs.py"
DEFAULT_TEXDEPS_DIR = REPO_ROOT / "papers" / "huq_emb" / "_texdeps"
SUPPLEMENTARY_ONLY_TABLES = {
    "map_parameter_comparison.csv",
    "map_parameter_comparison.tex",
}
DEFAULT_PYTHON_CANDIDATES = [
    REPO_ROOT / ".venv" / "bin" / "python",
    REPO_ROOT / "_vega" / "venv" / "bin" / "python",
]


def _default_python_bin() -> str:
    for candidate in DEFAULT_PYTHON_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))
    return True


def _resolve_campaign_id(paper_data_root: Path, requested: str | None) -> str:
    runs_root = paper_data_root / "runs"
    if requested:
        campaign_root = runs_root / requested
        if campaign_root.is_dir():
            return requested
        available = sorted(path.name for path in runs_root.iterdir() if path.is_dir()) if runs_root.is_dir() else []
        raise ValueError(
            f"Campaign '{requested}' not found under {runs_root}. "
            f"Available campaigns: {available or ['<none>']}"
        )

    if not runs_root.is_dir():
        raise ValueError(f"Expected campaign runs under {runs_root}, but that directory does not exist.")

    candidates = sorted(path.name for path in runs_root.iterdir() if path.is_dir())
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(f"No campaigns found under {runs_root}.")
    raise ValueError(
        f"Multiple campaigns found under {runs_root}; pass --campaign-id explicitly. "
        f"Available campaigns: {candidates}"
    )


def _staging_manifest_path(*, campaign_root: Path, staging_dirname: str) -> Path:
    staging_root = campaign_root / "postprocess_graph" / staging_dirname
    return staging_root / "dnn_figure_input_staging_report.json"


def _required_tables(*, include_supplementary: bool) -> tuple[str, ...]:
    if include_supplementary:
        return MANDATORY_TABLES
    return tuple(name for name in MANDATORY_TABLES if name not in SUPPLEMENTARY_ONLY_TABLES)


def _resolve_tex_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.disable_tex:
        return {
            "disable_tex": True,
            "texdeps_dir": None,
            "source": "argument",
            "reason": "explicitly disabled",
        }

    if args.texdeps_dir:
        texdeps_dir = _resolve_path(args.texdeps_dir)
        if not texdeps_dir.exists():
            raise ValueError(f"--texdeps-dir does not exist: {texdeps_dir}")
        return {
            "disable_tex": False,
            "texdeps_dir": texdeps_dir,
            "source": "argument",
            "reason": "explicit texdeps directory",
        }

    env_texdeps = os.environ.get("MESOUQ_PAPER_TEXDEPS_DIR", "").strip()
    if env_texdeps:
        texdeps_dir = _resolve_path(env_texdeps)
        if not texdeps_dir.exists():
            raise ValueError(
                "MESOUQ_PAPER_TEXDEPS_DIR points to a missing path: "
                f"{texdeps_dir}"
            )
        return {
            "disable_tex": False,
            "texdeps_dir": texdeps_dir,
            "source": "environment",
            "reason": "environment texdeps directory",
        }

    if DEFAULT_TEXDEPS_DIR.exists():
        return {
            "disable_tex": False,
            "texdeps_dir": DEFAULT_TEXDEPS_DIR.resolve(),
            "source": "default",
            "reason": "repo-local texdeps directory",
        }

    return {
        "disable_tex": True,
        "texdeps_dir": None,
        "source": "auto",
        "reason": "no texdeps directory configured",
    }


def _run_step(
    *,
    name: str,
    command: list[str],
    logs_root: Path,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    logs_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    start_utc = utc_now_iso()
    proc = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    elapsed = time.perf_counter() - started
    stdout_log = logs_root / f"{name}.stdout.log"
    stderr_log = logs_root / f"{name}.stderr.log"
    stdout_log.write_text(proc.stdout or "", encoding="utf-8")
    stderr_log.write_text(proc.stderr or "", encoding="utf-8")
    return {
        "name": name,
        "status": "passed" if proc.returncode == 0 else "failed",
        "command": command,
        "returncode": int(proc.returncode),
        "start_utc": start_utc,
        "end_utc": utc_now_iso(),
        "elapsed_seconds": elapsed,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
    }


def _skip_step(*, name: str, reason: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "skipped",
        "reason": reason,
    }


def _copy_required_assets(
    *,
    generated_root: Path,
    main_out: Path,
    supp_out: Path,
    tables_out: Path,
    include_supplementary: bool,
) -> list[str]:
    generated_figures = generated_root / "figures"
    generated_supp = generated_root / "supplementary"
    copied: list[str] = []
    for name in MANDATORY_MAIN_FIGURES:
        if _copy_if_exists(generated_figures / name, main_out / name):
            copied.append(str((main_out / name).resolve()))
    if include_supplementary:
        for name in MANDATORY_SUPPLEMENTARY_FIGURES:
            if _copy_if_exists(generated_supp / name, supp_out / name):
                copied.append(str((supp_out / name).resolve()))
    for name in _required_tables(include_supplementary=include_supplementary):
        for base in (generated_root, generated_figures, generated_supp):
            if _copy_if_exists(base / name, tables_out / name):
                copied.append(str((tables_out / name).resolve()))
                break
    return copied


def _required_asset_groups(
    *,
    main_out: Path,
    supp_out: Path,
    tables_out: Path,
    include_supplementary: bool,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {
        "figures_main": build_required_asset_entries(
            category="figures/main",
            root=main_out,
            required_files=MANDATORY_MAIN_FIGURES,
        ),
        "tables": build_required_asset_entries(
            category="tables",
            root=tables_out,
            required_files=_required_tables(include_supplementary=include_supplementary),
        ),
    }
    if include_supplementary:
        groups["figures_supplementary"] = build_required_asset_entries(
            category="figures/supplementary",
            root=supp_out,
            required_files=MANDATORY_SUPPLEMENTARY_FIGURES,
        )
    return groups


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay the exact HUQ-EMB paper figures and tables from a stored paper_data campaign."
    )
    parser.add_argument("--paper-data-root", required=True)
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--python-bin", default=_default_python_bin())
    parser.add_argument("--site", choices=["vega", "karolina"], default="vega")
    parser.add_argument("--staging-dirname", default="dnn_figure_input_staging")
    parser.add_argument("--texdeps-dir", default=None)
    parser.add_argument("--skip-supplementary", action="store_true", default=False)
    parser.add_argument("--disable-tex", action="store_true", default=False)
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args(argv)

    paper_data_root = _resolve_path(args.paper_data_root)
    campaign_id = _resolve_campaign_id(paper_data_root, args.campaign_id)
    campaign_root = paper_data_root / "runs" / campaign_id
    stage_root = campaign_root / "paper_exact_stage"
    generated_root = stage_root / "generated"
    logs_root = stage_root / "logs"
    report_path = stage_root / "run_exact_uqdpd_asset_port.report.json"
    main_out = paper_data_root / "figures" / "main"
    supp_out = paper_data_root / "figures" / "supplementary"
    tables_out = paper_data_root / "tables"
    include_supplementary = not args.skip_supplementary

    if args.force and stage_root.exists():
        shutil.rmtree(str(stage_root))
    for path in (generated_root, logs_root, main_out, supp_out, tables_out):
        path.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "created_at_utc": utc_now_iso(),
        "repo_root": str(REPO_ROOT),
        "paper_data_root": str(paper_data_root),
        "campaign_id": campaign_id,
        "campaign_root": str(campaign_root),
        "stage_root": str(stage_root),
        "generated_root": str(generated_root),
        "site": args.site,
        "python_bin": str(_resolve_path(args.python_bin)),
        "include_supplementary": include_supplementary,
        "status": "running",
        "steps": [],
    }

    tex_config = _resolve_tex_config(args)
    report["tex"] = {
        "disable_tex": tex_config["disable_tex"],
        "texdeps_dir": (
            str(tex_config["texdeps_dir"])
            if tex_config["texdeps_dir"] is not None
            else None
        ),
        "source": tex_config["source"],
        "reason": tex_config["reason"],
    }

    stage_command = [
        str(_resolve_path(args.python_bin)),
        str(STAGING_SCRIPT),
        "--paper-data-root",
        str(paper_data_root),
        "--campaign-id",
        campaign_id,
        "--python-bin",
        str(_resolve_path(args.python_bin)),
        "--site",
        args.site,
        "--staging-dirname",
        args.staging_dirname,
    ]
    if args.force:
        stage_command.append("--force")
    stage_step = _run_step(
        name="dnn_figure_input_staging",
        command=stage_command,
        logs_root=logs_root,
    )
    report["steps"].append(stage_step)
    if stage_step["returncode"] != 0:
        report["status"] = "failed"
        write_manifest(report_path, report)
        print(f"Exact stage root: {stage_root}")
        print(f"Report: {report_path}")
        return 1

    staging_report_path = _staging_manifest_path(
        campaign_root=campaign_root,
        staging_dirname=args.staging_dirname,
    )
    if not staging_report_path.exists():
        stage_step["status"] = "failed"
        stage_step["returncode"] = 1
        stage_step["missing_outputs"] = [str(staging_report_path)]
        report["status"] = "failed"
        write_manifest(report_path, report)
        print(f"Exact stage root: {stage_root}")
        print(f"Report: {report_path}")
        return 1

    staging_report = json.loads(staging_report_path.read_text(encoding="utf-8"))
    report["dnn_staging_report"] = str(staging_report_path)
    report["group_holdout_root"] = staging_report.get("group_holdout_root")
    report["sobol_root"] = staging_report.get("sobol_root")
    if staging_report.get("status") != "passed":
        stage_step["status"] = "failed"
        stage_step["returncode"] = 1
        stage_step["staging_status"] = staging_report.get("status")
        report["status"] = "failed"
        write_manifest(report_path, report)
        print(f"Exact stage root: {stage_root}")
        print(f"Report: {report_path}")
        return 1

    env = dict(os.environ)
    vega_paths = get_vega_paths(REPO_ROOT)
    tinytex_bin = vega_paths.tinytex_bin_dir
    if tinytex_bin.is_dir():
        env["PATH"] = str(tinytex_bin) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = build_runtime_pythonpath(
        REPO_ROOT,
        vega_paths.korali_site_packages,
        env.get("PYTHONPATH"),
        include_existing=True,
    )
    env.update(
        {
            "PYTHON_BIN": str(_resolve_path(args.python_bin)),
            "MESOUQ_PAPER_CAMPAIGN_ROOT": str(campaign_root),
            "MESOUQ_PAPER_STAGE_ROOT": str(stage_root),
            "MESOUQ_PAPER_GROUP_HOLDOUT_ROOT": str(
                _resolve_path(staging_report["group_holdout_root"])
            ),
            "MESOUQ_PAPER_SOBOL_ROOT": str(_resolve_path(staging_report["sobol_root"])),
            "HUQ_PAPER_FIGURES_DIR": str(generated_root / "figures"),
        }
    )
    if tex_config["texdeps_dir"] is not None:
        env["MESOUQ_PAPER_TEXDEPS_DIR"] = str(tex_config["texdeps_dir"])
    if tex_config["disable_tex"]:
        env["HUQ_PAPER_DISABLE_TEX"] = "1"

    main_step = _run_step(
        name="main_figures",
        command=[env["PYTHON_BIN"], str(MAIN_SCRIPT)],
        logs_root=logs_root,
        env=env,
    )
    report["steps"].append(main_step)
    if main_step["returncode"] != 0:
        report["status"] = "failed"
        write_manifest(report_path, report)
        print(f"Exact stage root: {stage_root}")
        print(f"Report: {report_path}")
        return 1

    if include_supplementary:
        supp_step = _run_step(
            name="supplementary_figures",
            command=[env["PYTHON_BIN"], str(SUPP_SCRIPT)],
            logs_root=logs_root,
            env=env,
        )
        report["steps"].append(supp_step)
        if supp_step["returncode"] != 0:
            report["status"] = "failed"
            write_manifest(report_path, report)
            print(f"Exact stage root: {stage_root}")
            print(f"Report: {report_path}")
            return 1
    else:
        report["steps"].append(
            _skip_step(
                name="supplementary_figures",
                reason="skip requested",
            )
        )

    copied_assets = _copy_required_assets(
        generated_root=generated_root,
        main_out=main_out,
        supp_out=supp_out,
        tables_out=tables_out,
        include_supplementary=include_supplementary,
    )
    required_assets = _required_asset_groups(
        main_out=main_out,
        supp_out=supp_out,
        tables_out=tables_out,
        include_supplementary=include_supplementary,
    )
    hard_failures = [
        entry["relative_path"]
        for entries in required_assets.values()
        for entry in entries
        if not entry["exists"]
    ]

    report["copied_assets"] = copied_assets
    report["required_assets"] = required_assets
    report["hard_failures"] = hard_failures
    report["status"] = "failed" if hard_failures else "passed"
    write_manifest(report_path, report)

    print(f"Exact stage root: {stage_root}")
    print(f"Report: {report_path}")
    return 1 if hard_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
