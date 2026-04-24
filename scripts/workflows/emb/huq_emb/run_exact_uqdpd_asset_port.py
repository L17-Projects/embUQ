#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from meso_uq.campaign_manifests import MANDATORY_MAIN_FIGURES, MANDATORY_SUPPLEMENTARY_FIGURES, MANDATORY_TABLES
from meso_uq.vega import build_runtime_pythonpath, get_vega_paths

MAIN_SCRIPT = REPO_ROOT / 'papers' / 'huq_emb' / 'uqdpd_generate_reduced_story_assets.py'
SUPP_SCRIPT = REPO_ROOT / 'papers' / 'huq_emb' / 'uqdpd_generate_supplementary_map_figures.py'
UPSTREAM_TEXDEPS = Path('/ceph/hpc/home/eubrieucb/workspace/UQ_DPD/Hierarchical_UQ_compression/_paper/v3/files/_texdeps')
DEFAULT_PYTHON_CANDIDATES = [
    REPO_ROOT / '.venv' / 'bin' / 'python',
    REPO_ROOT / '_vega' / 'venv' / 'bin' / 'python',
]


def _default_python_bin():
    for candidate in DEFAULT_PYTHON_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _copy_if_exists(src, dst):
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))
    return True


def _run(script, env):
    result = subprocess.run([env['PYTHON_BIN'], str(script)], cwd=str(REPO_ROOT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logs = Path(env['MESOUQ_PAPER_STAGE_ROOT']) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    stem = script.stem
    (logs / ('%s.stdout.log' % stem)).write_text(result.stdout or '', encoding='utf-8')
    (logs / ('%s.stderr.log' % stem)).write_text(result.stderr or '', encoding='utf-8')
    if result.returncode != 0:
        raise RuntimeError('%s failed with return code %s' % (script.name, result.returncode))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run exact UQ_DPD paper asset port against a MesoUQ campaign.')
    parser.add_argument('--paper-data-root', required=True)
    parser.add_argument('--campaign-id', required=True)
    parser.add_argument('--python-bin', default=_default_python_bin())
    parser.add_argument('--group-holdout-root', required=True)
    parser.add_argument('--sobol-root', required=True)
    parser.add_argument('--skip-supplementary', action='store_true', default=False)
    parser.add_argument('--disable-tex', action='store_true', default=False)
    parser.add_argument('--force', action='store_true', default=False)
    args = parser.parse_args(argv)

    paper_data_root = Path(args.paper_data_root).expanduser().resolve()
    campaign_root = paper_data_root / 'runs' / args.campaign_id
    stage_root = campaign_root / 'paper_exact_stage'
    generated_root = stage_root / 'generated'
    main_out = paper_data_root / 'figures' / 'main'
    supp_out = paper_data_root / 'figures' / 'supplementary'
    tables_out = paper_data_root / 'tables'

    if args.force and stage_root.exists():
        shutil.rmtree(str(stage_root))
    generated_root.mkdir(parents=True, exist_ok=True)

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
    env.update({
        'PYTHON_BIN': str(Path(args.python_bin).expanduser()),
        'MESOUQ_PAPER_CAMPAIGN_ROOT': str(campaign_root),
        'MESOUQ_PAPER_STAGE_ROOT': str(stage_root),
        'MESOUQ_PAPER_GROUP_HOLDOUT_ROOT': str(Path(args.group_holdout_root).expanduser().resolve()),
        'MESOUQ_PAPER_SOBOL_ROOT': str(Path(args.sobol_root).expanduser().resolve()),
        'MESOUQ_PAPER_TEXDEPS_DIR': str(UPSTREAM_TEXDEPS),
        'HUQ_PAPER_FIGURES_DIR': str(generated_root / 'figures'),
    })
    if args.disable_tex:
        env['HUQ_PAPER_DISABLE_TEX'] = '1'

    _run(MAIN_SCRIPT, env)
    if not args.skip_supplementary:
        _run(SUPP_SCRIPT, env)

    generated_figures = generated_root / 'figures'
    generated_supp = generated_root / 'supplementary'
    copied = []
    for name in MANDATORY_MAIN_FIGURES:
        if _copy_if_exists(generated_figures / name, main_out / name):
            copied.append(str(main_out / name))
    for name in MANDATORY_SUPPLEMENTARY_FIGURES:
        if _copy_if_exists(generated_supp / name, supp_out / name):
            copied.append(str(supp_out / name))
    for name in MANDATORY_TABLES:
        for base in (generated_root, generated_figures, generated_supp):
            if _copy_if_exists(base / name, tables_out / name):
                copied.append(str(tables_out / name))
                break

    report = stage_root / 'run_exact_uqdpd_asset_port.report.txt'
    report.write_text('\n'.join(copied) + ('\n' if copied else ''), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
