from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
KAROLINA_SBATCH_DIR = REPO_ROOT / "scripts" / "platforms" / "karolina" / "sbatch"
VEGA_SBATCH_DIR = REPO_ROOT / "scripts" / "platforms" / "vega" / "sbatch"
SURROGATE_HOLDOUT_KNOBS = (
    'SEED="${SEED:-20260317}"',
    'VAL_FRACTION="${VAL_FRACTION:-0.10}"',
    'PRED_MC_SAMPLES="${PRED_MC_SAMPLES:-64}"',
    'PRED_MC_CHUNK="${PRED_MC_CHUNK:-8}"',
    'DEVICE="${DEVICE:-gpu}"',
    'DISP_SOURCE="${DISP_SOURCE:-auto}"',
    'RUPTURE_RATIO="${RUPTURE_RATIO:-2.0}"',
    '--seed "${SEED}"',
    '--val-fraction "${VAL_FRACTION}"',
    '--predictive-mc-samples "${PRED_MC_SAMPLES}"',
    '--predictive-mc-chunk-size "${PRED_MC_CHUNK}"',
    '--device "${DEVICE}"',
    '--disp-source "${DISP_SOURCE}"',
    '--rupture-ratio "${RUPTURE_RATIO}"',
)


def _karolina_gpu_templates() -> list[Path]:
    templates: list[Path] = []
    for path in sorted(KAROLINA_SBATCH_DIR.glob("*.sbatch")):
        text = path.read_text(encoding="utf-8")
        if any(line.startswith("#SBATCH --partition=qgpu") for line in text.splitlines()):
            templates.append(path)
    return templates


@pytest.mark.parametrize("template_path", _karolina_gpu_templates(), ids=lambda path: path.name)
def test_karolina_gpu_templates_use_account_and_gpus_directive(template_path: Path) -> None:
    text = template_path.read_text(encoding="utf-8")

    assert "#SBATCH --account=eu-26-17" in text
    assert "#SBATCH --gpus=" in text
    assert "#SBATCH --gres=gpu" not in text


def test_karolina_surrogate_holdout_uses_single_gpu_and_scratch_runtime() -> None:
    text = (KAROLINA_SBATCH_DIR / "surrogate_group_holdout.sbatch").read_text(encoding="utf-8")

    assert "#SBATCH --partition=qgpu" in text
    assert "#SBATCH --gpus=1" in text
    assert "#SBATCH --gres=gpu" not in text
    assert 'source scripts/platforms/karolina/env_karolina.sh' in text
    assert 'source scripts/platforms/hpc/site_env.sh' in text
    assert 'mesouq_activate_site_env karolina "${REPO_ROOT}"' in text
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-${MESOUQ_RUNS_ROOT}/surrogate_group_holdout/${RUN_TAG}}"' in text
    legacy_venv = 'VENV' + '_DIR'
    assert f'{legacy_venv}=' not in text
    assert f'source "${{{legacy_venv}}}/bin/activate"' not in text
    assert ("_vega" + "/") not in text
    assert "--site karolina" in text


@pytest.mark.parametrize(
    "template_path,site",
    [
        (VEGA_SBATCH_DIR / "surrogate_group_holdout.sbatch", "vega"),
        (KAROLINA_SBATCH_DIR / "surrogate_group_holdout.sbatch", "karolina"),
    ],
    ids=["vega", "karolina"],
)
def test_site_surrogate_holdout_templates_expose_shared_hpc_knobs(template_path: Path, site: str) -> None:
    text = template_path.read_text(encoding="utf-8")

    assert f"--site {site}" in text
    for snippet in SURROGATE_HOLDOUT_KNOBS:
        assert snippet in text
