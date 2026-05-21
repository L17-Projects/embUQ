from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_gv_paper_figure_replay_karolina_template_uses_public_command_and_karolina_runtime_env() -> None:
    template = REPO_ROOT / "scripts" / "platforms" / "karolina" / "sbatch" / "gv_paper_figure_replay.sbatch"

    text = template.read_text(encoding="utf-8")

    assert 'REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"' in text
    assert 'CAMPAIGN_ID="${CAMPAIGN_ID:-}"' in text
    assert 'LANES="${LANES:-}"' in text
    assert 'PAPER_EXACT="${PAPER_EXACT:-0}"' in text
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-${MESOUQ_RUNS_ROOT}/gv/figure_replay/${CAMPAIGN_ID}}"' in text
    assert 'STRETCHING_POINT_START="${STRETCHING_POINT_START:-}"' in text
    assert 'STRETCHING_POINT_STOP="${STRETCHING_POINT_STOP:-}"' in text
    assert 'BUCKLING_TIMEOUT_SECONDS="${BUCKLING_TIMEOUT_SECONDS:-}"' in text
    assert "scripts/workflows/gv/run_paper_figure_replay.py" in text
    assert 'command+=(--lane "${lane}")' in text
    assert "command+=(--paper-exact)" in text
    assert 'command+=(--stretching-point-start "${STRETCHING_POINT_START}")' in text
    assert 'command+=(--stretching-point-stop "${STRETCHING_POINT_STOP}")' in text
    assert 'source scripts/platforms/karolina/env_karolina.sh' in text
    assert text.index('source scripts/platforms/karolina/env_karolina.sh') < text.index('OUTPUT_ROOT="${OUTPUT_ROOT:-${MESOUQ_RUNS_ROOT}/gv/figure_replay/${CAMPAIGN_ID}}"')
    assert 'GV_ENV_SCRIPT="${MESOUQ_GV_ENV_SCRIPT:-${MESOUQ_SITE_RUNTIME_ROOT}/env/env.sh}"' in text
    assert "Missing required GV runtime environment" in text
    assert "for lane in ${LANES}; do" in text
    assert "_vega" + "/env/env.sh" not in text
    assert "#SBATCH --partition=qgpu" in text
    assert "#SBATCH --account=eu-26-17" in text
    assert "#SBATCH --ntasks=2" in text
    assert 'MESOUQ_GV_MPI_RANKS="${MESOUQ_GV_MPI_RANKS:-2}"' in text
    assert 'MESOUQ_GV_EIGENMODES_MPI_RANKS="${MESOUQ_GV_EIGENMODES_MPI_RANKS:-2}"' in text
    assert 'MESOUQ_GV_EIGENMODES_DOMAIN_RANKS="${MESOUQ_GV_EIGENMODES_DOMAIN_RANKS:-1,1,1}"' in text


def test_gv_paper_figure_replay_karolina_submitter_sets_lane_aware_walltime() -> None:
    submitter = REPO_ROOT / "scripts" / "platforms" / "karolina" / "submit_gv_paper_figure_replay.sh"

    text = submitter.read_text(encoding="utf-8")

    assert "GV_PAPER_REPLAY_TIME_LIMIT" in text
    assert 'LANES="stretching buckling torsion eigenmodes"' in text
    assert 'MESOUQ_SCRATCH_ROOT="${MESOUQ_SCRATCH_ROOT:-/scratch/project/${MESOUQ_PROJECT_ID}/eubrieucb/mesouq}"' in text
    assert 'MESOUQ_RUNS_ROOT="${MESOUQ_RUNS_ROOT:-${MESOUQ_SCRATCH_ROOT}/runs}"' in text
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-${MESOUQ_RUNS_ROOT}/gv/figure_replay/${CAMPAIGN_ID}}"' in text
    assert 'exec sbatch --time="${TIME_LIMIT}"' in text
    assert "STRETCHING_POINT_START" in text
    assert "STRETCHING_POINT_STOP" in text
    assert "BUCKLING_TIMEOUT_SECONDS" in text
    assert "torsion)" in text and 'echo "01:00:00"' in text
    assert "buckling)" in text and 'echo "04:00:00"' in text
    assert "eigenmodes)" in text and 'echo "08:00:00"' in text
    assert '(( count <= 30 ))' in text and 'echo "03:00:00"' in text
