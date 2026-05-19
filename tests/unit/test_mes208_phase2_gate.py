from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYZE_PATH = REPO_ROOT / "scripts" / "platforms" / "karolina" / "analyze_mes208_phase2_gate.py"
SBATCH_PATH = (
    REPO_ROOT / "scripts" / "platforms" / "karolina" / "sbatch" / "native_cuda_phase2_replicate.sbatch"
)


def _load_analyzer():
    spec = importlib.util.spec_from_file_location("analyze_mes208_phase2_gate", ANALYZE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_mes208_gate_statistics_detect_shift() -> None:
    analyzer = _load_analyzer()
    base = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float)
    shifted = base + 1.0

    assert analyzer.ks_statistic(base, base) == 0.0
    assert analyzer.wasserstein_1d(base, base) == 0.0
    assert analyzer.ks_statistic(base, shifted) > 0.0
    assert analyzer.wasserstein_1d(base, shifted) == 1.0
    assert analyzer.normalize(2.0, 4.0) == 0.5


def test_mes208_replicate_sbatch_records_explicit_korali_env() -> None:
    script = SBATCH_PATH.read_text(encoding="utf-8")

    assert "source \"${ENV_SCRIPT}\"" in script
    assert "KORALI_ENV_SCRIPT" in script
    assert "source \"${KORALI_ENV_SCRIPT}\"" in script
    assert "scripts/platforms/vega/run_inference_stage.py" in script
    assert "--phase2-backend" in script
