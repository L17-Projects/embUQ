from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.active_learning.emb_34um_retrospective_al_benchmark import (  # noqa: E402
    EMB_34UM_RETROSPECTIVE_AL_BENCHMARK_MANIFEST_FILENAME,
    EMB_34UM_RETROSPECTIVE_AL_BENCHMARK_SCHEMA_VERSION,
    build_emb_34um_retrospective_al_benchmark,
    write_emb_34um_retrospective_al_benchmark_artifacts,
)
from meso_uq.active_learning.final_gate_reports import validate_plot_path  # noqa: E402


def _write_samples_all(path: Path, *, count: int = 80) -> None:
    rows = []
    forces = np.array([0.0, 2500.0, 5000.0])
    for index in range(count):
        yt = 1.0e5 * (1.0e4 ** (index / max(count - 1, 1)))
        kb = 400.0 + (70000.0 - 400.0) * ((index * 17) % count) / max(count - 1, 1)
        radp = 6.8
        disp = (forces / 5000.0) * (0.2 + 0.3 * np.log10(yt / 1.0e5) / 4.0 + 0.2 * kb / 70000.0)
        outputs = 2.0 * radp - disp
        rows.append([yt, 1.0, kb, 0.0, 0.0, 0.0, 0.0, radp, *outputs.tolist(), *forces.tolist()])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(rows), fmt="%.12g")


def test_retrospective_benchmark_builds_round_summary(tmp_path: Path) -> None:
    samples = tmp_path / "samples_all.dat"
    _write_samples_all(samples)

    manifest, rows, selected = build_emb_34um_retrospective_al_benchmark(
        samples_all_path=samples,
        candidate_pool_size=30,
        max_executed_curves=12,
        round_size=4,
        validation_count=20,
        lhs_replicates=3,
        seed=4,
    )

    assert manifest["schema_version"] == EMB_34UM_RETROSPECTIVE_AL_BENCHMARK_SCHEMA_VERSION
    assert manifest["candidate_pool_size"] == 30
    assert manifest["max_executed_curves"] == 12
    assert manifest["primary_metric"] == "median_curve_rel_l2_pct"
    data = np.loadtxt(samples)
    features = np.column_stack([np.log10(data[:, 0]), np.log10(data[:, 2])])
    permutation = np.random.default_rng(4).permutation(len(features))
    pool_indices = permutation[20:50]
    expected_min = np.min(features[pool_indices], axis=0)
    expected_span = np.maximum(np.max(features[pool_indices], axis=0) - expected_min, 1.0e-12)
    assert manifest["feature_scaling"]["fit_scope"] == "candidate_pool"
    np.testing.assert_allclose(manifest["feature_scaling"]["min"], expected_min)
    np.testing.assert_allclose(manifest["feature_scaling"]["span"], expected_span)
    assert len(rows) == 3
    assert rows[-1]["curve_count"] == 12
    assert len(selected) == 12
    assert all("al_median_curve_rel_l2_pct" in row for row in rows)


def test_retrospective_benchmark_writes_plot_manifest_and_csv(tmp_path: Path) -> None:
    samples = tmp_path / "samples_all.dat"
    _write_samples_all(samples)

    artifacts = write_emb_34um_retrospective_al_benchmark_artifacts(
        samples_all_path=samples,
        output_root=tmp_path / "artifacts",
        candidate_pool_size=30,
        max_executed_curves=12,
        round_size=4,
        validation_count=20,
        lhs_replicates=3,
        seed=4,
        include_plot=False,
    )

    assert artifacts.manifest_path == tmp_path / "artifacts" / EMB_34UM_RETROSPECTIVE_AL_BENCHMARK_MANIFEST_FILENAME
    assert artifacts.manifest_path.is_file()
    assert artifacts.summary_csv_path.is_file()
    assert artifacts.selected_candidates_csv_path.is_file()
    validate_plot_path(artifacts.plot_path)

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "ready"
    assert manifest["plot_path"] == str(artifacts.plot_path)
    assert "retrospective oracle" in " ".join(manifest["notes"])
