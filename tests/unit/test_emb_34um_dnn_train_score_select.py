from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _script_path() -> Path:
    return (
        _repo_root()
        / "scripts"
        / "workflows"
        / "emb"
        / "active_learning"
        / "run_emb_34um_dnn_train_score_select.py"
    )


def _load_module():
    repo_root = _repo_root()
    src_root = repo_root / "src"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    spec = importlib.util.spec_from_file_location("run_emb_34um_dnn_train_score_select", _script_path())
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _completed_rows() -> list[dict[str, object]]:
    force_grid = [5000.0 * index / 7.0 for index in range(8)]
    return [
        {
            "curve_id": "curve-a",
            "parameters": {"ka": 1.0e3, "kb": 5.0e2},
            "force_grid": force_grid,
            "target_curve": [0.1 * index for index, _ in enumerate(force_grid, start=1)],
        },
        {
            "curve_id": "curve-b",
            "parameters": {"ka": 1.0e4, "kb": 5.0e3},
            "force_grid": force_grid,
            "target_curve": [0.2 * index for index, _ in enumerate(force_grid, start=1)],
        },
    ]


def _candidate_pool() -> list[dict[str, float]]:
    return [
        {"candidate_id": f"cand-{index:03d}", "ka": 1.0e2 * (index + 1), "kb": 4.0e2 * (index + 1)}
        for index in range(12)
    ]


def test_cli_dry_run_timing_canary_writes_manifest_report_and_plots(tmp_path: Path) -> None:
    module = _load_module()
    completed_rows_path = tmp_path / "completed_rows.json"
    candidate_pool_path = tmp_path / "candidate_pool.json"
    output_root = tmp_path / "out"
    completed_rows_path.write_text(json.dumps(_completed_rows()), encoding="utf-8")
    candidate_pool_path.write_text(json.dumps(_candidate_pool()), encoding="utf-8")

    rc = module.main(
        [
            "--completed-rows",
            str(completed_rows_path),
            "--candidate-pool",
            str(candidate_pool_path),
            "--output-root",
            str(output_root),
            "--dry-run",
            "--timing-canary",
        ]
    )

    assert rc == 0
    manifest = json.loads((output_root / module.EMB_34UM_DNN_TRAIN_SCORE_SELECT_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    report = json.loads((output_root / module.EMB_34UM_DNN_TRAIN_SCORE_SELECT_REPORT_FILENAME).read_text(encoding="utf-8"))
    timing = json.loads((output_root / module.EMB_34UM_DNN_TRAIN_SCORE_SELECT_TIMING_REPORT_FILENAME).read_text(encoding="utf-8"))

    assert manifest["fit"]["ensemble_size"] == 10
    assert manifest["selection"]["selected_count"] == 12
    assert manifest["timing_canary"]["enabled"] is True
    assert report["status"] == "dry-run"
    assert timing["enabled"] is True

    assert (output_root / module.EMB_34UM_DNN_TRAIN_SCORE_SELECT_SELECTION_PLOT_FILENAME).is_file()
    assert (output_root / module.EMB_34UM_DNN_TRAIN_SCORE_SELECT_DISTRIBUTION_PLOT_FILENAME).is_file()
    assert (output_root / module.EMB_34UM_DNN_TRAIN_SCORE_SELECT_LOSS_PLOT_FILENAME).is_file()
    assert (output_root / module.EMB_34UM_DNN_TRAIN_SCORE_SELECT_TIMING_PLOT_FILENAME).is_file()


def test_dnn_stage_files_do_not_import_lightweight_selector() -> None:
    repo_root = _repo_root()
    paths = [
        repo_root / "src" / "meso_uq" / "active_learning" / "emb_34um_dnn_surrogate.py",
        repo_root / "src" / "meso_uq" / "active_learning" / "emb_34um_dnn_acquisition.py",
        _script_path(),
    ]

    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert "meso_uq.active_learning.emb_34um_final_gate_surrogate" not in names
