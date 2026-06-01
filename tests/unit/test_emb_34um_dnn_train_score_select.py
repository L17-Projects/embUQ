from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest


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
            "parameters": {"ka": 6.0e3, "kb": 6.0e3},
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


def _completed_rows_d4() -> list[dict[str, object]]:
    rows = _completed_rows()
    for index, row in enumerate(rows):
        parameters = dict(row["parameters"])  # type: ignore[index]
        parameters["radp"] = 6.45 + 0.05 * index
        parameters["shell_th"] = 3.75e-9 + 0.2e-9 * index
        row["parameters"] = parameters
    return rows


def _candidate_pool() -> list[dict[str, float]]:
    return [
        {"candidate_id": f"cand-{index:03d}", "ka": 7.0e3 + 1.0e3 * index, "kb": 7.0e3 + 1.0e3 * index}
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
            "--candidate-space",
            "d2",
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
    assert manifest["existing_points"]["source"] == "completed_rows"
    assert manifest["existing_points"]["count"] == len(_completed_rows())
    assert manifest["existing_points"]["used_for_diversity"] is True
    assert manifest["timing_canary"]["enabled"] is True
    assert report["status"] == "dry-run"
    assert timing["enabled"] is True

    assert (output_root / module.EMB_34UM_DNN_TRAIN_SCORE_SELECT_SELECTION_PLOT_FILENAME).is_file()
    assert (output_root / module.EMB_34UM_DNN_TRAIN_SCORE_SELECT_DISTRIBUTION_PLOT_FILENAME).is_file()
    assert (output_root / module.EMB_34UM_DNN_TRAIN_SCORE_SELECT_LOSS_PLOT_FILENAME).is_file()
    assert (output_root / module.EMB_34UM_DNN_TRAIN_SCORE_SELECT_TIMING_PLOT_FILENAME).is_file()


def test_cli_dry_run_curve_error_mode_records_risk_and_plot(tmp_path: Path) -> None:
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
            "--candidate-space",
            "d2",
            "--dry-run",
            "--acquisition-score-mode",
            "curve_error",
            "--top-n",
            "3",
        ]
    )

    assert rc == 0
    manifest = json.loads((output_root / module.EMB_34UM_DNN_TRAIN_SCORE_SELECT_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["selection"]["acquisition_score_mode"] == "curve_error"
    assert manifest["selection"]["predicted_curve_relative_l2_error"]["max"] > 0.0
    assert manifest["selected_points"][0]["acquisition_base_field"] == "predicted_curve_relative_l2_error"
    assert manifest["plots"]["curve_error_risk"].endswith(module.EMB_34UM_DNN_TRAIN_SCORE_SELECT_CURVE_ERROR_PLOT_FILENAME)
    assert (output_root / module.EMB_34UM_DNN_TRAIN_SCORE_SELECT_CURVE_ERROR_PLOT_FILENAME).is_file()


def test_cli_rejects_candidate_pool_in_runtime_risk_region(tmp_path: Path) -> None:
    module = _load_module()
    completed_rows_path = tmp_path / "completed_rows.json"
    candidate_pool_path = tmp_path / "candidate_pool.json"
    output_root = tmp_path / "out"
    completed_rows_path.write_text(json.dumps(_completed_rows_d4()), encoding="utf-8")
    candidate_pool_path.write_text(
        json.dumps([{"candidate_id": "bad-runtime-region", "ka": 1272.145949837055, "kb": 9011.372292672091, "radp": 6.489854540732141, "shell_th": 3.9170402720459855e-9}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="runtime-risk timeout"):
        module.main(
            [
                "--completed-rows",
                str(completed_rows_path),
                "--candidate-pool",
                str(candidate_pool_path),
                "--output-root",
                str(output_root),
                "--candidate-space",
                "d2",
                "--dry-run",
            ]
        )


def _candidate_pool_d4() -> list[dict[str, object]]:
    return [
        {"candidate_id": "c1", "ka": 3.0e3, "kb": 6.0e3, "radp": 6.45, "shell_th": 3.75e-9},
        {"candidate_id": "c2", "ka": 3.0e3, "kb": 6.0e3, "radp": 6.55, "shell_th": 3.75e-9},
        {"candidate_id": "c3", "ka": 3.0e3, "kb": 6.0e3, "radp": 6.60, "shell_th": 4.0e-9},
    ]


def test_cli_d4_mode_selects_candidates_with_radp_and_shell_th_signal(tmp_path: Path) -> None:
    module = _load_module()
    completed_rows_path = tmp_path / "completed_rows.json"
    candidate_pool_path = tmp_path / "candidate_pool.json"
    output_root = tmp_path / "out"
    completed_rows_path.write_text(json.dumps(_completed_rows_d4()), encoding="utf-8")
    candidate_pool_path.write_text(json.dumps(_candidate_pool_d4()), encoding="utf-8")

    rc_d4 = module.main(
        [
            "--completed-rows",
            str(completed_rows_path),
            "--candidate-pool",
            str(candidate_pool_path),
            "--output-root",
            str(output_root / "d4"),
            "--existing-points",
            json.dumps([[1.0e3, 2.0e3, 6.49, 3.8e-9]]),
            "--candidate-space",
            "d4",
            "--dry-run",
            "--top-n",
            "1",
        ]
    )

    assert rc_d4 == 0
    manifest_d4 = json.loads((output_root / "d4" / module.EMB_34UM_DNN_TRAIN_SCORE_SELECT_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest_d4["candidate_space"] == "d4"
    assert len(manifest_d4["fit"]["feature_mean"]) == 5
    assert manifest_d4["existing_points"]["source"] == "cli_existing_points"
    assert manifest_d4["existing_points"]["points"] == [[1.0e3, 2.0e3, 6.49, 3.8e-09]]
    assert manifest_d4["selection"]["candidate_pool_count"] == 3
    assert manifest_d4["selected_points"][0]["candidate_id"] == "c3"
    assert manifest_d4["selected_points"][0]["radp"] == 6.6
    assert manifest_d4["selected_points"][0]["shell_th"] == 4e-9
    assert (output_root / "d4" / module.EMB_34UM_DNN_TRAIN_SCORE_SELECT_KA_RADP_PLOT_FILENAME).is_file()
    assert (output_root / "d4" / module.EMB_34UM_DNN_TRAIN_SCORE_SELECT_KB_SHELL_TH_PLOT_FILENAME).is_file()
    assert (output_root / "d4" / module.EMB_34UM_DNN_TRAIN_SCORE_SELECT_RADP_SHELL_TH_PLOT_FILENAME).is_file()

    rc_d2 = module.main(
        [
            "--completed-rows",
            str(completed_rows_path),
            "--candidate-pool",
            str(candidate_pool_path),
            "--output-root",
            str(output_root / "d2"),
            "--candidate-space",
            "d2",
            "--dry-run",
            "--top-n",
            "1",
        ]
    )

    assert rc_d2 == 0
    manifest_d2 = json.loads((output_root / "d2" / module.EMB_34UM_DNN_TRAIN_SCORE_SELECT_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest_d2["candidate_space"] == "d2"
    assert manifest_d2["selected_points"][0]["candidate_id"] == "c1"


def test_surrogate_rejects_d4_geometry_against_d2_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = _repo_root()
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    import meso_uq.active_learning.emb_34um_dnn_surrogate as surrogate

    fit = surrogate.Emb34umDnnSurrogateFit(
        architecture="dnn_mlp_64_64",
        backend="fixed_architecture_dnn",
        ensemble_seeds=(1,),
        ensemble_checkpoints=(Path("fake.pt"),),
        ensemble_member_count=1,
        train_runtime_seconds=0.01,
        score_runtime_seconds=0.0,
        feature_mean=(0.0, 0.0, 0.0),
        feature_scale=(1.0, 1.0, 1.0),
        target_mean=0.0,
        target_scale=1.0,
        force_min=0.0,
        force_max=5000.0,
        force_grid=(0.0, 5000.0),
        device="cpu",
        train_row_count=2,
        validation_row_count=1,
        model_path=Path("fake.pt"),
        training_history=(),
    )
    monkeypatch.setattr(
        surrogate,
        "_load_ensemble_members",
        lambda _: ((object(), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.0, 1.0),),
    )
    monkeypatch.setattr(
        surrogate,
        "_predict_member_rows",
        lambda model, rows, *, scaler: [0.0 for _ in rows],
    )

    with pytest.raises(ValueError, match="D4 geometry was supplied to a 3-feature D2 surrogate fit"):
        surrogate.load_ensemble_member_predictions(
            fit,
            fit.force_grid,
            ka=1.0e3,
            kb=2.0e3,
            radp=6.5,
            shell_th=3.75e-9,
        )


def test_cli_candidate_space_defaults_to_d4() -> None:
    module = _load_module()
    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--completed-rows",
            "completed_rows.json",
            "--candidate-pool",
            "candidate_pool.json",
            "--output-root",
            "out",
        ]
    )

    assert args.candidate_space == "d4"


def test_cli_existing_points_argument_overrides_completed_rows_diversity_source(tmp_path: Path) -> None:
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
            "--existing-points",
            json.dumps([[1234.0, 4567.0]]),
            "--candidate-space",
            "d2",
            "--dry-run",
        ]
    )

    assert rc == 0
    manifest = json.loads((output_root / module.EMB_34UM_DNN_TRAIN_SCORE_SELECT_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["existing_points"]["source"] == "cli_existing_points"
    assert manifest["existing_points"]["count"] == 1


def test_cli_excludes_candidate_pool_points_already_completed(tmp_path: Path) -> None:
    module = _load_module()
    completed_rows_path = tmp_path / "completed_rows.json"
    candidate_pool_path = tmp_path / "candidate_pool.json"
    output_root = tmp_path / "out"
    completed_rows_path.write_text(json.dumps(_completed_rows()), encoding="utf-8")
    candidate_pool_path.write_text(
        json.dumps(
            [
                {"candidate_id": "dup-a", "ka": 6.0e3, "kb": 6.0e3},
                {"candidate_id": "dup-b", "ka": 1.0e4, "kb": 5.0e3},
                {"candidate_id": "novel", "ka": 2.0e4, "kb": 1.5e4},
            ]
        ),
        encoding="utf-8",
    )

    rc = module.main(
        [
            "--completed-rows",
            str(completed_rows_path),
            "--candidate-pool",
            str(candidate_pool_path),
            "--output-root",
            str(output_root),
            "--candidate-space",
            "d2",
            "--dry-run",
            "--top-n",
            "1",
        ]
    )

    assert rc == 0
    manifest = json.loads((output_root / module.EMB_34UM_DNN_TRAIN_SCORE_SELECT_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["selection"]["candidate_pool_count"] == 1
    assert manifest["selection"]["selected_count"] == 1
    assert manifest["selection"]["skipped_existing_candidate_count"] == 2
    assert set(manifest["selection"]["skipped_existing_candidate_ids"]) == {"dup-a", "dup-b"}


def test_existing_points_from_completed_rows_includes_flat_d4_rows() -> None:
    module = _load_module()
    completed_rows = [
        {
            "curve_id": "flat-d4",
            "ka": 3.0e3,
            "kb": 6.0e3,
            "radp": 6.45,
            "shell_th": 3.75e-9,
        }
    ]

    existing_points = module._existing_points_from_completed_rows(completed_rows, candidate_space="d4")
    filtered, skipped = module._exclude_existing_points_from_candidate_pool(
        [
            {"candidate_id": "duplicate", "ka": 3.0e3, "kb": 6.0e3, "radp": 6.45, "shell_th": 3.75e-9},
            {"candidate_id": "novel", "ka": 4.0e3, "kb": 7.0e3, "radp": 6.55, "shell_th": 3.95e-9},
        ],
        existing_points=existing_points,
        candidate_space="d4",
    )

    assert existing_points == ((3.0e3, 6.0e3, 6.45, 3.75e-9),)
    assert [row["candidate_id"] for row in filtered] == ["novel"]
    assert skipped == ("duplicate",)


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
