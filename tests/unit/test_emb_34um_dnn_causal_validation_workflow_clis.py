from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _seed_minimal_campaign(campaign_root: Path) -> None:
    manifest_path = campaign_root / "emb_34um_dnn_causal_validation_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "meso_uq.active_learning.emb_34um_dnn_causal_validation.v1",
            "campaign_root": str(campaign_root),
        },
    )
    stage_root = campaign_root / "unseen_test"
    output_root = stage_root / "emb" / "unseen-001"
    manifest = output_root / "dpd_sampling_candidate_manifest.json"
    _write_json(
        manifest,
        {
            "candidate_id": "unseen-001",
            "output_root": str(output_root),
            "normalized_payload": {
                "candidate_id": "unseen-001",
                "output_root": str(output_root),
                "parameters": {"ka": 1.0e3, "kb": 2.0e3},
                "force_grid": [5000.0 * index / 7.0 for index in range(8)],
            },
            "rendered_payload": {
                "request_payload": {
                    "candidate_id": "unseen-001",
                    "output_root": str(output_root),
                    "parameters": {"ka": 1.0e3, "kb": 2.0e3},
                    "force_grid": [5000.0 * index / 7.0 for index in range(8)],
                }
            },
            "active_learning_metadata": {},
        },
    )
    _write_json(
        stage_root / "emb_34um_batch_summary.json",
        {"rendered_candidate_manifests": [str(manifest)]},
    )
    _write_json(
        output_root / "emb_34um_result.json",
        {
            "candidate_id": "unseen-001",
            "force_grid": [5000.0 * index / 7.0 for index in range(8)],
            "vertical_diameter": [float(index + 1) for index in range(8)],
        },
    )
    _write_json(output_root / "emb_34um_runtime_status.json", {"status": "completed", "retry_count": 0})


def test_ingest_dnn_cli_writes_artifacts_from_campaign_root(tmp_path: Path) -> None:
    module = _load_module(
        REPO_ROOT / "scripts" / "workflows" / "emb" / "active_learning" / "ingest_emb_34um_dnn_causal_validation.py",
        "ingest_emb_34um_dnn_causal_validation",
    )
    campaign_root = tmp_path / "campaign"
    _seed_minimal_campaign(campaign_root)

    rc = module.main(["--campaign-root", str(campaign_root)])
    assert rc == 0
    assert (campaign_root / "ingest" / "emb_34um_dnn_causal_validation_ingestion_manifest.json").is_file()
    assert (campaign_root / "ingest" / "emb_34um_dnn_causal_validation_ingestion_report.json").is_file()
    assert (campaign_root / "ingest" / "emb_34um_dnn_causal_validation_completed_rows.json").is_file()


def test_analyze_dnn_cli_resolves_campaign_root_metric_rows_first(tmp_path: Path, monkeypatch) -> None:
    module = _load_module(
        REPO_ROOT / "scripts" / "workflows" / "emb" / "active_learning" / "analyze_emb_34um_dnn_causal_validation.py",
        "analyze_emb_34um_dnn_causal_validation",
    )
    campaign_root = tmp_path / "campaign"
    ingest_root = campaign_root / "ingest"
    ingest_manifest = ingest_root / "emb_34um_dnn_causal_validation_ingestion_manifest.json"
    metric_rows = ingest_root / "emb_34um_dnn_causal_validation_rows.json"
    _write_json(ingest_manifest, {"schema_version": "test", "records": []})
    _write_json(metric_rows, {"schema_version": "test", "rows": []})

    def _fake_writer(*, rows, output_root, metadata, bootstrap_resamples, bootstrap_seed, confidence, include_plot, generation_command):
        assert rows == metric_rows
        assert output_root == campaign_root / "analyze"
        assert metadata == {"ensemble_size": 10}
        return SimpleNamespace(
            report_path=output_root / "emb_34um_dnn_causal_validation_report.json",
            report={"decision": {"passed": True}, "status": "passed"},
        )

    monkeypatch.setattr(module, "write_emb_34um_dnn_causal_validation_artifacts", _fake_writer)

    rc = module.main(["--campaign-root", str(campaign_root)])
    assert rc == 0


def test_metric_dnn_cli_writes_analyze_rows_from_completed_rows(tmp_path: Path) -> None:
    module = _load_module(
        REPO_ROOT / "scripts" / "workflows" / "emb" / "active_learning" / "run_emb_34um_dnn_causal_validation_metrics.py",
        "run_emb_34um_dnn_causal_validation_metrics",
    )
    force_grid = [5000.0 * index / 7.0 for index in range(8)]

    def row(candidate_id: str, branch: str, replicate: int, cycle: int, ka: float, kb: float) -> dict[str, object]:
        return {
            "candidate_id": candidate_id,
            "branch": branch,
            "replicate": replicate,
            "cycle": cycle,
            "ka": ka,
            "kb": kb,
            "force_grid": force_grid,
            "reference_curve": [0.1 + 0.01 * index + 1.0e-6 * ka + 1.0e-7 * kb for index in range(8)],
        }

    completed_rows_path = tmp_path / "completed_rows.json"
    _write_json(
        completed_rows_path,
        {
            "records": [
                row("u-1", "unseen_test", 0, 0, 1.0e3, 2.0e3),
                row("u-2", "unseen_test", 0, 0, 2.0e3, 3.0e3),
                row("s-1", "shared_initial", 1, 0, 1.5e3, 2.2e3),
                row("a-1", "al", 1, 1, 3.0e3, 2.5e3),
                row("l-1", "lhs", 1, 1, 5.0e3, 4.5e3),
            ]
        },
    )
    output_root = tmp_path / "metrics"

    rc = module.main(
        [
            "--completed-rows",
            str(completed_rows_path),
            "--output-root",
            str(output_root),
            "--dry-run",
        ]
    )

    assert rc == 0
    rows_payload = json.loads((output_root / "emb_34um_dnn_causal_validation_rows.json").read_text(encoding="utf-8"))
    assert rows_payload["row_count"] == 2
    assert {row["branch"] for row in rows_payload["rows"]} == {"al", "lhs"}


def test_metric_sbatch_runs_metric_cli_and_analysis_on_gpu_node() -> None:
    script_path = (
        REPO_ROOT
        / "scripts"
        / "platforms"
        / "karolina"
        / "sbatch"
        / "emb_34um_dnn_causal_validation_metrics.sbatch"
    )
    text = script_path.read_text(encoding="utf-8")

    assert "#SBATCH --partition=qgpu" in text
    assert "#SBATCH --time=08:00:00" in text
    assert "#SBATCH --gpus=1" in text
    assert "source /scratch/project/eu-26-17/eubrieucb/mesouq/load_mesouq_karolina.sh" in text
    assert "run_emb_34um_dnn_causal_validation_metrics.py" in text
    assert "analyze_emb_34um_dnn_causal_validation.py" in text
    assert "emb_34um_dnn_causal_validation_rows.json" in text
