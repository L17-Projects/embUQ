from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.active_learning.emb_34um_dnn_causal_validation_ingestion import (
    build_emb_34um_dnn_causal_validation_ingestion_report,
    normalize_emb_34um_dnn_causal_validation_records,
)
from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import EMB_34UM_DNN_CAUSAL_FORCE_GRID


def _force_grid() -> list[float]:
    return list(EMB_34UM_DNN_CAUSAL_FORCE_GRID)


def _shared_unseen_record(index: int, *, used_for_training: bool = False, force_grid: list[float] | None = None, source: str = "fresh_dpd_shared_unseen") -> dict[str, object]:
    return {
        "candidate_id": f"unseen-{index:03d}",
        "stage": "unseen_test",
        "source": source,
        "force_grid": list(force_grid or _force_grid()),
        "used_for_training": used_for_training,
    }


def _final_lhs_record(index: int, *, source: str = "fresh_dpd") -> dict[str, object]:
    return {
        "candidate_id": f"lhs-{index:03d}",
        "method": "lhs",
        "source": source,
        "force_grid": list(_force_grid()),
    }


def _selection_training_record(index: int, *, ensemble_size: object = 10) -> dict[str, object]:
    record = {
        "candidate_id": f"train-{index:03d}",
        "kind": "training",
    }
    if ensemble_size is not None:
        record["ensemble_size"] = ensemble_size
    return record


def _payload(*, unseen_count: int = 100, unseen_kwargs: dict[str, object] | None = None, lhs_source: str = "fresh_dpd", training_ensemble_size: object = 10) -> dict[str, object]:
    unseen_kwargs = unseen_kwargs or {}
    return {
        "shared_unseen": [_shared_unseen_record(index + 1, **unseen_kwargs) for index in range(unseen_count)],
        "final_lhs": [_final_lhs_record(1, source=lhs_source)],
        "selection_training": [_selection_training_record(1, ensemble_size=training_ensemble_size)],
    }


def test_normalize_and_validate_mapping_payload() -> None:
    report = build_emb_34um_dnn_causal_validation_ingestion_report(_payload())
    assert report["status"] == "passed"
    assert report["counts"] == {
        "shared_unseen": 100,
        "final_lhs": 1,
        "selection_training": 1,
        "total": 102,
    }
    assert report["blockers"] == []


def test_normalize_accepts_direct_list_and_json_path(tmp_path: Path) -> None:
    records = normalize_emb_34um_dnn_causal_validation_records(
        [_shared_unseen_record(1), _final_lhs_record(1), _selection_training_record(1)]
    )
    assert len(records) == 3

    path = tmp_path / "records.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    report = build_emb_34um_dnn_causal_validation_ingestion_report(path)
    assert report["status"] == "passed"


def test_missing_shared_unseen_blocker() -> None:
    payload = _payload()
    payload.pop("shared_unseen")
    report = build_emb_34um_dnn_causal_validation_ingestion_report(payload)
    assert report["status"] == "blocked"
    assert any("missing shared unseen test set" in blocker for blocker in report["blockers"])


def test_99_shared_unseen_blocker() -> None:
    report = build_emb_34um_dnn_causal_validation_ingestion_report(_payload(unseen_count=99))
    assert report["status"] == "blocked"
    assert any("exactly 100 records" in blocker for blocker in report["blockers"])


def test_wrong_shared_unseen_force_grid_blocker() -> None:
    wrong_grid = [0.0, 700.0, 1400.0, 2100.0, 2800.0, 3500.0, 4300.0, 5000.0]
    report = build_emb_34um_dnn_causal_validation_ingestion_report(
        _payload(unseen_kwargs={"force_grid": wrong_grid})
    )
    assert report["status"] == "blocked"
    assert any("force grid" in blocker.lower() for blocker in report["blockers"])


def test_reused_shared_unseen_blocker() -> None:
    report = build_emb_34um_dnn_causal_validation_ingestion_report(
        _payload(unseen_kwargs={"used_for_training": True})
    )
    assert report["status"] == "blocked"
    assert any("not be used for training" in blocker for blocker in report["blockers"])


def test_final_lhs_rejects_samples_all_source() -> None:
    report = build_emb_34um_dnn_causal_validation_ingestion_report(
        _payload(lhs_source="samples_all.dat")
    )
    assert report["status"] == "blocked"
    assert any("samples_all.dat" in blocker for blocker in report["blockers"])


@pytest.mark.parametrize(
    ("ensemble_size", "expected"),
    [
        (None, "must include ensemble_size=10"),
        (9, "fixed at 10"),
    ],
)
def test_selection_training_requires_ensemble_size_10(ensemble_size: object, expected: str) -> None:
    report = build_emb_34um_dnn_causal_validation_ingestion_report(
        _payload(training_ensemble_size=ensemble_size)
    )
    assert report["status"] == "blocked"
    assert any(expected in blocker for blocker in report["blockers"])
