from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import meso_uq.active_learning.emb_34um_dnn_surrogate as module
from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import EMB_34UM_DNN_CAUSAL_FORCE_GRID


def _completed_rows() -> list[dict[str, object]]:
    force_grid = list(EMB_34UM_DNN_CAUSAL_FORCE_GRID)
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


def test_convert_completed_curves_to_long_rows_normalizes_force_and_logs_parameters() -> None:
    rows = module.convert_completed_curves_to_long_rows(_completed_rows(), force_grid=EMB_34UM_DNN_CAUSAL_FORCE_GRID)

    assert len(rows) == 16
    first = rows[0]
    last_first_curve = rows[7]
    assert first.curve_id == "curve-a"
    assert first.ka_log10 == pytest.approx(3.0)
    assert first.kb_log10 == pytest.approx(2.69897, rel=1e-5)
    assert first.force == 0.0
    assert first.force_norm == pytest.approx(0.0)
    assert last_first_curve.force == EMB_34UM_DNN_CAUSAL_FORCE_GRID[-1]
    assert last_first_curve.force_norm == pytest.approx(1.0)


def test_train_emb_34um_dnn_surrogate_ensemble_writes_ten_member_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _fake_train_single_member(**kwargs):
        checkpoint = kwargs["output_root"] / "members" / f"member_{kwargs['index']:02d}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(f"seed={kwargs['seed']}\n", encoding="utf-8")
        return (
            module._EnsembleMember(
                seed=int(kwargs["seed"]),
                checkpoint_path=checkpoint,
                train_loss_history=(0.5, 0.25),
                validation_loss_history=(0.6, 0.3),
                train_runtime_seconds=0.0,
            ),
            object(),
        )

    monkeypatch.setattr(module, "_train_single_member", _fake_train_single_member)

    fit, report = module.train_emb_34um_dnn_surrogate_ensemble(
        _completed_rows(),
        force_grid=EMB_34UM_DNN_CAUSAL_FORCE_GRID,
        output_root=tmp_path,
    )

    assert fit.ensemble_member_count == 10
    assert fit.ensemble_seeds == tuple(range(1, 11))
    assert len(fit.ensemble_checkpoints) == 10
    assert all(path.is_file() for path in fit.ensemble_checkpoints)
    assert len(fit.training_history) == 10
    assert report["ensemble_seeds"] == list(range(1, 11))

    manifest = json.loads((tmp_path / module.EMB_34UM_DNN_SURROGATE_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["ensemble_size"] == 10
    assert manifest["ensemble_checkpoints"][0].endswith("member_01.pt")


def test_train_emb_34um_dnn_surrogate_ensemble_rejects_lightweight_architecture(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lightweight"):
        module.train_emb_34um_dnn_surrogate_ensemble(
            _completed_rows(),
            architecture="dnn_mlp_64_64_lightweight",
            force_grid=EMB_34UM_DNN_CAUSAL_FORCE_GRID,
            output_root=tmp_path,
        )
