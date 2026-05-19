from __future__ import annotations

import sys
from pathlib import Path

import pytest

if str((Path(__file__).resolve().parents[2] / "src")) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import meso_uq.active_learning.emb_34um_dpd_adapter as adapter


def test_load_emb_34um_force_grid_matches_training_file() -> None:
    grid = adapter.load_emb_34um_force_grid()

    assert len(grid) == 15
    assert grid[0] == 0.0
    assert list(grid) == sorted(grid)
    assert len(set(grid)) == len(grid)


def test_derive_emb_34um_canary_grid_is_three_points_from_full_grid() -> None:
    full_grid = adapter.derive_emb_34um_force_grid()
    canary_grid = adapter.derive_emb_34um_force_grid(canary=True)

    assert len(canary_grid) == adapter.EMB_34UM_CANARY_FORCE_POINT_COUNT
    assert canary_grid[0] == full_grid[0]
    assert canary_grid[-1] == full_grid[-1]
    assert set(canary_grid).issubset(set(full_grid))


def test_load_emb_34um_force_grid_rejects_inconsistent_rows(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad_samples.dat"
    bad_path.write_text(
        "\n".join(
            [
                "1 2 3 4 5 6 7 8 "
                + " ".join(str(index) for index in range(15))
                + " "
                + " ".join(f"{index:.1f}" for index in range(15)),
                "1 2 3 4 5 6 7 8 "
                + " ".join(str(index) for index in range(15))
                + " "
                + " ".join(f"{index:.1f}" for index in range(1, 16)),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Inconsistent force grids"):
        adapter.load_emb_34um_force_grid(data_path=bad_path)


def test_build_emb_34um_request_builds_payload_and_rejects_bad_dimensions(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    artifacts = adapter.build_emb_34um_request(
        candidate_id="candidate-emb",
        payload={"Yt": 1.2e6, "kb": 1000.0, "experiment": "indentation"},
        campaign_root=root,
    )
    request_payload = artifacts.normalized_request_payload()

    assert request_payload["schema_version"] == adapter.EMB_34UM_DPD_SCHEMA_VERSION
    assert request_payload["request_type"] == "emb_34um_full_force_sweep"
    assert request_payload["candidate_id"] == "candidate-emb"
    assert request_payload["parameters"]["Yt"] == 1.2e6
    assert request_payload["parameters"]["kb"] == 1000.0
    assert request_payload["retry_limit"] == adapter.EMB_34UM_RETRY_LIMIT
    assert request_payload["force_grid_count"] == 15
    assert request_payload["fingerprint"]["radp"] == adapter.EMB_34UM_RUNTIME_FINGERPRINT["radp"]
    assert request_payload["fingerprint"]["L"] == float(adapter.EMB_34UM_RUNTIME_FINGERPRINT["L"])
    assert request_payload["platform_defaults"]["platform"] == "karolina"

    with pytest.raises(ValueError, match="outside bounds"):
        adapter.build_emb_34um_request(
            candidate_id="too-low-yt",
            payload={"Yt": 1.0, "kb": 1000.0, "experiment": "indentation"},
            campaign_root=root,
        )
    with pytest.raises(ValueError, match="outside bounds"):
        adapter.build_emb_34um_request(
            candidate_id="too-high-kb",
            payload={"Yt": 1.2e6, "kb": 1234567.0, "experiment": "indentation"},
            campaign_root=root,
        )
    with pytest.raises(ValueError, match="missing required EMB 3.4um dimensions"):
        adapter.build_emb_34um_request(
            candidate_id="missing-kb",
            payload={"Yt": 1.2e6, "experiment": "indentation"},
            campaign_root=root,
        )


def test_build_emb_34um_request_rejects_wrong_fingerprint_defaults(monkeypatch, tmp_path: Path) -> None:
    def _bad_defaults(path: Path | None = None) -> dict[str, object]:
        return {
            "fscale": 0.01,
            "shell_th": adapter.EMB_34UM_RUNTIME_FINGERPRINT["shell_th"],
            "numsteps": adapter.EMB_34UM_RUNTIME_FINGERPRINT["numsteps"],
            "numsteps_eq": adapter.EMB_34UM_RUNTIME_FINGERPRINT["numsteps_eq"],
        }

    monkeypatch.setattr(adapter, "_load_default_parameters", _bad_defaults)

    with pytest.raises(ValueError, match="expected fscale"):
        adapter.build_emb_34um_request(
            candidate_id="bad-fingerprint",
            payload={"Yt": 1.2e6, "kb": 1000.0, "experiment": "indentation"},
            campaign_root=tmp_path,
        )
