from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "workflows"
    / "emb"
    / "uq_emb"
    / "materialize_hbi_config.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("materialize_hbi_config", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _definity_config() -> dict:
    return {
        "out": "/old/run",
        "pop_size": 50_000,
        "hbi_pop_size": 50_000,
        "phase3b_pop_size": 50_000,
        "hyperprior_mu_ka": [13_000.0, 18_000.0],
        "hyperprior_sigma_ka": [1_000.0, 2_000.0],
        "experiments": [
            {
                "name": "compression",
                "data_dir": "emb/compression/evalkit/data",
                "surrogate_dir": "emb/compression/surrogate/diameters",
                "diameters": [2.1, 2.9, 3.0],
            },
            {
                "name": "resonance",
                "lane": "source3",
                "grouped_reference_data": True,
                "data_dir": "emb/resonance/evalkit/data",
                "diameters": [4.68],
                "reference_diameters": [4.68, 5.18, 11.2],
            },
        ],
        "resonance": {
            "observations": [
                {"diameter_um": 4.68, "frequency_MHz": 1.69},
                {"diameter_um": 11.2, "frequency_MHz": 0.58},
            ],
            "evaluator": {
                "artifact_path": "/old/bank.json",
                "bank_build_report_path": "/old/report.json",
                "independent_go_path": "/old/go.json",
                "promotion_contract_path": "/old/promotion.json",
            },
        },
    }


def test_definity_rewrite_preserves_grouped_production_science(tmp_path: Path) -> None:
    module = _load_module()
    source = _definity_config()

    materialized, rewrites = module.rewrite_hbi_config(
        source,
        agent="definity",
        dependency_root=tmp_path / "dependencies",
        run_root=tmp_path / "run",
        population=10_000,
    )

    grouped = materialized["experiments"][1]
    assert grouped["lane"] == "source3"
    assert grouped["grouped_reference_data"] is True
    assert grouped["diameters"] == [4.68]
    assert grouped["reference_diameters"] == [4.68, 5.18, 11.2]
    assert materialized["resonance"]["observations"] == source["resonance"]["observations"]
    assert materialized["hyperprior_mu_ka"] == [13_000.0, 18_000.0]
    assert materialized["hyperprior_sigma_ka"] == [1_000.0, 2_000.0]
    assert materialized["pop_size"] == 10_000
    assert materialized["hbi_pop_size"] == 10_000
    assert materialized["phase3b_pop_size"] == 10_000
    assert source["out"] == "/old/run"
    assert len(rewrites) == 11


def test_materialize_validates_frozen_inputs_and_writes_receipt(tmp_path: Path) -> None:
    module = _load_module()
    artifact_root = tmp_path / "artifacts"
    accepted_root = artifact_root / module.ACCEPTED_SET
    dependency_root = artifact_root / module.DEPENDENCY_SET
    manifest_root = tmp_path / "manifests"
    config_relative = module.AGENT_PATHS["definity"]["config"]
    config_path = accepted_root / config_relative
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(_definity_config(), sort_keys=False), encoding="utf-8")

    dependency_relatives = (
        module.AGENT_PATHS["definity"]["bank"],
        module.AGENT_PATHS["definity"]["bank_report"],
        module.AGENT_PATHS["definity"]["independent_go"],
        module.PROMOTION_CONTRACT,
    )
    for relative in dependency_relatives:
        path = dependency_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative}\n", encoding="utf-8")

    manifest_root.mkdir()
    accepted_manifest = {
        "paper_id": module.PAPER_ID,
        "artifact_set_id": "accepted-production-outputs-202607",
        "locked": True,
        "files": [{"path": config_relative, "sha256": _sha256(config_path)}],
    }
    dependency_manifest = {
        "paper_id": module.PAPER_ID,
        "artifact_set_id": "frozen-runtime-dependencies-202607",
        "locked": True,
        "files": [
            {"path": relative, "sha256": _sha256(dependency_root / relative)}
            for relative in dependency_relatives
        ],
    }
    (manifest_root / f"{module.ACCEPTED_SET}.files.json").write_text(
        json.dumps(accepted_manifest), encoding="utf-8"
    )
    (manifest_root / f"{module.DEPENDENCY_SET}.files.json").write_text(
        json.dumps(dependency_manifest), encoding="utf-8"
    )

    receipt = module.materialize(
        agent="definity",
        artifact_root=artifact_root,
        manifest_root=manifest_root,
        output_dir=tmp_path / "configs",
        run_root=tmp_path / "run",
        population=50_000,
    )

    output_config = Path(receipt["materialized_config"])
    assert output_config.is_file()
    assert Path(receipt["receipt"]).is_file()
    assert receipt["source_config_sha256"] == _sha256(config_path)
    assert yaml.safe_load(output_config.read_text(encoding="utf-8"))["out"] == str(
        (tmp_path / "run").resolve()
    )


def test_materialize_rejects_mutated_dependency(tmp_path: Path) -> None:
    module = _load_module()
    artifact_root = tmp_path / "artifacts"
    accepted_root = artifact_root / module.ACCEPTED_SET
    dependency_root = artifact_root / module.DEPENDENCY_SET
    manifest_root = tmp_path / "manifests"
    config_relative = module.AGENT_PATHS["definity"]["config"]
    config_path = accepted_root / config_relative
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(_definity_config()), encoding="utf-8")
    bank_relative = module.AGENT_PATHS["definity"]["bank"]
    bank_path = dependency_root / bank_relative
    bank_path.parent.mkdir(parents=True)
    bank_path.write_text("mutated\n", encoding="utf-8")
    manifest_root.mkdir()
    (manifest_root / f"{module.ACCEPTED_SET}.files.json").write_text(
        json.dumps(
            {
                "paper_id": module.PAPER_ID,
                "artifact_set_id": "accepted-production-outputs-202607",
                "locked": True,
                "files": [{"path": config_relative, "sha256": _sha256(config_path)}],
            }
        ),
        encoding="utf-8",
    )
    (manifest_root / f"{module.DEPENDENCY_SET}.files.json").write_text(
        json.dumps(
            {
                "paper_id": module.PAPER_ID,
                "artifact_set_id": "frozen-runtime-dependencies-202607",
                "locked": True,
                "files": [{"path": bank_relative, "sha256": "0" * 64}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        module.materialize(
            agent="definity",
            artifact_root=artifact_root,
            manifest_root=manifest_root,
            output_dir=tmp_path / "configs",
            run_root=tmp_path / "run",
            population=10_000,
        )
