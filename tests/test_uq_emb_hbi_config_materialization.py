from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from meso_uq.inference.emb_resonance import _resolve_relocated_provenance_path


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


def test_relocated_provenance_path_uses_explicit_override(tmp_path: Path) -> None:
    staged = tmp_path / "staged.csv"
    staged.write_text("frozen\n", encoding="utf-8")

    resolved = _resolve_relocated_provenance_path(
        tmp_path,
        "/missing/karolina/workspace/source.csv",
        {"/missing/karolina/workspace/source.csv": str(staged)},
    )

    assert resolved == staged.resolve()


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
    assert materialized["resonance"]["evaluator"]["provenance_path_overrides"] == {}
    assert len(rewrites) == 12


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
        module.POLYNOMIAL_BANK_BUILD_TOOL,
        *module.PROMOTION_SOURCES.values(),
        module.AGENT_PATHS["definity"]["bank_report"],
        module.AGENT_PATHS["definity"]["independent_go"],
        module.PROMOTION_CONTRACT,
    )
    for relative in dependency_relatives:
        path = dependency_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative}\n", encoding="utf-8")

    build_tool_path = dependency_root / module.POLYNOMIAL_BANK_BUILD_TOOL
    bank_path = dependency_root / module.AGENT_PATHS["definity"]["bank"]
    bank_path.write_text(
        json.dumps(
            {
                "provenance": {
                    "build_tool": {
                        "path": "/frozen/original/freeze_approved_polynomial_banks.py",
                        "sha256": _sha256(build_tool_path),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    promotion_records = {}
    for label, relative in module.PROMOTION_SOURCES.items():
        staged_path = dependency_root / relative
        promotion_records[label] = {
            "path": f"/frozen/original/{staged_path.name}",
            "sha256": _sha256(staged_path),
        }
    promotion_path = dependency_root / module.PROMOTION_CONTRACT
    promotion_path.write_text(json.dumps(promotion_records), encoding="utf-8")

    manifest_root.mkdir()
    accepted_manifest = {
        "paper_id": module.PAPER_ID,
        "artifact_set_id": "accepted-production-outputs-202607",
        "artifact_set_dir": module.ACCEPTED_SET,
        "locked": True,
        "file_count": 1,
        "logical_size_bytes": config_path.stat().st_size,
        "files": [
            {
                "path": config_relative,
                "size_bytes": config_path.stat().st_size,
                "sha256": _sha256(config_path),
            }
        ],
    }
    dependency_files = [
        {
            "path": relative,
            "size_bytes": (dependency_root / relative).stat().st_size,
            "sha256": _sha256(dependency_root / relative),
        }
        for relative in dependency_relatives
    ]
    dependency_manifest = {
        "paper_id": module.PAPER_ID,
        "artifact_set_id": "frozen-runtime-dependencies-202607",
        "artifact_set_dir": module.DEPENDENCY_SET,
        "locked": True,
        "file_count": len(dependency_files),
        "logical_size_bytes": sum(item["size_bytes"] for item in dependency_files),
        "files": dependency_files,
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
    assert len(receipt["semantic_config_sha256"]) == 64
    assert receipt["accepted_manifest_sha256"] == _sha256(
        manifest_root / f"{module.ACCEPTED_SET}.files.json"
    )
    assert receipt["dependency_manifest_sha256"] == _sha256(
        manifest_root / f"{module.DEPENDENCY_SET}.files.json"
    )
    assert receipt["accepted_source_verification"]["status"] == "PASS"
    assert receipt["dependency_verification"]["status"] == "PASS"
    assert len(receipt["provenance"]["git_commit"]) == 40
    assert yaml.safe_load(output_config.read_text(encoding="utf-8"))["out"] == str(
        (tmp_path / "run").resolve()
    )
    evaluator = yaml.safe_load(output_config.read_text(encoding="utf-8"))["resonance"][
        "evaluator"
    ]
    assert len(evaluator["provenance_path_overrides"]) == 4


def test_semantic_config_digest_ignores_only_relocated_roots(tmp_path: Path) -> None:
    module = _load_module()
    source = _definity_config()
    first_root = tmp_path / "karolina-artifacts"
    second_root = tmp_path / "vega-artifacts"
    override_sources = {
        "/legacy/a.py": "acoustic_surrogates/code/a.py",
        "/legacy/b.csv": "acoustic_surrogates/data/b.csv",
    }
    first, _ = module.rewrite_hbi_config(
        source,
        agent="definity",
        dependency_root=first_root,
        run_root=tmp_path / "karolina-run",
        population=10_000,
        provenance_path_overrides={
            key: str(first_root / relative) for key, relative in override_sources.items()
        },
    )
    second, _ = module.rewrite_hbi_config(
        source,
        agent="definity",
        dependency_root=second_root,
        run_root=tmp_path / "vega-run",
        population=10_000,
        provenance_path_overrides={
            key: str(second_root / relative) for key, relative in override_sources.items()
        },
    )

    assert module._semantic_config_sha256(
        first, agent="definity", dependency_root=first_root
    ) == module._semantic_config_sha256(
        second, agent="definity", dependency_root=second_root
    )

    second["hyperprior_mu_ka"] = [14_000.0, 18_000.0]
    assert module._semantic_config_sha256(
        first, agent="definity", dependency_root=first_root
    ) != module._semantic_config_sha256(
        second, agent="definity", dependency_root=second_root
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
                "artifact_set_dir": module.ACCEPTED_SET,
                "locked": True,
                "file_count": 1,
                "logical_size_bytes": config_path.stat().st_size,
                "files": [
                    {
                        "path": config_relative,
                        "size_bytes": config_path.stat().st_size,
                        "sha256": _sha256(config_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (manifest_root / f"{module.DEPENDENCY_SET}.files.json").write_text(
        json.dumps(
            {
                "paper_id": module.PAPER_ID,
                "artifact_set_id": "frozen-runtime-dependencies-202607",
                "artifact_set_dir": module.DEPENDENCY_SET,
                "locked": True,
                "file_count": 1,
                "logical_size_bytes": bank_path.stat().st_size,
                "files": [
                    {
                        "path": bank_relative,
                        "size_bytes": bank_path.stat().st_size,
                        "sha256": "0" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="content mismatch"):
        module.materialize(
            agent="definity",
            artifact_root=artifact_root,
            manifest_root=manifest_root,
            output_dir=tmp_path / "configs",
            run_root=tmp_path / "run",
            population=10_000,
        )
