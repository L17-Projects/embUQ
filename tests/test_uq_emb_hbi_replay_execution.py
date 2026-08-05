from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
REPLAY_SCRIPT = REPO_ROOT / "scripts" / "workflows" / "emb" / "uq_emb" / "run_hbi_replay.py"
WRAPPERS = {
    "phase1": REPO_ROOT / "reduced" / "scripts" / "run_phase_1.py",
    "phase2": REPO_ROOT / "reduced" / "scripts" / "run_phase_2.py",
    "phase3b": REPO_ROOT / "reduced" / "scripts" / "run_phase_3b.py",
}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _binding(config_path: Path) -> dict[str, object]:
    from replay_provenance import sha256

    return {
        "config_sha256": sha256(config_path),
        "accepted_stage_seeds": {"phase1": 1101, "phase2": 2101, "phase3b": 3104},
        "accepted_phase3b_seed_mode": "repeat",
        "accepted_artifact_root": str(config_path.parent / "accepted"),
        "dependency_artifact_root": str(config_path.parent / "dependencies"),
    }


def test_execute_uses_private_snapshot_and_records_accepted_stage_seeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load(REPLAY_SCRIPT, "uq_emb_hbi_replay_execution")
    config_path = tmp_path / "materialized.yaml"
    config_path.write_text("replay: accepted\n", encoding="utf-8")
    output_root = tmp_path / "replay"
    captured: dict[str, object] = {}

    monkeypatch.setattr(module, "load_materialization_binding", lambda path, **_: _binding(path))
    monkeypatch.setattr(module, "runtime_provenance", lambda **_: {"git_commit": "test"})

    def build_commands(**kwargs):
        captured["config_path"] = kwargs["config_path"]
        captured["stage_seeds"] = kwargs["stage_seeds"]
        captured["phase3b_seed_mode"] = kwargs["phase3b_seed_mode"]
        return "definity", 10_000, [[stage] for stage in kwargs["stages"]]

    monkeypatch.setattr(module, "build_replay_commands", build_commands)
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: None)

    receipt = module.run_replay(
        config_path=config_path,
        output_root=output_root,
        python_bin="/verified/python",
        site="karolina",
        stages=["phase1", "phase2", "phase3b"],
        execute=True,
    )

    snapshot = output_root / "runtime_inputs" / "hbi_config.yaml"
    assert captured["config_path"] == snapshot
    assert captured["stage_seeds"] == {"phase1": 1101, "phase2": 2101, "phase3b": 3104}
    assert receipt["accepted_stage_seeds"] == captured["stage_seeds"]
    assert captured["phase3b_seed_mode"] == "repeat"
    assert [item["korali_random_seed"] for item in receipt["stage_results"]] == [
        1101,
        2101,
        3104,
    ]
    assert all(item["run_input_before"] == item["run_input_after"] for item in receipt["stage_results"])
    assert receipt["stage_results"][-1]["phase3b_seed_mode"] == "repeat"


@pytest.mark.parametrize("locked_key", ("accepted_artifact_root", "dependency_artifact_root"))
def test_replay_rejects_output_inside_locked_artifact_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, locked_key: str
) -> None:
    module = _load(REPLAY_SCRIPT, f"uq_emb_hbi_replay_locked_output_{locked_key}")
    config_path = tmp_path / "materialized.yaml"
    config_path.write_text("replay: accepted\n", encoding="utf-8")
    binding = _binding(config_path)
    forbidden_output = Path(str(binding[locked_key])) / "replay"
    monkeypatch.setattr(module, "load_materialization_binding", lambda path, **_: binding)

    with pytest.raises(ValueError, match="outside immutable artifact roots"):
        module.run_replay(
            config_path=config_path,
            output_root=forbidden_output,
            python_bin="/verified/python",
            site="karolina",
            stages=["phase1"],
            execute=False,
        )

    assert not forbidden_output.exists()


@pytest.mark.parametrize("stage", ("phase1", "phase2", "phase3b"))
def test_replay_rejects_preexisting_selected_stage_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stage: str
) -> None:
    module = _load(REPLAY_SCRIPT, f"uq_emb_hbi_replay_freshness_{stage}")
    config_path = tmp_path / "materialized.yaml"
    config_path.write_text("replay: accepted\n", encoding="utf-8")
    output_root = tmp_path / "replay"
    (output_root / module.STAGE_OUTPUTS[stage]).mkdir(parents=True)
    monkeypatch.setattr(
        module,
        "load_materialization_binding",
        lambda *_args, **_kwargs: pytest.fail("freshness must be checked before provenance loading"),
    )

    with pytest.raises(FileExistsError, match=f"existing {stage} output"):
        module.run_replay(
            config_path=config_path,
            output_root=output_root,
            python_bin="/verified/python",
            site="karolina",
            stages=[stage],
            execute=False,
        )


def test_replay_allows_prerequisite_outputs_but_requires_them_when_not_selected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load(REPLAY_SCRIPT, "uq_emb_hbi_replay_prerequisites")
    config_path = tmp_path / "materialized.yaml"
    config_path.write_text("replay: accepted\n", encoding="utf-8")
    output_root = tmp_path / "replay"

    with pytest.raises(FileNotFoundError, match="phase2 requires existing phase1 output"):
        module.run_replay(
            config_path=config_path,
            output_root=output_root,
            python_bin="/verified/python",
            site="karolina",
            stages=["phase2"],
            execute=False,
        )

    (output_root / module.STAGE_OUTPUTS["phase1"]).mkdir(parents=True)
    monkeypatch.setattr(module, "load_materialization_binding", lambda path, **_: _binding(path))
    monkeypatch.setattr(module, "runtime_provenance", lambda **_: {"git_commit": "test"})
    monkeypatch.setattr(
        module,
        "build_replay_commands",
        lambda **kwargs: ("definity", 10_000, [["phase2"]]),
    )

    receipt = module.run_replay(
        config_path=config_path,
        output_root=output_root,
        python_bin="/verified/python",
        site="karolina",
        stages=["phase2"],
        execute=False,
    )
    assert receipt["status"] == "planned"


def test_replay_rejects_run_input_mutation_between_stages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load(REPLAY_SCRIPT, "uq_emb_hbi_replay_mutation")
    config_path = tmp_path / "materialized.yaml"
    config_path.write_text("replay: accepted\n", encoding="utf-8")
    output_root = tmp_path / "replay"
    captured: dict[str, Path] = {}

    monkeypatch.setattr(module, "load_materialization_binding", lambda path, **_: _binding(path))
    monkeypatch.setattr(module, "runtime_provenance", lambda **_: {"git_commit": "test"})

    def build_commands(**kwargs):
        captured["snapshot"] = kwargs["config_path"]
        return "definity", 10_000, [[stage] for stage in kwargs["stages"]]

    def mutate_snapshot(*_args, **_kwargs):
        snapshot = captured["snapshot"]
        snapshot.chmod(0o644)
        snapshot.write_text("replay: mutated\n", encoding="utf-8")

    monkeypatch.setattr(module, "build_replay_commands", build_commands)
    monkeypatch.setattr(module.subprocess, "run", mutate_snapshot)

    with pytest.raises(ValueError, match="snapshot changed after verification"):
        module.run_replay(
            config_path=config_path,
            output_root=output_root,
            python_bin="/verified/python",
            site="karolina",
            stages=["phase1", "phase2", "phase3b"],
            execute=True,
        )

    receipt = json.loads((output_root / "uq_emb_hbi_replay_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["stage_results"] == []


def test_replay_isolated_from_later_materialized_config_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load(REPLAY_SCRIPT, "uq_emb_hbi_replay_source_mutation")
    config_path = tmp_path / "materialized.yaml"
    config_path.write_text("replay: accepted\n", encoding="utf-8")
    output_root = tmp_path / "replay"
    captured: dict[str, Path] = {}

    monkeypatch.setattr(module, "load_materialization_binding", lambda path, **_: _binding(path))
    monkeypatch.setattr(module, "runtime_provenance", lambda **_: {"git_commit": "test"})

    def build_commands(**kwargs):
        captured["snapshot"] = kwargs["config_path"]
        return "definity", 10_000, [[stage] for stage in kwargs["stages"]]

    def mutate_source(*_args, **_kwargs):
        config_path.write_text("replay: later mutation\n", encoding="utf-8")

    monkeypatch.setattr(module, "build_replay_commands", build_commands)
    monkeypatch.setattr(module.subprocess, "run", mutate_source)

    receipt = module.run_replay(
        config_path=config_path,
        output_root=output_root,
        python_bin="/verified/python",
        site="karolina",
        stages=["phase1", "phase2", "phase3b"],
        execute=True,
    )

    assert receipt["status"] == "passed"
    assert captured["snapshot"].read_text(encoding="utf-8") == "replay: accepted\n"
    assert all(item["run_input_before"] == item["run_input_after"] for item in receipt["stage_results"])


def test_replay_preserves_config_alias_for_provenance_rejection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load(REPLAY_SCRIPT, "uq_emb_hbi_replay_alias")
    config_path = tmp_path / "materialized.yaml"
    config_path.write_text("replay: accepted\n", encoding="utf-8")
    alias_path = tmp_path / "materialized-alias.yaml"
    alias_path.symlink_to(config_path)

    def reject_alias(path: Path, **_kwargs):
        assert path == alias_path
        raise ValueError("symlinked path")

    monkeypatch.setattr(module, "load_materialization_binding", reject_alias)
    with pytest.raises(ValueError, match="symlinked path"):
        module.run_replay(
            config_path=alias_path,
            output_root=tmp_path / "replay",
            python_bin="/verified/python",
            site="karolina",
            stages=["phase1"],
            execute=False,
        )


@pytest.mark.parametrize("stage", ("phase1", "phase2", "phase3b"))
def test_reduced_wrapper_forwards_korali_seed(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    module = _load(WRAPPERS[stage], f"uq_emb_reduced_wrapper_{stage}")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        module.sys,
        "argv",
        [str(WRAPPERS[stage]), "--korali-random-seed", "123"],
    )
    monkeypatch.setattr(
        module.subprocess,
        "call",
        lambda command, **kwargs: captured.update(command=command, kwargs=kwargs) or 0,
    )

    assert module.main() == 0
    command = captured["command"]
    assert command[-2:] == ["--korali-random-seed", "123"]
