from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _parse_arg(command: list[str], flag: str) -> str:
    idx = command.index(flag)
    return command[idx + 1]


def _make_spec(tmp_path: Path, name: str) -> dict[str, str]:
    data = tmp_path / f"{name}.dat"
    dnn = tmp_path / f"{name}.pkl"
    script = tmp_path / f"{name}_train.py"
    for path in (data, dnn, script):
        path.write_text("placeholder", encoding="utf-8")
    return {
        "name": name,
        "modality": "compression",
        "diameter_um": "2.1",
        "data": str(data),
        "dnn_artifact": str(dnn),
        "bnn_artifact": str(tmp_path / f"{name}.pt"),
        "dnn_train_script": str(script),
        "dnn_multi_arch_script": str(script),
        "bnn_train_script": str(script),
        "group_holdout_script": str(script),
    }


def test_bnn_sweep_runner_passes_max_epochs_instead_of_max_steps(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "karolina" / "run_bnn_sweep_matrix.py",
        "run_bnn_sweep_matrix_max_epochs_test",
    )
    spec = _make_spec(tmp_path, "spec_a")
    monkeypatch.setattr(module, "resolve_emb_dataset_specs", lambda _root: [spec])

    called: list[list[str]] = []

    def fake_run(command, cwd, check):  # noqa: ANN001
        del cwd, check
        called.append(command)
        out_path = Path(_parse_arg(command, "--out"))
        report_path = Path(_parse_arg(command, "--report-path"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("artifact", encoding="utf-8")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({"training": {"best_val_rmse": 0.2, "final_val_rmse": 0.21}}),
            encoding="utf-8",
        )

        class _Done:
            returncode = 0

        return _Done()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(
        [
            "--output-root",
            str(tmp_path / "out"),
            "--seed",
            "101",
            "--architectures",
            "w32_d2",
            "--top-k",
            "1",
            "--prior-scales",
            "0.5",
            "--obs-noise-prior-scales",
            "0.1",
            "--lrs",
            "0.001",
            "--max-epochs",
            "25",
        ]
    )
    assert rc == 0
    assert called
    for command in called:
        assert "--max-epochs" in command
        assert "--max-steps" not in command

    matrix = json.loads((tmp_path / "out" / "bnn_sweep_matrix_report.json").read_text(encoding="utf-8"))
    assert matrix["config"]["max_epochs"] == 25
