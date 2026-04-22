from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_postprocess_figures_stops_on_comparison_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        Path("scripts/vega/run_postprocess_figures.py"),
        "run_postprocess_figures_failfast_test",
    )

    calls: list[str] = []

    def fake_run(cmd, label):  # noqa: ANN001
        del cmd
        calls.append(label)
        if label == "Family comparison + BNN decision":
            return 1
        return 0

    monkeypatch.setattr(module, "_run", fake_run)

    rc = module.main(["--run-root", str(tmp_path)])

    assert rc == 1
    assert calls == [
        "Holdout L2 figure",
        "Sensitivity figure",
        "Family comparison + BNN decision",
    ]


def test_run_postprocess_figures_runs_all_steps_on_success(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        Path("scripts/vega/run_postprocess_figures.py"),
        "run_postprocess_figures_success_test",
    )

    calls: list[str] = []

    def fake_run(cmd, label):  # noqa: ANN001
        del cmd
        calls.append(label)
        return 0

    monkeypatch.setattr(module, "_run", fake_run)

    rc = module.main(["--run-root", str(tmp_path)])

    assert rc == 0
    assert calls == [
        "Holdout L2 figure",
        "Sensitivity figure",
        "Family comparison + BNN decision",
        "UQ_DPD parity report",
        "Final report",
    ]
