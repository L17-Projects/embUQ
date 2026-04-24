from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ops_huq_emb_shim_delegates_to_maintained_runner(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "ops" / "huq_emb" / "run_paper_data_campaign.py",
        "ops_huq_emb_run_paper_data_campaign_test",
    )
    captured: dict[str, object] = {}

    def fake_run_path(path: str, run_name: str):
        captured["path"] = path
        captured["run_name"] = run_name

        def fake_main(argv: list[str]) -> int:
            captured["argv"] = argv
            return 7

        return {"main": fake_main}

    monkeypatch.setattr(module.runpy, "run_path", fake_run_path)

    rc = module.main(["--paper-data-root", "/tmp/paper_data", "--skip-workflow"])

    assert rc == 7
    assert captured["path"] == str(
        repo_root / "scripts" / "workflows" / "emb" / "huq_emb" / "run_paper_data_campaign.py"
    )
    assert captured["argv"] == ["--paper-data-root", "/tmp/paper_data", "--skip-workflow"]


def test_ops_huq_emb_shim_requires_target_main(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "ops" / "huq_emb" / "run_paper_data_campaign.py",
        "ops_huq_emb_run_paper_data_campaign_missing_main_test",
    )

    monkeypatch.setattr(module.runpy, "run_path", lambda *args, **kwargs: {})

    try:
        module._target_main()
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected missing main() guard to raise RuntimeError.")

    assert "callable main()" in message
    assert "scripts/workflows/emb/huq_emb/run_paper_data_campaign.py" in message
