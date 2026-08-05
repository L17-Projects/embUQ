from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/workflows/emb/uq_emb/render_figure6_replay.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("uq_emb_figure6_replay", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_figure6_verifies_locked_inputs_before_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    output = tmp_path / "output"
    inputs = {name: tmp_path / name for name in ("renderer.py", "style.py", "source.py", "rows.csv")}
    for path in inputs.values():
        path.write_text("\n", encoding="utf-8")
    tex_bin = tmp_path / "tex-bin"
    texdeps = tmp_path / "texdeps"
    tex_bin.mkdir()
    texdeps.mkdir()
    monkeypatch.setattr(
        module,
        "require_output_outside_consumed_roots",
        lambda *, output_path, **_kwargs: output_path.resolve(),
    )

    def reject_before_write(**_kwargs):
        assert not output.exists()
        raise ValueError("locked root drift")

    monkeypatch.setattr(module, "replay_receipt_provenance", reject_before_write)
    with pytest.raises(ValueError, match="locked root drift"):
        module.render_figure6(
            renderer=inputs["renderer.py"],
            paper_style=inputs["style.py"],
            paper_style_source=inputs["source.py"],
            rows=inputs["rows.csv"],
            tex_bin_dir=tex_bin,
            texdeps_dir=texdeps,
            output_dir=output,
            baseline_pdf=None,
        )
    assert not output.exists()


def test_figure6_rejects_lexical_input_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    renderer = tmp_path / "renderer.py"
    renderer.write_text("\n", encoding="utf-8")
    alias = tmp_path / "renderer-alias.py"
    alias.symlink_to(renderer)

    with pytest.raises(ValueError, match="symlinked path or ancestor"):
        module.render_figure6(
            renderer=alias,
            paper_style=tmp_path / "style.py",
            paper_style_source=tmp_path / "source.py",
            rows=tmp_path / "rows.csv",
            tex_bin_dir=tmp_path / "tex-bin",
            texdeps_dir=tmp_path / "texdeps",
            output_dir=tmp_path / "output",
            baseline_pdf=None,
        )
