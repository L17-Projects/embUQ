from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "GV_PAPER_REPLAY_CLOSEOUT.md"


def test_gv_paper_replay_closeout_doc_locks_gv_only_inventory() -> None:
    text = DOC.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    for expected in (
        "Figure 3, GV stretching",
        "Figure 7, GV pressure buckling",
        "Figure 8, GV eigenmodes",
        "SI-backed GV torsion diagnostic",
        "EMB figures are excluded",
    ):
        assert expected in text
    assert "GV `shear_flow` / Figure 9 is also excluded" in normalized

    assert "_runs/gv/figure_replay/<campaign>/lanes/stretching" in text
    assert "_runs/gv/figure_replay/<campaign>/lanes/buckling" in text
    assert "_runs/gv/figure_replay/<campaign>/lanes/eigenmodes" in text
    assert "_runs/gv/figure_replay/<campaign>/lanes/torsion" in text


def test_gv_paper_replay_closeout_doc_records_profile_provenance_and_blocker() -> None:
    text = DOC.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "20eb25ba752f5daea116ffcf9a382d0ea6c0e9286ed7744d9a8544b6fcfae182" in text
    assert "6559522c2d7e2d428b5f190815447b8e421fb5190f29bf3312c1f8db61d8f97e" in text
    assert "7a7ff7e98b1ce03cd39ce25c06a7ce4769e26a8a7d24fb6958da95a0b1630715" in text
    assert "runtime_default" in text
    assert "not `paper_confirmed`" in normalized
    assert "exact paper-confirmed GV geometry and nine-material-parameter tuple" in normalized
    assert "Dropped GV paper scripts are not part of the checked-in source tree" in text


def test_gv_docs_link_to_paper_replay_closeout() -> None:
    docs_to_check = (
        REPO_ROOT / "docs" / "README.md",
        REPO_ROOT / "docs" / "GV_NUMERICAL_DATA_GENERATION.md",
        REPO_ROOT / "docs" / "GV_EXTENSION_CLOSEOUT.md",
    )

    for path in docs_to_check:
        text = path.read_text(encoding="utf-8")
        assert "GV_PAPER_REPLAY_CLOSEOUT.md" in text
