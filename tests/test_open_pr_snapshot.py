from __future__ import annotations

from scripts.qa.collect_open_pr_snapshot import summarize_checks


def test_summarize_checks_preserves_skips_failures_and_pending_state() -> None:
    summary = summarize_checks(
        [
            {"status": "COMPLETED", "conclusion": "SUCCESS"},
            {"status": "COMPLETED", "conclusion": "SKIPPED"},
            {"status": "COMPLETED", "conclusion": "FAILURE"},
            {"status": "IN_PROGRESS", "conclusion": None},
        ]
    )

    assert summary == {
        "count": 4,
        "incomplete": 1,
        "conclusions": {"FAILURE": 1, "NONE": 1, "SKIPPED": 1, "SUCCESS": 1},
    }
