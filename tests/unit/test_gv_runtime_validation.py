from __future__ import annotations

from pathlib import Path

from meso_uq.structures.gv.runtime.validation import (
    DiskCapValidationResult,
    LogValidationIssue,
    LogValidationResult,
    RuntimeCanaryValidationResult,
    evaluate_runtime_canary,
    evaluate_runs_disk_cap,
    measure_runs_footprint_bytes,
    summarize_log_validation,
    validate_mirheo_log_file,
    validate_mirheo_log_text,
    validate_mirheo_logs,
)


def test_validate_mirheo_log_text_accepts_clean_log() -> None:
    result = validate_mirheo_log_text("step 1 energy 2.0\nstep 2 energy 3.0\n")

    assert result.passed
    assert result.checked_lines == 2


def test_validate_mirheo_log_text_rejects_nan_vector() -> None:
    result = validate_mirheo_log_text("stats velocity [1.0 nan 2.0]\n", log_path="mirheo.log")

    assert not result.passed
    assert result.issues[0].token == "nan"
    assert result.issues[0].line_number == 1
    assert result.issues[0].log_path == Path("mirheo.log")


def test_validate_mirheo_log_text_rejects_inf_tokens() -> None:
    result = validate_mirheo_log_text("force +inf\npressure -Infinity\n")

    assert not result.passed
    assert [issue.token for issue in result.issues] == ["+inf", "-Infinity"]


def test_validate_mirheo_log_text_avoids_word_false_positives() -> None:
    text = "information: nanometer inflection finite\n"

    assert validate_mirheo_log_text(text).passed


def test_validate_mirheo_log_file_reports_excerpt(tmp_path: Path) -> None:
    log_path = tmp_path / "output.out"
    log_path.write_text("line 1\nforce: NaN after launch\n", encoding="utf-8")

    result = validate_mirheo_log_file(log_path)
    summary = summarize_log_validation([result])

    assert not result.passed
    assert result.issues[0].excerpt == "force: NaN after launch"
    assert "FAIL" in summary
    assert "NaN" in summary


def test_runtime_validation_result_helpers_cover_dict_and_limit_paths(tmp_path: Path) -> None:
    issue = LogValidationIssue(log_path=None, line_number=3, token="inf", excerpt="value inf")
    result = LogValidationResult(log_path=None, checked_lines=7, issues=(issue,))
    disk = DiskCapValidationResult(root=tmp_path, used_bytes=8, cap_bytes=7)
    canary = RuntimeCanaryValidationResult(log_results=(result,), disk_result=disk)

    assert issue.to_dict()["log_path"] is None
    assert result.to_dict() == {
        "log_path": None,
        "checked_lines": 7,
        "passed": False,
        "issues": [issue.to_dict()],
    }
    assert disk.to_dict()["passed"] is False
    assert canary.passed is False
    assert canary.to_dict()["disk"]["root"] == str(tmp_path)

    capped = validate_mirheo_log_text("nan\ninf\n", max_issues=1)
    assert capped.checked_lines == 1
    assert len(capped.issues) == 1

    long_line = "x" * 300 + "\n"
    assert validate_mirheo_log_text(long_line).checked_lines == 1
    assert len(validate_mirheo_log_text("value nan " + "x" * 300 + "\n").issues[0].excerpt) == 240


def test_evaluate_runs_disk_cap_passes_and_fails(tmp_path: Path) -> None:
    runs = tmp_path / "_runs"
    runs.mkdir()
    (runs / "nested").mkdir()
    (runs / "small.bin").write_bytes(b"12345")

    assert measure_runs_footprint_bytes(runs) == 5
    assert evaluate_runs_disk_cap(runs, cap_bytes=5).passed
    assert not evaluate_runs_disk_cap(runs, cap_bytes=4).passed
    assert measure_runs_footprint_bytes(tmp_path / "missing") == 0
    assert measure_runs_footprint_bytes(runs / "small.bin") == 5


def test_validate_mirheo_logs_and_runtime_canary(tmp_path: Path) -> None:
    runs = tmp_path / "_runs"
    runs.mkdir()
    first = runs / "clean.out"
    second = runs / "bad.out"
    first.write_text("step 1 ok\n", encoding="utf-8")
    second.write_text("step 2 inf\n", encoding="utf-8")

    results = validate_mirheo_logs([first, second])
    canary = evaluate_runtime_canary(log_paths=[first, second], runs_root=runs, disk_cap_bytes=1024)

    assert [result.passed for result in results] == [True, False]
    assert not canary.passed
    assert canary.disk_result.passed
    summary = summarize_log_validation(results)
    assert "PASS" in summary
    assert "FAIL" in summary
