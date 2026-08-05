from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "qa" / "plan_repository_quarantine.py"


def _run(*args: str, cwd: Path) -> None:
    subprocess.run([*args], cwd=cwd, check=True, capture_output=True, text=True)


def test_quarantine_planner_never_moves_or_deletes_candidates(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "scripts/platforms/vega").mkdir(parents=True)
    (repo / "scripts/misc").mkdir(parents=True)
    (repo / "papers/UQ_EMB/manifests").mkdir(parents=True)
    (repo / "scripts/platforms/vega/job.py").write_text("print('vega')\n", encoding="utf-8")
    (repo / "scripts/misc/used.py").write_text("print('used')\n", encoding="utf-8")
    (repo / "scripts/misc/review.py").write_text("print('review')\n", encoding="utf-8")
    (repo / "scripts/misc/pr_only.py").write_text("print('pr')\n", encoding="utf-8")
    (repo / "README.md").write_text("Run scripts/misc/used.py.\n", encoding="utf-8")
    (repo / "papers/UQ_EMB/manifests/accepted.json").write_text("{}\n", encoding="utf-8")
    _run("git", "init", cwd=repo)
    _run("git", "add", ".", cwd=repo)
    _run(
        "git",
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "fixture",
        cwd=repo,
    )

    candidates = [
        "scripts/platforms/vega/job.py",
        "scripts/misc/used.py",
        "scripts/misc/review.py",
        "scripts/misc/pr_only.py",
    ]
    orphan = {
        "by_subsystem": {
            "fixture": [
                {"path": path, "module": path.removesuffix(".py").replace("/", ".")}
                for path in candidates
            ]
        }
    }
    orphan_path = tmp_path / "orphan.json"
    orphan_path.write_text(json.dumps(orphan), encoding="utf-8")
    pr_path = tmp_path / "prs.json"
    pr_path.write_text(
        json.dumps(
            {
                "pull_requests": [
                    {"number": 9, "changed_files": ["scripts/misc/pr_only.py"]}
                ]
            }
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / "plan.json"
    output_md = tmp_path / "plan.md"
    quarantine = tmp_path / "quarantine"

    _run(
        sys.executable,
        str(SCRIPT),
        "--repo-root",
        str(repo),
        "--orphan-report",
        str(orphan_path),
        "--open-pr-snapshot",
        str(pr_path),
        "--quarantine-root",
        str(quarantine),
        "--manifest-dir",
        str(repo / "papers/UQ_EMB/manifests"),
        "--output-json",
        str(output_json),
        "--output-markdown",
        str(output_md),
        cwd=repo,
    )

    plan = json.loads(output_json.read_text(encoding="utf-8"))
    by_path = {row["path"]: row for row in plan["records"]}
    assert plan["policy"]["moves_performed"] is False
    assert plan["policy"]["deletions_performed"] is False
    assert by_path["scripts/platforms/vega/job.py"]["classification"] == "false_positive"
    assert by_path["scripts/misc/used.py"]["classification"] == "document"
    assert by_path["scripts/misc/review.py"]["classification"] == "needs_owner_review"
    assert by_path["scripts/misc/pr_only.py"]["classification"] == "keep"
    assert not quarantine.exists()
    for relative in candidates:
        assert (repo / relative).is_file()
