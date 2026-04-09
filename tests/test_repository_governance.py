from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_codeowners_covers_repo_root() -> None:
    codeowners = (REPO_ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8")
    assert "* @BrieucB" in codeowners
    assert "/src/ @BrieucB" in codeowners
    assert "/tests/ @BrieucB" in codeowners


def test_dependabot_covers_actions_and_python() -> None:
    payload = yaml.safe_load((REPO_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8"))

    assert payload["version"] == 2
    updates = payload["updates"]
    ecosystems = {(entry["package-ecosystem"], entry["directory"]) for entry in updates}
    assert ("github-actions", "/") in ecosystems
    assert ("pip", "/") in ecosystems
    for entry in updates:
        assert entry["schedule"]["interval"] == "weekly"
        assert entry["open-pull-requests-limit"] == 5
        assert "dependencies" in entry["labels"]


def test_security_policy_documents_private_reporting() -> None:
    policy = (REPO_ROOT / "SECURITY.md").read_text(encoding="utf-8")

    assert "do **not** open a public GitHub issue" in policy
    assert "main" in policy
    assert "latest `v0.1.x` release line" in policy
