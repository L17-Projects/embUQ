from pathlib import Path

from meso_uq.vega import (
    build_runtime_pythonpath,
    find_external_korali_entries,
    get_vega_paths,
    render_korali_env_script,
)


def _make_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    (repo_root / "extern" / "korali").mkdir(parents=True)
    (repo_root / "src").mkdir()
    (repo_root / "pyproject.toml").write_text("[project]\nname='mesouq'\n", encoding="utf-8")
    return repo_root


def test_vega_paths_use_repo_local_visible_state(tmp_path):
    repo_root = _make_repo(tmp_path)
    paths = get_vega_paths(repo_root)

    assert paths.vega_root == repo_root / "_vega"
    assert paths.korali_prefix == repo_root / "_vega" / "korali" / "install"
    assert paths.korali_env_script == repo_root / "_vega" / "korali" / "env.sh"


def test_build_runtime_pythonpath_prefers_repo_local_korali(tmp_path):
    repo_root = _make_repo(tmp_path)
    paths = get_vega_paths(repo_root)
    external_site = tmp_path / "external" / "lib" / "python3.10" / "site-packages"
    (external_site / "korali").mkdir(parents=True)
    other_entry = tmp_path / "other"
    other_entry.mkdir()

    built = build_runtime_pythonpath(
        repo_root,
        paths.korali_site_packages,
        current_pythonpath=f"{external_site}:{other_entry}",
        include_existing=True,
    )

    entries = built.split(":")
    assert entries[:3] == [
        str(paths.korali_site_packages),
        str(repo_root / "src"),
        str(repo_root),
    ]
    assert str(external_site) not in entries
    assert str(other_entry) in entries


def test_find_external_korali_entries_reports_user_global_path(tmp_path):
    repo_root = _make_repo(tmp_path)
    paths = get_vega_paths(repo_root)
    external_site = tmp_path / "external" / "lib" / "python3.10" / "site-packages"
    (external_site / "korali").mkdir(parents=True)

    external = find_external_korali_entries(
        f"{external_site}:{repo_root / 'src'}",
        repo_root,
        paths.korali_site_packages,
    )

    assert external == [str(external_site)]


def test_render_korali_env_script_uses_deterministic_pythonpath(tmp_path):
    repo_root = _make_repo(tmp_path)
    paths = get_vega_paths(repo_root)
    env_script = render_korali_env_script(paths)

    assert "KORALI_PYTHONPATH" in env_script
    assert str(paths.korali_site_packages) in env_script
    assert f"{repo_root / 'src'}:{repo_root}" in env_script
    assert "replaces inherited PYTHONPATH" in env_script
