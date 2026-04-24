from pathlib import Path

from meso_uq.vega import (
    build_runtime_pythonpath,
    find_external_korali_entries,
    gather_mirheo_source_snapshot,
    get_vega_paths,
    load_mirheo_source_lock,
    render_mirheo_env_script,
    render_korali_env_script,
    render_tinytex_env_script,
    resolve_repo_root,
    resolve_mirheo_source,
)


def _make_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    (repo_root / "extern" / "korali").mkdir(parents=True)
    (repo_root / "extern").mkdir(exist_ok=True)
    (repo_root / "src").mkdir()
    (repo_root / "pyproject.toml").write_text("[project]\nname='mesouq'\n", encoding="utf-8")
    return repo_root


def test_vega_paths_use_repo_local_visible_state(tmp_path):
    repo_root = _make_repo(tmp_path)
    paths = get_vega_paths(repo_root)

    assert paths.vega_root == repo_root / "_vega"
    assert paths.korali_prefix == repo_root / "_vega" / "korali" / "install"
    assert paths.korali_env_script == repo_root / "_vega" / "korali" / "env.sh"
    assert paths.tinytex_root == repo_root / "_vega" / "tinytex"
    assert paths.tinytex_env_script == repo_root / "_vega" / "tinytex" / "env.sh"
    assert paths.mirheo_prefix == repo_root / "_vega" / "mirheo" / "install"
    assert paths.mirheo_env_script == repo_root / "_vega" / "mirheo" / "env.sh"


def test_resolve_repo_root_finds_root_from_nested_file(tmp_path):
    repo_root = _make_repo(tmp_path)
    nested_file = repo_root / "scripts" / "platforms" / "vega" / "doctor_vega.py"
    nested_file.parent.mkdir(parents=True, exist_ok=True)
    nested_file.write_text("# test\n", encoding="utf-8")

    assert resolve_repo_root(nested_file) == repo_root


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


def test_render_tinytex_env_script_exports_repo_local_bin(tmp_path):
    repo_root = _make_repo(tmp_path)
    paths = get_vega_paths(repo_root)

    env_script = render_tinytex_env_script(paths)

    assert "MESOUQ_TINYTEX_ROOT" in env_script
    assert str(paths.tinytex_root) in env_script
    assert str(paths.tinytex_bin_dir) in env_script


def test_resolve_mirheo_source_uses_repo_lock(tmp_path, monkeypatch):
    repo_root = _make_repo(tmp_path)
    source_root = tmp_path / "mirheo-src"
    source_root.mkdir()
    (repo_root / "extern" / "mirheo.lock.json").write_text(
        f'{{"source_path": "{source_root}"}}',
        encoding="utf-8",
    )
    monkeypatch.delenv("MESOUQ_MIRHEO_SRC", raising=False)

    resolved = resolve_mirheo_source(repo_root)

    assert resolved == source_root.resolve()
    assert load_mirheo_source_lock(repo_root)["source_path"] == str(source_root)


def test_load_mirheo_source_lock_defaults_when_lock_missing(tmp_path):
    repo_root = _make_repo(tmp_path)

    lock = load_mirheo_source_lock(repo_root)

    assert "source_path" in lock


def test_load_mirheo_source_lock_rejects_non_mapping_payload(tmp_path):
    repo_root = _make_repo(tmp_path)
    (repo_root / "extern" / "mirheo.lock.json").write_text('["bad"]', encoding="utf-8")

    try:
        load_mirheo_source_lock(repo_root)
    except ValueError as exc:
        assert "Invalid Mirheo source lock payload" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for non-mapping Mirheo lock payload")


def test_resolve_mirheo_source_prefers_explicit_override(tmp_path):
    repo_root = _make_repo(tmp_path)
    override_root = tmp_path / "override-src"
    override_root.mkdir()

    assert resolve_mirheo_source(repo_root, override=override_root) == override_root.resolve()


def test_render_mirheo_env_script_exports_local_paths(tmp_path):
    repo_root = _make_repo(tmp_path)
    paths = get_vega_paths(repo_root)
    source_root = tmp_path / "Mirheo"
    source_root.mkdir()

    env_script = render_mirheo_env_script(
        paths,
        source_root=source_root,
        snapshot_path=paths.mirheo_snapshot_path,
    )

    assert "MESOUQ_MIRHEO_SRC" in env_script
    assert str(source_root.resolve()) in env_script
    assert str(paths.mirheo_build_dir) in env_script
    assert str(paths.mirheo_snapshot_path) in env_script


def test_gather_mirheo_source_snapshot_ignores_build_artifacts(tmp_path):
    source_root = tmp_path / "Mirheo"
    (source_root / "mirheo").mkdir(parents=True)
    (source_root / "build").mkdir()
    (source_root / "mirheo" / "__init__.py").write_text("value = 1\n", encoding="utf-8")
    (source_root / "build" / "junk.txt").write_text("ignored\n", encoding="utf-8")

    snapshot = gather_mirheo_source_snapshot(source_root)

    assert snapshot["file_count"] == 1
    assert snapshot["total_bytes"] == len("value = 1\n")
    assert snapshot["tree_sha256"]
