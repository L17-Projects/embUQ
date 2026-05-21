import os
import shlex
import subprocess
from pathlib import Path

from meso_uq.site_runtime import normalize_runtime_site
from meso_uq.vega import (
    build_runtime_pythonpath,
    build_unified_env_pythonpath,
    discover_python_runtime_library_dirs,
    discover_hdf5_runtime_roots,
    find_external_korali_entries,
    gather_mirheo_source_snapshot,
    get_runtime_paths,
    get_vega_paths,
    load_mirheo_source_lock,
    render_gv_cgal_tools_env_script,
    render_mirheo_env_script,
    render_korali_env_script,
    render_tinytex_env_script,
    render_unified_env_script,
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
    assert paths.env_root == repo_root / "_vega" / "env"
    assert paths.env_script == repo_root / "_vega" / "env" / "env.sh"
    assert paths.korali_prefix == repo_root / "_vega" / "korali" / "install"
    assert paths.korali_env_script == repo_root / "_vega" / "korali" / "env.sh"
    assert paths.tinytex_root == repo_root / "_vega" / "tinytex"
    assert paths.tinytex_env_script == repo_root / "_vega" / "tinytex" / "env.sh"
    assert paths.mirheo_prefix == repo_root / "_vega" / "mirheo" / "install"
    assert paths.mirheo_env_script == repo_root / "_vega" / "mirheo" / "env.sh"


def test_runtime_paths_support_karolina_repo_local_root(tmp_path):
    repo_root = _make_repo(tmp_path)
    paths = get_runtime_paths(repo_root, site="karolina", env={})

    assert paths.site == "karolina"
    assert paths.site_root == repo_root / "_karolina"
    assert paths.karolina_root == repo_root / "_karolina"
    assert paths.korali_prefix == repo_root / "_karolina" / "korali" / "install"
    assert paths.mirheo_prefix == repo_root / "_karolina" / "mirheo" / "install"
    assert paths.gv_cgal_tools_env_script == repo_root / "_karolina" / "gv_cgal_tools" / "env.sh"
    assert paths.scale_space_binary == repo_root / "_karolina" / "gv_cgal_tools" / "bin" / "scale_space"
    assert paths.provenance_root == repo_root / "provenance"


def test_runtime_paths_honor_site_runtime_root_override(tmp_path):
    repo_root = _make_repo(tmp_path)
    runtime_root = tmp_path / "scratch" / "runtime"
    paths = get_runtime_paths(
        repo_root,
        site="karolina",
        env={"MESOUQ_SITE_RUNTIME_ROOT": str(runtime_root)},
    )

    assert paths.site_root == runtime_root.resolve()
    assert paths.env_root == runtime_root.resolve() / "env"
    assert paths.env_script == runtime_root.resolve() / "env" / "env.sh"
    assert paths.venv_root == runtime_root.resolve() / "venv"
    assert paths.korali_env_script == runtime_root.resolve() / "korali" / "env.sh"
    assert paths.provenance_root == runtime_root.resolve().parent / "provenance"

    explicit_paths = get_runtime_paths(
        repo_root,
        site="karolina",
        runtime_root=runtime_root,
        env={},
    )

    assert explicit_paths.site_root == runtime_root.resolve()


def test_runtime_paths_honor_karolina_provenance_root_override(tmp_path):
    repo_root = _make_repo(tmp_path)
    provenance_root = tmp_path / "scratch" / "provenance"
    paths = get_runtime_paths(
        repo_root,
        site="karolina",
        env={"MESOUQ_PROVENANCE_ROOT": str(provenance_root)},
    )

    assert paths.provenance_root == provenance_root.resolve()


def test_normalize_runtime_site_rejects_unknown_site():
    try:
        normalize_runtime_site("unknown")
    except ValueError as exc:
        assert "Unsupported runtime site" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for an unsupported runtime site.")


def test_resolve_repo_root_finds_root_from_nested_file(tmp_path):
    repo_root = _make_repo(tmp_path)
    nested_file = repo_root / "scripts" / "platforms" / "vega" / "doctor_vega.py"
    nested_file.parent.mkdir(parents=True, exist_ok=True)
    nested_file.write_text("# test\n", encoding="utf-8")

    assert resolve_repo_root(nested_file) == repo_root


def test_resolve_repo_root_raises_when_no_repo_marker_exists(tmp_path):
    orphan = tmp_path / "not-a-repo" / "nested"
    orphan.mkdir(parents=True)

    try:
        resolve_repo_root(orphan)
    except FileNotFoundError as exc:
        assert "Could not locate the MesoUQ repo root" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected FileNotFoundError outside a MesoUQ checkout.")


def test_build_runtime_pythonpath_prefers_repo_local_korali(tmp_path):
    repo_root = _make_repo(tmp_path)
    paths = get_vega_paths(repo_root)
    external_site = tmp_path / "external" / "lib" / "python-external" / "site-packages"
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
    external_site = tmp_path / "external" / "lib" / "python-external" / "site-packages"
    (external_site / "korali").mkdir(parents=True)

    external = find_external_korali_entries(
        f"{external_site}:{repo_root / 'src'}",
        repo_root,
        paths.korali_site_packages,
    )

    assert external == [str(external_site)]


def test_build_unified_env_pythonpath_prefers_env_then_local_korali(tmp_path):
    repo_root = _make_repo(tmp_path)
    paths = get_vega_paths(repo_root)
    external_site = tmp_path / "external" / "lib" / "python-external" / "site-packages"
    (external_site / "korali").mkdir(parents=True)

    built = build_unified_env_pythonpath(
        repo_root,
        paths.env_site_packages,
        paths.korali_site_packages,
        current_pythonpath=str(external_site),
        include_existing=True,
    )

    entries = built.split(":")
    assert entries[:4] == [
        str(paths.env_site_packages),
        str(paths.korali_site_packages),
        str(repo_root / "src"),
        str(repo_root),
    ]
    assert str(external_site) not in entries


def test_discover_python_runtime_library_dirs_honors_env_override(tmp_path):
    lib_dir = tmp_path / "python" / "lib"
    ffi_dir = tmp_path / "libffi" / "lib64"
    lib_dir.mkdir(parents=True)
    ffi_dir.mkdir(parents=True)
    (lib_dir / "libpython3.10.so.1.0").touch()

    discovered = discover_python_runtime_library_dirs(
        env={
            "MESOUQ_PYTHON_LIB_DIR": str(lib_dir),
            "LD_LIBRARY_PATH": str(ffi_dir),
        }
    )

    assert lib_dir.resolve() in discovered
    assert ffi_dir.resolve() in discovered


def test_render_unified_env_script_exports_single_canonical_activation(tmp_path):
    repo_root = _make_repo(tmp_path)
    paths = get_runtime_paths(repo_root, site="karolina", env={})
    source_root = tmp_path / "Mirheo"
    source_root.mkdir()
    hdf5_root = tmp_path / "hdf5"
    (hdf5_root / "lib").mkdir(parents=True)
    python_lib = tmp_path / "python" / "lib"
    module_bin = tmp_path / "module" / "bin"
    pkg_config = tmp_path / "module" / "lib" / "pkgconfig"
    python_lib.mkdir(parents=True)
    module_bin.mkdir(parents=True)
    pkg_config.mkdir(parents=True)

    env_script = render_unified_env_script(
        paths,
        source_root=source_root,
        snapshot_path=paths.mirheo_snapshot_path,
        hdf5_runtime_roots=(hdf5_root,),
        python_runtime_lib_dirs=(python_lib,),
        captured_env={"PATH": str(module_bin), "PKG_CONFIG_PATH": str(pkg_config)},
    )

    assert "MESOUQ_SITE=karolina" in env_script
    assert f"MESOUQ_ENV_ROOT={paths.env_root}" in env_script
    assert f"MESOUQ_ENV_SCRIPT={paths.env_script}" in env_script
    assert f"MESOUQ_GV_ENV_SCRIPT={paths.env_script}" in env_script
    assert str(paths.env_site_packages) in env_script
    assert str(paths.korali_site_packages) in env_script
    assert str(paths.korali_env_script) not in env_script
    assert str(paths.mirheo_env_script) in env_script
    assert str(paths.gv_cgal_tools_env_script) in env_script
    assert str(hdf5_root) in env_script
    assert str(python_lib) in env_script
    assert str(module_bin) in env_script
    assert str(pkg_config) in env_script
    assert "MESOUQ_OPENMPI_LIB_DIR" in env_script



def test_render_korali_env_script_uses_deterministic_pythonpath(tmp_path):
    repo_root = _make_repo(tmp_path)
    paths = get_vega_paths(repo_root)
    env_script = render_korali_env_script(paths)

    assert "KORALI_PYTHONPATH" in env_script
    assert str(paths.korali_site_packages) in env_script
    assert f"{repo_root / 'src'}:{repo_root}" in env_script
    assert "replaces inherited PYTHONPATH" in env_script


def test_render_korali_env_script_exports_karolina_runtime_root(tmp_path):
    repo_root = _make_repo(tmp_path)
    paths = get_runtime_paths(repo_root, site="karolina", env={})

    env_script = render_korali_env_script(paths)

    assert "MESOUQ_SITE=karolina" in env_script
    assert "MESOUQ_SITE_RUNTIME_ROOT" in env_script
    assert "MESOUQ_PROVENANCE_ROOT" in env_script
    assert "MESOUQ_KAROLINA_ROOT" in env_script
    assert str(repo_root / "_karolina" / "korali" / "install") in env_script
    assert "MESOUQ_VEGA_ROOT" not in env_script


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


def test_load_mirheo_source_lock_uses_karolina_default_when_lock_missing(tmp_path):
    repo_root = _make_repo(tmp_path)

    lock = load_mirheo_source_lock(repo_root, site="karolina")

    assert lock["source_path"] == "/home/it4i-bbenvegnen/software/Mirheo"


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


def test_render_mirheo_env_script_exports_karolina_root(tmp_path):
    repo_root = _make_repo(tmp_path)
    paths = get_runtime_paths(repo_root, site="karolina", env={})
    source_root = tmp_path / "Mirheo"
    source_root.mkdir()

    env_script = render_mirheo_env_script(paths, source_root=source_root)

    assert "MESOUQ_SITE=karolina" in env_script
    assert "MESOUQ_PROVENANCE_ROOT" in env_script
    assert "MESOUQ_KAROLINA_ROOT" in env_script
    assert "MESOUQ_VEGA_ROOT" not in env_script


def test_render_mirheo_env_script_omits_snapshot_when_not_provided(tmp_path):
    repo_root = _make_repo(tmp_path)
    paths = get_vega_paths(repo_root)
    source_root = tmp_path / "Mirheo"
    source_root.mkdir()

    env_script = render_mirheo_env_script(paths, source_root=source_root)

    assert "MIRHEO_SOURCE_SNAPSHOT" not in env_script

def test_render_gv_cgal_tools_env_script_exports_karolina_library_paths(tmp_path):
    repo_root = _make_repo(tmp_path)
    paths = get_runtime_paths(repo_root, site="karolina", env={})

    env_script = render_gv_cgal_tools_env_script(paths)

    assert "MESOUQ_SITE=karolina" in env_script
    assert "MESOUQ_PROVENANCE_ROOT" in env_script
    assert "MESOUQ_KAROLINA_ROOT" in env_script
    assert str(paths.scale_space_binary) in env_script
    assert str(paths.gv_cgal_tools_bin_dir) in env_script
    assert "MPFR/4.2.0-GCCcore-12.2.0/lib" in env_script
    assert "GMP/6.2.1-GCCcore-12.2.0/lib" in env_script
    assert "LD_LIBRARY_PATH" in env_script


def test_render_gv_cgal_tools_env_script_exports_vega_root_without_karolina_libs(tmp_path):
    repo_root = _make_repo(tmp_path)
    paths = get_vega_paths(repo_root)

    env_script = render_gv_cgal_tools_env_script(paths)

    assert "MESOUQ_SITE=vega" in env_script
    assert "MESOUQ_VEGA_ROOT" in env_script
    assert "MESOUQ_KAROLINA_ROOT" not in env_script
    assert "MPFR/4.2.0-GCCcore-12.2.0/lib" not in env_script


def test_discover_hdf5_runtime_roots_deduplicates_and_normalizes_lib_dirs(tmp_path):
    hdf5_root = tmp_path / "hdf5"
    hdf5_lib = hdf5_root / "lib"
    hdf5_lib.mkdir(parents=True)

    roots = discover_hdf5_runtime_roots(
        {"MESOUQ_HDF5_ROOT": str(hdf5_root), "EBROOTHDF5": str(hdf5_lib), "HDF5_DIR": ""}
    )

    assert roots == (hdf5_root.resolve(),)

def test_render_tinytex_env_script_exports_karolina_root(tmp_path):
    repo_root = _make_repo(tmp_path)
    paths = get_runtime_paths(repo_root, site="karolina", env={})

    env_script = render_tinytex_env_script(paths)

    assert "MESOUQ_SITE=karolina" in env_script
    assert "MESOUQ_PROVENANCE_ROOT" in env_script
    assert "MESOUQ_KAROLINA_ROOT" in env_script
    assert "MESOUQ_VEGA_ROOT" not in env_script


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
