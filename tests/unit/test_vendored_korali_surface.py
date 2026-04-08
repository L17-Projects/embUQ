from pathlib import Path


def test_vendored_korali_surface_supports_repo_local_bootstrap() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    required_paths = [
        repo_root / "extern/korali/source/engine.cpp",
        repo_root / "extern/korali/source/modules/meson.build",
        repo_root / "extern/korali/python/korali/cxx/meson.build",
        repo_root / "extern/korali/subprojects/eigen.wrap",
        repo_root / "extern/korali/subprojects/gsl.wrap",
        repo_root / "extern/korali/subprojects/pybind11.wrap",
        repo_root / "extern/korali/tools/build/__init__.py",
        repo_root / "extern/korali/tools/build/codeBuilders/builders.py",
    ]

    missing = [str(path.relative_to(repo_root)) for path in required_paths if not path.exists()]

    assert not missing, f"Vendored Korali bootstrap surface is incomplete: {missing}"
