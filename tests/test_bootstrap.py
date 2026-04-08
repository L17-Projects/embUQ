from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


def test_bootstrap_package_imports():
    import meso_uq

    assert hasattr(meso_uq, "__version__")


def test_workflow_dependency_contract_includes_trimesh():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    dependencies = payload["project"]["dependencies"]

    assert any(dep.startswith("trimesh") for dep in dependencies)
