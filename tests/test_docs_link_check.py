import importlib.util
import sys
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_find_broken_links_reports_missing_local_markdown_targets(tmp_path):
    repo_root = tmp_path
    docs_dir = repo_root / "docs"
    docs_dir.mkdir()
    readme = repo_root / "README.md"
    page = docs_dir / "page.md"
    readme.write_text("[ok](docs/page.md)\n[missing](docs/missing.md)\n", encoding="utf-8")
    page.write_text("[back](../README.md)\n[anchor](#section)\n", encoding="utf-8")

    module = _load_module(
        Path(__file__).resolve().parents[1] / "scripts" / "ci" / "check_docs_links.py",
        "check_docs_links_test",
    )

    broken = module.find_broken_links([readme, page], repo_root)
    assert len(broken) == 1
    assert broken[0].document == readme
    assert broken[0].target == "docs/missing.md"
