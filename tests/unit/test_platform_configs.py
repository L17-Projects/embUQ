from __future__ import annotations

from pathlib import Path

from meso_uq.configs import load_structured_document, validate_config_file


REPO_ROOT = Path(__file__).resolve().parents[2]
PLATFORM_CONFIG_ROOT = REPO_ROOT / "configs" / "platforms"


def test_platform_configs_validate() -> None:
    expected = {
        PLATFORM_CONFIG_ROOT / "generic_slurm.example.yaml",
        PLATFORM_CONFIG_ROOT / "karolina.example.yaml",
        PLATFORM_CONFIG_ROOT / "vega.example.yaml",
        PLATFORM_CONFIG_ROOT / "workstation.example.yaml",
    }

    assert expected.issubset(set(PLATFORM_CONFIG_ROOT.glob("*.yaml")))

    for path in sorted(expected):
        assert validate_config_file(path) == [], path.as_posix()


def test_karolina_platform_policy_forbids_private_home_root() -> None:
    document = load_structured_document(PLATFORM_CONFIG_ROOT / "karolina.example.yaml")

    assert document["path_policy"]["forbidden_literals"] == ["/ceph/hpc/home/eubrieucb"]


def test_platform_configs_use_placeholder_based_paths() -> None:
    for path in sorted(PLATFORM_CONFIG_ROOT.glob("*.yaml")):
        document = load_structured_document(path)
        for value in document["spec"]["paths"].values():
            assert not str(value).startswith("/")
