from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterator

import pytest
import yaml

from meso_uq.structures.gv import material_parameters as material_parameters_module
from meso_uq.structures.gv.material_parameters import (
    apply_material_overrides_to_glob,
    apply_material_overrides_to_files,
    apply_material_overrides_to_yaml,
    main,
    validate_material_parameter_overrides,
)


_VALID_MATERIAL_PARAMETER_OVERRIDES = {
    "ka": 1.1,
    "kb": 0.9,
    "mu": 0.7,
    "b1": 0.2,
    "b2": 0.3,
    "a3": 0.4,
    "a4": 0.5,
    "mu_l": 0.6,
    "c": 0.8,
}


def test_validate_material_overrides_rejects_missing_values() -> None:
    incomplete = dict(_VALID_MATERIAL_PARAMETER_OVERRIDES)
    incomplete.pop("c")
    with pytest.raises(ValueError, match="Missing required GV material parameters"):
        validate_material_parameter_overrides(incomplete)


def test_validate_material_overrides_rejects_non_finite_and_unknown_fields() -> None:
    with pytest.raises(ValueError, match="must be numeric"):
        validate_material_parameter_overrides({**_VALID_MATERIAL_PARAMETER_OVERRIDES, "ka": object()})

    with pytest.raises(ValueError, match="must be finite and > 0"):
        validate_material_parameter_overrides({**_VALID_MATERIAL_PARAMETER_OVERRIDES, "ka": -1})

    with pytest.raises(ValueError, match="must be finite and > 0"):
        validate_material_parameter_overrides({**_VALID_MATERIAL_PARAMETER_OVERRIDES, "kb": float("inf")})

    with pytest.raises(ValueError, match="Unexpected material parameter"):
        validate_material_parameter_overrides({**_VALID_MATERIAL_PARAMETER_OVERRIDES, "not_a_param": 1.0})

    duplicate_alias = dict(_VALID_MATERIAL_PARAMETER_OVERRIDES)
    duplicate_alias["muL"] = duplicate_alias["mu_l"]
    with pytest.raises(ValueError, match="duplicated"):
        validate_material_parameter_overrides(duplicate_alias)


def test_validate_material_overrides_accepts_muL_alias() -> None:
    aliases = dict(_VALID_MATERIAL_PARAMETER_OVERRIDES)
    aliases["muL"] = aliases.pop("mu_l")
    normalized = validate_material_parameter_overrides(aliases)
    assert normalized["mu_l"] == _VALID_MATERIAL_PARAMETER_OVERRIDES["mu_l"]


def test_validate_material_overrides_defensive_extra_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    class WeirdNames:
        def __contains__(self, item: object) -> bool:
            return item in _VALID_MATERIAL_PARAMETER_OVERRIDES

        def __iter__(self) -> Iterator[str]:
            return iter(tuple(name for name in _VALID_MATERIAL_PARAMETER_OVERRIDES if name != "c"))

    monkeypatch.setattr(material_parameters_module, "GV_MATERIAL_PARAMETER_NAMES", WeirdNames())

    with pytest.raises(ValueError, match="Unexpected GV material parameters: c"):
        validate_material_parameter_overrides(_VALID_MATERIAL_PARAMETER_OVERRIDES)


def test_apply_material_overrides_to_yaml_preserves_existing_muL_alias(tmp_path: Path) -> None:
    payload_path = tmp_path / "parameters.prms00001.yaml"
    payload_path.write_text(
        yaml.safe_dump(
            {
                "ka": 2.0,
                "kb": 4.0,
                "mu": 6.0,
                "b1": 0.1,
                "b2": 0.2,
                "a3": 0.3,
                "a4": 0.4,
                "muL": 0.5,
                "c": 0.6,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    apply_material_overrides_to_yaml(payload_path, _VALID_MATERIAL_PARAMETER_OVERRIDES)

    updated = yaml.safe_load(payload_path.read_text(encoding="utf-8"))
    assert updated["ka"] == 1.1
    assert updated["muL"] == 0.6
    assert "mu_l" not in updated


def test_apply_material_overrides_to_glob_updates_parameters_prms_yaml_files_only(tmp_path: Path) -> None:
    first = tmp_path / "parameters.prms00001.yaml"
    first.write_text(
        yaml.safe_dump(
            {
                "ka": 2.0,
                "kb": 4.0,
                "mu": 6.0,
                "b1": 0.1,
                "b2": 0.2,
                "a3": 0.3,
                "a4": 0.4,
                "muL": 0.5,
                "c": 0.6,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    second = tmp_path / "parameters.prms00001eq.yaml"
    second.write_text(
        yaml.safe_dump(
            {
                "ka": 3.0,
                "kb": 5.0,
                "mu": 7.0,
                "b1": 0.2,
                "b2": 0.3,
                "a3": 0.4,
                "a4": 0.5,
                "mu_l": 0.9,
                "c": 1.0,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "ignore.yaml").write_text(yaml.safe_dump({"ka": 99}), encoding="utf-8")

    updated = apply_material_overrides_to_glob(tmp_path, _VALID_MATERIAL_PARAMETER_OVERRIDES)

    assert {path.name for path in updated} == {"parameters.prms00001.yaml", "parameters.prms00001eq.yaml"}
    assert yaml.safe_load(first.read_text(encoding="utf-8"))["muL"] == 0.6
    assert "mu_l" not in yaml.safe_load(first.read_text(encoding="utf-8"))
    assert yaml.safe_load(second.read_text(encoding="utf-8"))["mu_l"] == 0.6
    assert "muL" not in yaml.safe_load(second.read_text(encoding="utf-8"))
    assert yaml.safe_load((tmp_path / "ignore.yaml").read_text(encoding="utf-8"))["ka"] == 99


def test_material_override_file_helpers_reject_bad_paths_and_payloads(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError, match="File not found"):
        apply_material_overrides_to_yaml(missing, _VALID_MATERIAL_PARAMETER_OVERRIDES)

    non_mapping = tmp_path / "parameters.prms00001.yaml"
    non_mapping.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected YAML mapping"):
        apply_material_overrides_to_yaml(non_mapping, _VALID_MATERIAL_PARAMETER_OVERRIDES)

    with pytest.raises(NotADirectoryError, match="Expected a directory"):
        apply_material_overrides_to_glob(tmp_path / "not-a-dir", _VALID_MATERIAL_PARAMETER_OVERRIDES)

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No parameter files matched"):
        apply_material_overrides_to_glob(empty_dir, _VALID_MATERIAL_PARAMETER_OVERRIDES)

    mapping = tmp_path / "parameters.prms00002.yaml"
    mapping.write_text(yaml.safe_dump({"ka": 2.0}), encoding="utf-8")
    updated = apply_material_overrides_to_files([mapping], _VALID_MATERIAL_PARAMETER_OVERRIDES)
    assert updated == [mapping]


def test_material_parameter_cli_applies_json_overrides(tmp_path: Path) -> None:
    payload_path = tmp_path / "parameters.prms00001.yaml"
    payload_path.write_text(
        yaml.safe_dump(
            {
                "ka": 2.0,
                "kb": 4.0,
                "mu": 6.0,
                "b1": 0.1,
                "b2": 0.2,
                "a3": 0.3,
                "a4": 0.4,
                "muL": 0.5,
                "c": 0.6,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    rc = main(
        [
            str(payload_path),
            "--overrides-json",
            json.dumps(_VALID_MATERIAL_PARAMETER_OVERRIDES),
        ]
    )

    updated = yaml.safe_load(payload_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert updated["ka"] == 1.1
    assert updated["muL"] == 0.6


def test_material_parameter_cli_rejects_non_mapping_json(tmp_path: Path) -> None:
    payload_path = tmp_path / "parameters.prms00001.yaml"
    payload_path.write_text(yaml.safe_dump({"ka": 2.0}), encoding="utf-8")

    with pytest.raises(ValueError, match="must decode to a mapping"):
        main([str(payload_path), "--overrides-json", json.dumps([1, 2, 3])])


@pytest.mark.parametrize("experiment", ["stretching", "buckling", "torsion", "eigenmodes"])
def test_non_shear_gv_run_scripts_apply_material_override_hook(experiment: str) -> None:
    run_script = Path("gv") / experiment / "src" / "run.sh"
    text = run_script.read_text(encoding="utf-8")

    assert "MESOUQ_GV_MATERIAL_OVERRIDES_JSON" in text
    assert "python3 -m meso_uq.structures.gv.material_parameters" in text
    assert "parameters.prms${simnum}.yaml" in text
    assert text.index("python3 parameters.py") < text.index("MESOUQ_GV_MATERIAL_OVERRIDES_JSON")
    assert text.index("MESOUQ_GV_MATERIAL_OVERRIDES_JSON") < text.index("mpirun")
