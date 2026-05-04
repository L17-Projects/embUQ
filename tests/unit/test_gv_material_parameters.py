from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from meso_uq.structures.gv.material_parameters import (
    apply_material_overrides_to_glob,
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
    with pytest.raises(ValueError, match="must be finite and > 0"):
        validate_material_parameter_overrides({**_VALID_MATERIAL_PARAMETER_OVERRIDES, "ka": -1})

    with pytest.raises(ValueError, match="must be finite and > 0"):
        validate_material_parameter_overrides({**_VALID_MATERIAL_PARAMETER_OVERRIDES, "kb": float("inf")})

    with pytest.raises(ValueError, match="Unexpected material parameter"):
        validate_material_parameter_overrides({**_VALID_MATERIAL_PARAMETER_OVERRIDES, "not_a_param": 1.0})


def test_validate_material_overrides_accepts_muL_alias() -> None:
    aliases = dict(_VALID_MATERIAL_PARAMETER_OVERRIDES)
    aliases["muL"] = aliases.pop("mu_l")
    normalized = validate_material_parameter_overrides(aliases)
    assert normalized["mu_l"] == _VALID_MATERIAL_PARAMETER_OVERRIDES["mu_l"]


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


@pytest.mark.parametrize("experiment", ["stretching", "buckling", "torsion", "eigenmodes"])
def test_non_shear_gv_run_scripts_apply_material_override_hook(experiment: str) -> None:
    run_script = Path("gv") / experiment / "src" / "run.sh"
    text = run_script.read_text(encoding="utf-8")

    assert "MESOUQ_GV_MATERIAL_OVERRIDES_JSON" in text
    assert "python3 -m meso_uq.structures.gv.material_parameters" in text
    assert "parameters.prms${simnum}.yaml" in text
    assert text.index("python3 parameters.py") < text.index("MESOUQ_GV_MATERIAL_OVERRIDES_JSON")
    assert text.index("MESOUQ_GV_MATERIAL_OVERRIDES_JSON") < text.index("mpirun")
