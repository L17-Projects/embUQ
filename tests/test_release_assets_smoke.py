from pathlib import Path
import pickletools

from meso_uq.surrogate.compat import list_serialized_surrogate_aliases


def test_required_release_assets_exist():
    repo_root = Path(__file__).resolve().parents[1]

    expected_paths = [
        repo_root / "emb" / "compression" / "evalkit" / "data" / "data_1.csv",
        repo_root / "emb" / "compression" / "evalkit" / "data" / "data_2.csv",
        repo_root / "emb" / "compression" / "evalkit" / "data" / "data_3.csv",
        repo_root / "emb" / "indentation" / "evalkit" / "data" / "data_morris_3.19.csv",
        repo_root / "emb" / "indentation" / "evalkit" / "data" / "data_morris_3.40.csv",
        repo_root / "emb" / "indentation" / "evalkit" / "data" / "data_morris_5.76.csv",
        repo_root / "emb" / "compression" / "surrogate" / "diameters" / "2.1um" / "trained" / "microbubble_force_BEST.pkl",
        repo_root / "emb" / "compression" / "surrogate" / "diameters" / "2.9um" / "trained" / "microbubble_force_BEST.pkl",
        repo_root / "emb" / "compression" / "surrogate" / "diameters" / "3.0um" / "trained" / "microbubble_force_BEST.pkl",
        repo_root / "emb" / "indentation" / "surrogate" / "diameters" / "3.2um" / "trained" / "microbubble_displacement_BEST.pkl",
        repo_root / "emb" / "indentation" / "surrogate" / "diameters" / "3.4um" / "trained" / "microbubble_displacement_BEST.pkl",
        repo_root / "emb" / "indentation" / "surrogate" / "diameters" / "5.8um" / "trained" / "microbubble_displacement_BEST.pkl",
    ]

    missing = [path for path in expected_paths if not path.exists()]
    assert not missing, f"Missing release assets: {missing}"


def test_committed_dnn_surrogate_pickles_use_documented_class_paths():
    repo_root = Path(__file__).resolve().parents[1]
    aliases = list_serialized_surrogate_aliases()
    supported_modules = {alias.legacy_module for alias in aliases}
    supported_modules.update(alias.replacement_module for alias in aliases)

    pickle_paths = [
        repo_root / "emb" / "compression" / "surrogate" / "diameters" / "2.1um" / "trained" / "microbubble_force_BEST.pkl",
        repo_root / "emb" / "compression" / "surrogate" / "diameters" / "2.9um" / "trained" / "microbubble_force_BEST.pkl",
        repo_root / "emb" / "compression" / "surrogate" / "diameters" / "3.0um" / "trained" / "microbubble_force_BEST.pkl",
        repo_root / "emb" / "indentation" / "surrogate" / "diameters" / "3.2um" / "trained" / "microbubble_displacement_BEST.pkl",
        repo_root / "emb" / "indentation" / "surrogate" / "diameters" / "3.4um" / "trained" / "microbubble_displacement_BEST.pkl",
        repo_root / "emb" / "indentation" / "surrogate" / "diameters" / "5.8um" / "trained" / "microbubble_displacement_BEST.pkl",
    ]

    for path in pickle_paths:
        pickle_strings = {
            arg
            for opcode, arg, _position in pickletools.genops(path.read_bytes())
            if isinstance(arg, str) and opcode.name in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"}
        }
        modules = {value for value in pickle_strings if value.endswith(".model")}
        assert modules <= supported_modules
        assert "MLP" in pickle_strings
