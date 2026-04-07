from pathlib import Path


def test_required_release_assets_exist():
    repo_root = Path(__file__).resolve().parents[1]

    expected_paths = [
        repo_root / "compression" / "evalkit" / "data" / "data_1.csv",
        repo_root / "compression" / "evalkit" / "data" / "data_2.csv",
        repo_root / "compression" / "evalkit" / "data" / "data_3.csv",
        repo_root / "indentation" / "evalkit" / "data" / "data_morris_3.19.csv",
        repo_root / "indentation" / "evalkit" / "data" / "data_morris_3.40.csv",
        repo_root / "indentation" / "evalkit" / "data" / "data_morris_5.76.csv",
        repo_root / "compression" / "surrogate" / "diameters" / "2.1um" / "trained" / "microbubble_force_BEST.pkl",
        repo_root / "compression" / "surrogate" / "diameters" / "2.9um" / "trained" / "microbubble_force_BEST.pkl",
        repo_root / "compression" / "surrogate" / "diameters" / "3.0um" / "trained" / "microbubble_force_BEST.pkl",
        repo_root / "indentation" / "surrogate" / "diameters" / "3.2um" / "trained" / "microbubble_displacement_BEST.pkl",
        repo_root / "indentation" / "surrogate" / "diameters" / "3.4um" / "trained" / "microbubble_displacement_BEST.pkl",
        repo_root / "indentation" / "surrogate" / "diameters" / "5.8um" / "trained" / "microbubble_displacement_BEST.pkl",
    ]

    missing = [path for path in expected_paths if not path.exists()]
    assert not missing, f"Missing release assets: {missing}"
