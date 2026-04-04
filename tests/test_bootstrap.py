def test_bootstrap_package_imports():
    import meso_uq

    assert hasattr(meso_uq, "__version__")
