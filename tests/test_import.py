def test_package_exposes_version() -> None:
    import statefuzz

    assert statefuzz.__version__ == "0.1.0"
