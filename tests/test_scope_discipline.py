from pathlib import Path


def test_clean_mainline_feature_contains_no_port_comsol_or_activation_solver():
    root = Path(__file__).resolve().parents[1]
    package = root / "parastell"
    relative_paths = {
        path.relative_to(root).as_posix().lower()
        for path in package.rglob("*")
        if path.is_file()
    }
    prohibited = {
        path
        for path in relative_paths
        if "native_port" in path
        or "/ports/" in f"/{path}/"
        or "comsol" in path
        or path.startswith("parastell/activation/")
    }
    assert prohibited == set()


def test_reference_policy_ledgers_are_present_and_explicitly_read_only():
    root = Path(__file__).resolve().parents[1]
    ledger = (root / "docs" / "REFERENCE_WORKFLOW_LEDGER.md").read_text(
        encoding="utf-8"
    )
    migration = (
        root / "docs" / "MAGNET_RADIATION_MAINLINE_MIGRATION_LEDGER.md"
    ).read_text(encoding="utf-8")
    combined = f"{ledger}\n{migration}".lower()
    assert "read-only" in combined
    assert "no source" in combined
    assert "live repository import" in combined
