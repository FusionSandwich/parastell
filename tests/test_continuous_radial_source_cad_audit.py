from scripts import audit_continuous_radial_source_cad as audit_module


def test_source_audit_releases_solids_between_checks(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "audit"
    state = {"active": 0, "maximum": 0}

    class Solid:
        def __init__(self, name):
            self.name = name
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])

        def __del__(self):
            state["active"] -= 1

    monkeypatch.setattr(
        audit_module,
        "_load_source",
        lambda *args, **kwargs: {"manifest": {}, "artifacts": {"x": "y"}},
    )
    monkeypatch.setattr(
        audit_module, "_one_solid", lambda path: Solid(path.stem)
    )
    monkeypatch.setattr(
        audit_module,
        "_shape_row",
        lambda solid, identity: {
            "identity": identity,
            "valid": True,
            "closed": True,
            "numeric_values_finite": True,
            "volume_cm3": 1.0,
        },
    )
    monkeypatch.setattr(
        audit_module,
        "_intersection",
        lambda *args, **kwargs: {"status": "PASS", "volume_cm3": 0.0},
    )
    monkeypatch.setattr(
        audit_module,
        "radial_separation_proof",
        lambda manifest: [{"pass": True} for _ in range(28)],
    )

    report = audit_module.audit(
        source, output, expected_manifest_sha256="a" * 64
    )

    assert report["status"] == "PASS"
    assert state == {"active": 0, "maximum": 2}
