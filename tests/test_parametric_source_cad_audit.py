import hashlib
import json

import pytest

import scripts.audit_parametric_source_cad as audit_module
from scripts.audit_parametric_source_cad import _intersection_pass
from scripts.audit_parametric_source_cad import _validate_source


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_intersection_gate_is_numeric_and_fail_closed():
    assert _intersection_pass({"status": "MEASURED", "volume_cm3": 0.0})
    assert not _intersection_pass({"status": "MEASURED", "volume_cm3": 1.0e-3})
    assert not _intersection_pass({"status": "ERROR", "volume_cm3": 0.0})
    assert not _intersection_pass({"status": "MEASURED", "volume_cm3": None})


def test_positive_exact_distance_skips_boolean_common(monkeypatch):
    monkeypatch.setattr(
        audit_module,
        "_distance",
        lambda *args, **kwargs: {
            "status": "MEASURED",
            "minimum_distance_cm": 2.0,
        },
    )
    monkeypatch.setattr(
        audit_module,
        "_intersection",
        lambda *args, **kwargs: pytest.fail("common operation was requested"),
    )
    result, separation = audit_module._distance_first_intersection(
        object(), object()
    )
    assert result["status"] == "EXACT_BREP_DISTANCE_DISJOINT"
    assert result["volume_cm3"] == 0.0
    assert separation["minimum_distance_cm"] == 2.0


def test_magnet_component_uses_distance_only_for_clearance(monkeypatch):
    monkeypatch.setattr(audit_module, "_MAGNETS", [object()])
    monkeypatch.setattr(
        audit_module,
        "_COMPONENTS",
        {"chamber": object(), "vacuum_gap": object()},
    )
    calls = []
    monkeypatch.setattr(
        audit_module,
        "_intersection",
        lambda *args, **kwargs: {
            "status": "EXACT_AABB_DISJOINT",
            "volume_cm3": 0.0,
        },
    )
    monkeypatch.setattr(
        audit_module,
        "_distance_first_intersection",
        lambda *args, **kwargs: (
            calls.append(kwargs)
            or {
                "status": "EXACT_BREP_DISTANCE_DISJOINT",
                "volume_cm3": 0.0,
            },
            {"status": "MEASURED", "minimum_distance_cm": 1.0},
        ),
    )
    chamber = audit_module._magnet_component((0, "chamber"))
    gap = audit_module._magnet_component((0, "vacuum_gap"))
    assert chamber["separation_distance"]["status"].startswith("NOT_REQUIRED")
    assert gap["minimum_clearance"]["minimum_distance_cm"] == 1.0
    assert calls == [{"witnesses_required": True}]


def test_source_manifest_binds_artifacts_components_and_all_magnets(tmp_path):
    artifact = tmp_path / "chamber.step"
    artifact.write_bytes(b"cad")
    manifest = {
        "schema": "parastell.parametric_source_cad/v1.0.0",
        "status": "SOURCE_CAD_BUILT_GEOMETRY_GATES_PENDING",
        "transport_eligible": False,
        "actual_invessel_component_order": ["chamber"],
        "artifacts": {
            "chamber.step": {
                "bytes": artifact.stat().st_size,
                "sha256": _sha(artifact),
            }
        },
        "magnet_inventory": {
            "magnets": [
                {"magnet_id": f"magnet-{index:04d}"} for index in range(18)
            ]
        },
    }
    manifest_path = tmp_path / "PARAMETRIC_GEOMETRY_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = _validate_source(tmp_path, _sha(manifest_path))
    assert result["components"] == ["chamber"]
    artifact.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="integrity"):
        _validate_source(tmp_path, _sha(manifest_path))


def test_one_worker_runs_in_process_without_constructing_pool(
    tmp_path, monkeypatch
):
    initialized = []
    monkeypatch.setattr(
        audit_module,
        "_configure",
        lambda source, names: initialized.append((source, names)),
    )
    monkeypatch.setattr(
        audit_module,
        "ProcessPoolExecutor",
        lambda *args, **kwargs: pytest.fail("process pool was constructed"),
    )
    rows = {"stage": []}
    audit_module._run_tasks(
        work=(("stage", lambda value: {"value": value}, [1, 2]),),
        rows=rows,
        progress=tmp_path / "progress.jsonl",
        workers=1,
        initializer=("source", ["component"]),
    )
    assert initialized == [("source", ["component"])]
    assert rows["stage"] == [{"value": 1}, {"value": 2}]
