import hashlib
import json

import pytest

import scripts.audit_parametric_source_cad as audit_module
from scripts.audit_parametric_source_cad import _intersection_pass
from scripts.audit_parametric_source_cad import _validate_source


class _Box:
    def __init__(self, minimum, maximum):
        self.xmin, self.ymin, self.zmin = minimum
        self.xmax, self.ymax, self.zmax = maximum


class _Shape:
    def __init__(self, minimum=(0.0, 0.0, 0.0), maximum=(1.0, 1.0, 1.0)):
        self.box = _Box(minimum, maximum)

    def BoundingBox(self):
        return self.box


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_intersection_gate_is_numeric_and_fail_closed():
    assert _intersection_pass({"status": "MEASURED", "volume_cm3": 0.0})
    assert not _intersection_pass({"status": "MEASURED", "volume_cm3": 1.0e-3})
    assert not _intersection_pass({"status": "ERROR", "volume_cm3": 0.0})
    assert not _intersection_pass({"status": "MEASURED", "volume_cm3": None})
    assert not _intersection_pass({"status": "MEASURED", "volume_cm3": -1.0})


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
        _Shape(), _Shape()
    )
    assert result["status"] == "EXACT_BREP_DISTANCE_DISJOINT"
    assert result["volume_cm3"] == 0.0
    assert separation["minimum_distance_cm"] == 2.0


def test_strict_aabb_gap_skips_distance_and_common(monkeypatch):
    monkeypatch.setattr(
        audit_module,
        "_distance",
        lambda *args, **kwargs: pytest.fail("distance was requested"),
    )
    monkeypatch.setattr(
        audit_module,
        "_intersection",
        lambda *args, **kwargs: pytest.fail("common operation was requested"),
    )
    result, separation = audit_module._distance_first_intersection(
        _Shape(), _Shape(minimum=(2.0, 0.0, 0.0), maximum=(3.0, 1.0, 1.0))
    )
    assert result["status"] == "CONSERVATIVE_AABB_DISJOINT"
    assert separation["minimum_distance_lower_bound_cm"] == 1.0


def test_uncertain_distance_falls_back_to_boolean_common(monkeypatch):
    monkeypatch.setattr(
        audit_module,
        "_distance",
        lambda *args, **kwargs: {"status": "ERROR", "reason": "uncertain"},
    )
    monkeypatch.setattr(
        audit_module,
        "_intersection",
        lambda *args, **kwargs: {
            "status": "MEASURED_EMPTY_INTERSECTION",
            "volume_cm3": 0.0,
        },
    )
    result, separation = audit_module._distance_first_intersection(
        _Shape(), _Shape()
    )
    assert result["status"] == "MEASURED_EMPTY_INTERSECTION"
    assert separation["status"] == "ERROR"


def test_nonfinite_aabb_fails_closed(monkeypatch):
    monkeypatch.setattr(
        audit_module,
        "_distance",
        lambda *args, **kwargs: pytest.fail("distance was requested"),
    )
    result, separation = audit_module._distance_first_intersection(
        _Shape(maximum=(float("nan"), 1.0, 1.0)), _Shape()
    )
    assert result["status"] == "ERROR"
    assert separation["status"] == "ERROR"


def test_inverted_aabb_fails_closed(monkeypatch):
    monkeypatch.setattr(
        audit_module,
        "_distance",
        lambda *args, **kwargs: pytest.fail("distance was requested"),
    )
    result, separation = audit_module._distance_first_intersection(
        _Shape(minimum=(2.0, 0.0, 0.0), maximum=(1.0, 1.0, 1.0)), _Shape()
    )
    assert result["status"] == "ERROR"
    assert separation["status"] == "ERROR"


def test_magnet_component_uses_distance_first_for_every_pair(monkeypatch):
    monkeypatch.setattr(audit_module, "_MAGNETS", [object()])
    monkeypatch.setattr(
        audit_module,
        "_COMPONENTS",
        {"chamber": object(), "vacuum_gap": object()},
    )
    calls = []
    monkeypatch.setattr(
        audit_module,
        "_distance_first_intersection",
        lambda *args, **kwargs: (
            calls.append(kwargs)
            or {
                "status": "EXACT_BREP_DISTANCE_DISJOINT",
                "volume_cm3": 0.0,
            },
            {
                "status": "MEASURED",
                "minimum_distance_cm": 1.0,
                "witnesses_required": kwargs["witnesses_required"],
                "solution_count": 1,
                "witnesses": [
                    {
                        "distance_reconstruction_pass": True,
                        "point_membership_pass": True,
                    }
                ],
            },
        ),
    )
    chamber = audit_module._magnet_component((0, "chamber"))
    gap = audit_module._magnet_component((0, "vacuum_gap"))
    assert chamber["separation_distance"]["status"] == "MEASURED"
    assert gap["minimum_clearance"]["minimum_distance_cm"] == 1.0
    assert gap["clearance_witness_pass"] is True
    assert calls == [
        {"witnesses_required": False},
        {"witnesses_required": True},
    ]


def test_magnet_pair_uses_distance_first_without_witnesses(monkeypatch):
    monkeypatch.setattr(audit_module, "_MAGNETS", [object(), object()])
    calls = []
    monkeypatch.setattr(
        audit_module,
        "_distance_first_intersection",
        lambda *args, **kwargs: (
            calls.append(kwargs)
            or {"status": "EXACT_BREP_DISTANCE_DISJOINT", "volume_cm3": 0.0},
            {"status": "MEASURED", "minimum_distance_cm": 1.0},
        ),
    )
    row = audit_module._magnet_pair((0, 1))
    assert row["pass"] is True
    assert calls == [{"witnesses_required": False}]


def test_clearance_witness_gate_rejects_bad_witness():
    valid = {
        "status": "MEASURED",
        "minimum_distance_cm": 1.0,
        "witnesses_required": True,
        "solution_count": 1,
        "witnesses": [
            {
                "distance_reconstruction_pass": True,
                "point_membership_pass": True,
            }
        ],
    }
    assert audit_module._clearance_witness_pass(valid)
    valid["witnesses"][0]["point_membership_pass"] = False
    assert not audit_module._clearance_witness_pass(valid)


def test_pair_tasks_are_exhaustive_and_canonical():
    component, magnet_component, magnet = audit_module._pair_tasks(
        [f"component-{index}" for index in range(8)]
    )
    assert len(component) == 28
    assert len(magnet_component) == 144
    assert len(magnet) == 153
    assert component[0] == ("component-0", "component-1")
    assert component[-1] == ("component-6", "component-7")
    assert magnet_component[0] == (0, "component-0")
    assert magnet_component[-1] == (17, "component-7")
    assert magnet[0] == (0, 1)
    assert magnet[-1] == (16, 17)


def test_source_mutation_blocks_audit_pass():
    summary = {
        "invalid_component_count": 0,
        "invalid_magnet_count": 0,
        "component_overlap_failures": 0,
        "magnet_component_overlap_failures": 0,
        "magnet_pair_overlap_failures": 0,
        "clearance_measurement_count": 18,
        "clearance_witness_pass_count": 18,
    }
    assert audit_module._audit_pass(summary, source_immutable=True)
    assert not audit_module._audit_pass(summary, source_immutable=False)


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


@pytest.mark.parametrize("workers", [0, 17])
def test_audit_rejects_worker_count_outside_bounded_contract(
    tmp_path, workers
):
    with pytest.raises(ValueError, match="between one and sixteen"):
        audit_module.audit(
            tmp_path / "source",
            tmp_path / "output",
            expected_manifest_sha256="0" * 64,
            workers=workers,
        )
