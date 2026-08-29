import hashlib
import json

import pytest
import numpy as np

from scripts import export_parametric_direct90_dagmc as exporter


def test_committed_material_counts_match_export_contract():
    path = (
        exporter.REPOSITORY_ROOT
        / "configs"
        / "wistell_d_parametric_direct90_material_counts.json"
    )
    observed = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        material: exporter.MATERIAL_TAGS.count(material)
        for material in set(exporter.MATERIAL_TAGS)
    }
    assert observed == expected


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _packet(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    artifacts = {}
    filenames = [
        *(f"{name}.step" for name in exporter.RADIAL_COMPONENTS),
        "magnet_set.step",
    ]
    for index, filename in enumerate(filenames):
        path = source / filename
        path.write_bytes(f"artifact-{index}".encode())
        artifacts[filename] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    magnets = [
        {
            "magnet_id": magnet_id,
            "material": "homogenized_magnet",
            "brep_valid": True,
            "solid_count": 1,
            "casing_winding_split": False,
            "cross_section_cm": {"thickness": 30.0, "width": 30.0},
            "volume_cm3": float(index + 1),
            "centroid_cm": [float(index), 0.0, 0.0],
        }
        for index, magnet_id in enumerate(exporter.MAGNET_IDS)
    ]
    manifest = {
        "schema": exporter.SOURCE_MANIFEST_SCHEMA,
        "status": "SOURCE_CAD_BUILT_GEOMETRY_GATES_PENDING",
        "transport_eligible": False,
        "actual_invessel_component_order": list(exporter.RADIAL_COMPONENTS),
        "plan": {
            "device": "WISTELL-D",
            "extent_degrees": 90.0,
            "n_field_periods": 4,
            "field_period_degrees": 90.0,
            "grid_shape": [80, 90],
            "magnet_representation": "swept_filaments",
            "radial_build_mode": "isolated_cumulative_shells",
            "component_order": [*exporter.RADIAL_COMPONENTS, "magnets"],
            "safety": {
                "ports": False,
                "post_build_transforms": False,
                "imported_physical_solids": False,
            },
        },
        "live_vmec_metadata": {
            "pass": True,
            "n_field_periods": 4,
            "stellarator_symmetric": True,
            "lasym_vmec_logical": False,
        },
        "radial_build_mode": "isolated_cumulative_shells",
        "magnet_inventory": {
            "solid_count": 18,
            "casing_winding_split": False,
            "one_homogenized_solid_per_coil": True,
            "magnets": magnets,
        },
        "artifacts": artifacts,
    }
    manifest_path = source / "PARAMETRIC_GEOMETRY_MANIFEST.json"
    manifest_hash = _write_json(manifest_path, manifest)
    summary = {
        "component_count": 8,
        "magnet_count": 18,
        **exporter.EXPECTED_PAIR_COUNTS,
        "invalid_component_count": 0,
        "invalid_magnet_count": 0,
        "component_overlap_failures": 0,
        "magnet_component_overlap_failures": 0,
        "magnet_pair_overlap_failures": 0,
        "clearance_measurement_count": 18,
    }
    integrity = [
        {
            "path": name,
            "actual_bytes": row["bytes"],
            "actual_sha256": row["sha256"],
            "pass": True,
        }
        for name, row in artifacts.items()
    ]
    audit = {
        "schema": exporter.SOURCE_AUDIT_SCHEMA,
        "status": "PASS",
        "source_manifest_sha256": manifest_hash,
        "source_immutable": True,
        "summary": summary,
        "component_pairs": [
            {"components": [first, second], "pass": True}
            for index, first in enumerate(exporter.RADIAL_COMPONENTS)
            for second in exporter.RADIAL_COMPONENTS[index + 1 :]
        ],
        "magnet_component_pairs": [
            {
                "magnet_id": magnet,
                "component": component,
                "pass": True,
                **(
                    {
                        "minimum_clearance": {
                            "status": "MEASURED",
                            "minimum_distance_cm": 1.0,
                        }
                    }
                    if component == "vacuum_gap"
                    else {}
                ),
            }
            for magnet in exporter.MAGNET_IDS
            for component in exporter.RADIAL_COMPONENTS
        ],
        "magnet_pairs": [
            {"magnets": [first, second], "pass": True}
            for index, first in enumerate(exporter.MAGNET_IDS)
            for second in exporter.MAGNET_IDS[index + 1 :]
        ],
        "source_integrity_before": integrity,
        "source_integrity_after": integrity,
    }
    audit_path = tmp_path / "audit.json"
    audit_hash = _write_json(audit_path, audit)
    seal = {
        "schema": exporter.SOURCE_AUDIT_SEAL_SCHEMA,
        "status": "PASS",
        "report_sha256": audit_hash,
        "source_manifest_sha256": manifest_hash,
        "source_immutable": True,
    }
    seal_path = tmp_path / "seal.json"
    seal_hash = _write_json(seal_path, seal)
    producer_path = tmp_path / "audit_producer.py"
    producer_path.write_text("# test producer\n", encoding="utf-8")
    producer_hash = hashlib.sha256(producer_path.read_bytes()).hexdigest()
    return (
        source,
        manifest_hash,
        audit_path,
        audit_hash,
        seal_path,
        seal_hash,
        producer_path,
        producer_hash,
    )


def test_semantic_and_material_contract_is_26_independent_volumes():
    assert len(exporter.SEMANTIC_ROLES) == 26
    assert len(set(exporter.SEMANTIC_ROLES)) == 26
    assert exporter.SEMANTIC_ROLES[:8] == exporter.RADIAL_COMPONENTS
    assert exporter.SEMANTIC_ROLES[8:] == exporter.MAGNET_IDS
    assert exporter.MATERIAL_TAGS.count("magnets") == 18
    assert len(exporter.ALLOWED_SHARED_ROLE_PAIRS) == 7
    assert all(
        not any(role.startswith("magnet-") for role in pair)
        for pair in exporter.ALLOWED_SHARED_ROLE_PAIRS
    )


def test_source_packet_requires_passed_hash_bound_physical_audit(tmp_path):
    packet = _packet(tmp_path)
    manifest, audit = exporter._validate_source_packet(
        packet[0],
        manifest_sha256=packet[1],
        audit_path=packet[2],
        audit_sha256=packet[3],
        audit_seal_path=packet[4],
        audit_seal_sha256=packet[5],
        audit_producer_path=packet[6],
        audit_producer_sha256=packet[7],
    )
    assert manifest["magnet_inventory"]["solid_count"] == 18
    assert audit["summary"]["magnet_pair_count"] == 153


def test_source_packet_rejects_unqualified_audit(tmp_path):
    packet = list(_packet(tmp_path))
    audit = json.loads(packet[2].read_text(encoding="utf-8"))
    audit["status"] = "BLOCKED_SOURCE_CAD_PHYSICAL_GATE"
    packet[3] = _write_json(packet[2], audit)
    seal = json.loads(packet[4].read_text(encoding="utf-8"))
    seal["status"] = audit["status"]
    seal["report_sha256"] = packet[3]
    packet[5] = _write_json(packet[4], seal)
    with pytest.raises(ValueError, match="physical audit did not pass"):
        exporter._validate_source_packet(
            packet[0],
            manifest_sha256=packet[1],
            audit_path=packet[2],
            audit_sha256=packet[3],
            audit_seal_path=packet[4],
            audit_seal_sha256=packet[5],
            audit_producer_path=packet[6],
            audit_producer_sha256=packet[7],
        )


def test_source_packet_rejects_artifact_mutation(tmp_path):
    packet = _packet(tmp_path)
    (packet[0] / "magnet_set.step").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact integrity failed"):
        exporter._validate_source_packet(
            packet[0],
            manifest_sha256=packet[1],
            audit_path=packet[2],
            audit_sha256=packet[3],
            audit_seal_path=packet[4],
            audit_seal_sha256=packet[5],
            audit_producer_path=packet[6],
            audit_producer_sha256=packet[7],
        )


def _role_rows():
    rows = [
        {
            "surface_tag": index + 1,
            "owner_roles": list(pair),
            "area_cm2": 1.0,
            "periodic_plane": None,
        }
        for index, pair in enumerate(
            (
                (
                    exporter.RADIAL_COMPONENTS[index],
                    exporter.RADIAL_COMPONENTS[index + 1],
                )
                for index in range(7)
            )
        )
    ]
    rows.extend(
        [
            {
                "surface_tag": 101,
                "owner_roles": ["vacuum_gap"],
                "area_cm2": 1.0,
                "periodic_plane": None,
            },
        ]
    )
    return rows


def test_surface_role_contract_rejects_exposed_internal_radial_patch():
    rows = _role_rows()
    rows.append(
        {
            "surface_tag": 102,
            "owner_roles": ["first_wall"],
            "area_cm2": 1.0,
            "periodic_plane": None,
        }
    )
    with pytest.raises(RuntimeError, match="exposed internal-interface"):
        exporter._surface_role_contract(rows)


def test_surface_role_contract_rejects_shared_magnet_surface():
    rows = _role_rows()
    rows.append(
        {
            "surface_tag": 102,
            "owner_roles": ["magnet-0000", "vacuum_gap"],
            "area_cm2": 1.0,
            "periodic_plane": None,
        }
    )
    with pytest.raises(RuntimeError, match="forbidden roles"):
        exporter._surface_role_contract(rows)


def test_surface_role_contract_accepts_only_complete_radial_chain():
    result = exporter._surface_role_contract(_role_rows())
    assert result["pass"] is True
    assert len(result["interface_surface_ids"]) == 7


def test_surface_mesh_gate_rejects_empty_or_wrong_volume_count():
    with pytest.raises(RuntimeError, match="volume count"):
        exporter._validate_surface_mesh([], [])
    vertices = [np.zeros((3, 3)) for _ in range(26)]
    triangles = [np.zeros((1, 3), dtype=int) for _ in range(26)]
    triangles[8] = np.empty((0, 3), dtype=int)
    with pytest.raises(RuntimeError, match="magnet-0000"):
        exporter._validate_surface_mesh(vertices, triangles)


def test_faceted_magnet_centers_reject_permuted_global_ids():
    source = [[100.0 * index, 0.0, 0.0] for index in range(18)]
    passed = exporter._match_faceted_magnet_centers(source, source)
    assert [row["volume_global_id"] for row in passed] == list(range(9, 27))
    permuted = [list(row) for row in source]
    permuted[0], permuted[1] = permuted[1], permuted[0]
    with pytest.raises(RuntimeError, match="global volume 9"):
        exporter._match_faceted_magnet_centers(permuted, source)
