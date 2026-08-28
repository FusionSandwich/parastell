from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import parastell.faceting_evidence as evidence
from parastell.faceting_evidence import TriangleIncidence


def _triangle(z=0.0):
    return np.asarray(
        [[0.0, 0.0, z], [1.0, 0.0, z], [0.0, 1.0, z]],
        dtype=float,
    )


def test_sealed_source_signature_rejects_nonfinite_or_nonpositive_scalars():
    signature = {
        "mass_cm3": 1.0,
        "center_cm": [0.0, 0.0, 0.0],
        "bounding_box_cm": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        "inertia_cm5": [1.0] * 9,
    }
    assert evidence._source_signature_matches_sealed(signature, signature)

    invalid = dict(signature, mass_cm3=0.0)
    assert not evidence._source_signature_matches_sealed(invalid, signature)
    invalid = dict(signature, center_cm=[float("nan"), 0.0, 0.0])
    assert not evidence._source_signature_matches_sealed(invalid, signature)


def test_recursive_subdivision_obeys_preregistered_cover_radius():
    triangle = np.asarray(
        [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 20.0, 0.0]]
    )
    leaves = list(evidence._leaf_triangles(triangle))
    assert len(leaves) > 1
    assert max(radius for _, radius in leaves) <= 0.25
    original_area = 0.5 * np.linalg.norm(
        np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    )
    leaf_area = sum(
        0.5 * np.linalg.norm(np.cross(leaf[1] - leaf[0], leaf[2] - leaf[0]))
        for leaf, _ in leaves
    )
    assert leaf_area == pytest.approx(original_area, rel=1.0e-14)


@pytest.mark.parametrize(
    "edge_cm,expected", [(5.0, 256), (10.0, 1024), (20.0, 4096)]
)
def test_projection_cost_preflight_is_exact(edge_cm, expected):
    row = _incidence(
        "magnet-0000",
        1,
        [0.0, 0.0, 0.0],
        "magnets",
    )
    row = TriangleIncidence(
        **{
            **row.__dict__,
            "coordinates_cm": np.asarray(
                [[0.0, 0.0, 0.0], [edge_cm, 0.0, 0.0], [0.0, edge_cm, 0.0]]
            ),
        }
    )
    assert evidence._estimate_leaf_sample_count([row]) == expected


def test_projection_workload_has_a_hard_fail_closed_cap():
    assert evidence.HARD_MAXIMUM_PROJECTED_SUBTRIANGLE_CENTROIDS == 5_000_000
    row = _incidence(
        "magnet-0000",
        1,
        [0.0, 0.0, 0.0],
        "magnets",
    )
    row = TriangleIncidence(
        **{
            **row.__dict__,
            "coordinates_cm": np.asarray(
                [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 20.0, 0.0]]
            ),
        }
    )
    with pytest.raises(RuntimeError, match="bounded limit"):
        evidence._estimate_leaf_sample_count([row], maximum_count=10)


def test_required_native_tag_preserves_declared_storage_type():
    class StrictTagMesh:
        def tag_get_handle(
            self,
            name,
            size,
            data_type,
            storage,
            *,
            create_if_missing,
        ):
            assert (name, size, data_type, storage) == ("TAG", 1, 2, 8)
            assert create_if_missing is False
            return "tag-handle"

    assert (
        evidence._required_native_tag(StrictTagMesh(), "TAG", 1, 2, 8)
        == "tag-handle"
    )


def test_optional_tag_read_uses_uint64_entity_handle_array():
    class StrictTagMesh:
        def tag_get_data(self, tag, handles, *, flat):
            assert tag == "TAG"
            assert flat is True
            assert isinstance(handles, np.ndarray)
            assert handles.dtype == np.uint64
            assert handles.tolist() == [37]
            return np.asarray([123], dtype=np.int32)

    assert evidence._optional_tag_value(StrictTagMesh(), "TAG", 37) == 123


def test_dagmc_group_geometry_dimension_is_minus_one():
    assert evidence._category_dimension_pass("Group", -1) is True
    assert evidence._category_dimension_pass("Group", 4) is False
    assert evidence._category_dimension_pass("Surface", 2) is True
    assert evidence._category_dimension_pass("Surface", None) is False


class _StrictNativeLoaderMesh:
    def __init__(self, group_dimension=-1):
        self.group_dimension = group_dimension
        self.handle_requests = []
        self.tag_reads = []
        self.vertices = {
            1: [0.0, 0.0, 0.0],
            2: [1.0, 0.0, 0.0],
            3: [0.0, 1.0, 0.0],
            4: [0.0, 0.0, 1.0],
        }
        self.triangles = {
            101: [2, 3, 4],
            102: [1, 4, 3],
            103: [1, 2, 4],
            104: [1, 3, 2],
        }
        self.surface_triangles = {
            11: [101],
            12: [102],
            13: [103],
            14: [104],
        }

    def load_file(self, path):
        assert path.endswith("fixture.h5m")

    def get_root_set(self):
        return 0

    def tag_get_handle(
        self,
        name,
        size,
        data_type,
        storage,
        *,
        create_if_missing,
    ):
        self.handle_requests.append((name, size, data_type, storage))
        assert create_if_missing is False
        return name

    def tag_get_data(self, tag, handles, *, flat):
        assert flat is True
        assert isinstance(handles, np.ndarray)
        assert handles.dtype == np.uint64
        assert handles.shape == (1,)
        handle = int(handles[0])
        self.tag_reads.append((tag, handle))
        category = {
            10: b"Volume",
            11: b"Surface",
            12: b"Surface",
            13: b"Surface",
            14: b"Surface",
            20: b"Group",
        }
        if tag == "CATEGORY":
            return np.asarray([category[handle]])
        if tag == "GLOBAL_ID":
            if handle == 20:
                raise RuntimeError("MB_TAG_NOT_FOUND")
            return np.asarray([{10: 1, 11: 1, 12: 2, 13: 3, 14: 4}[handle]])
        if tag == "GEOM_DIMENSION":
            return np.asarray(
                [
                    (
                        self.group_dimension
                        if handle == 20
                        else (3 if handle == 10 else 2)
                    )
                ]
            )
        if tag == "GEOM_SENSE_2":
            return np.asarray([10, 0], dtype=np.uint64)
        if tag == "NAME":
            return np.asarray([b"mat:magnets"])
        raise AssertionError(f"unexpected tag {tag}")

    def get_entities_by_type(self, handle, entity_type):
        if entity_type == 4:
            assert handle == 0
            return [10, 11, 12, 13, 14, 20]
        if entity_type == 2:
            if handle == 0:
                return [101, 102, 103, 104]
            return self.surface_triangles[handle]
        raise AssertionError(f"unexpected entity type {entity_type}")

    def get_entities_by_handle(self, handle):
        assert handle == 20
        return [10]


def _install_fake_pymoab(monkeypatch, mesh):
    fake_types = types.SimpleNamespace(
        CATEGORY_TAG_NAME="CATEGORY",
        CATEGORY_TAG_SIZE=32,
        GLOBAL_ID_TAG_NAME="GLOBAL_ID",
        GEOM_DIMENSION_TAG_NAME="GEOM_DIMENSION",
        NAME_TAG_NAME="NAME",
        NAME_TAG_SIZE=32,
        MB_TYPE_OPAQUE=1,
        MB_TYPE_INTEGER=2,
        MB_TYPE_HANDLE=3,
        MB_TAG_SPARSE=8,
        MB_TAG_DENSE=16,
        MBENTITYSET=4,
        MBTRI=2,
    )
    module = types.ModuleType("pymoab")
    module.core = types.SimpleNamespace(Core=lambda: mesh)
    module.types = fake_types
    monkeypatch.setitem(sys.modules, "pymoab", module)


def test_native_loader_uses_pymoab_array_and_dimension_contracts(
    monkeypatch, tmp_path
):
    mesh = _StrictNativeLoaderMesh()
    _install_fake_pymoab(monkeypatch, mesh)
    loaded = evidence._load_native_mesh(tmp_path / "fixture.h5m")
    assert loaded.volume_global_ids == {10: 1}
    assert loaded.volume_materials == {1: "magnets"}
    assert ("CATEGORY", 32, 1, 8) in mesh.handle_requests
    assert ("GEOM_DIMENSION", 1, 2, 16) in mesh.handle_requests
    assert ("GEOM_DIMENSION", 20) in mesh.tag_reads


def test_native_loader_rejects_moab_entity_type_as_group_dimension(
    monkeypatch, tmp_path
):
    mesh = _StrictNativeLoaderMesh(group_dimension=4)
    _install_fake_pymoab(monkeypatch, mesh)
    with pytest.raises(ValueError, match="Group.*invalid geometry dimension"):
        evidence._load_native_mesh(tmp_path / "fixture.h5m")


def test_triangle_measurement_projects_every_covering_centroid():
    calls = []

    def plane_distance(point):
        calls.append(tuple(point))
        return abs(float(point[2]))

    result = evidence._measure_triangle_deviation(
        _triangle(z=0.75), plane_distance
    )
    assert result["leaf_sample_count"] == len(calls)
    assert result["leaf_sample_count"] > 1
    assert result["maximum_projected_distance_cm"] == pytest.approx(0.75)
    assert result["actual_maximum_cover_radius_cm"] <= 0.25


def test_certificate_records_exact_upper_bound_arithmetic():
    result = evidence._certificate_arithmetic(
        maximum_projected_distance_cm=0.2,
        actual_maximum_cover_radius_cm=0.25,
        maximum_brep_entity_tolerance_cm=1.0e-6,
    )
    assert result["solver_margin_cm"] == 1.0e-7
    assert result["upper_bound_cm"] == (0.2 + 0.25 + 1.0e-7 + 1.0e-6)
    assert result["arithmetic_reconstruction_pass"] is True
    with pytest.raises(ValueError, match="exactly"):
        evidence._certificate_arithmetic(
            maximum_projected_distance_cm=0.2,
            actual_maximum_cover_radius_cm=0.25,
            maximum_brep_entity_tolerance_cm=1.0e-6,
            solver_margin_cm=2.0e-7,
        )


def _public_report():
    witnesses = {
        "magnet-0005": [1160.39, 380.476, 128.459],
        "magnet-0006": [1095.15, 498.581, 92.5898],
        "magnet-0011": [476.445, 1095.28, -86.2664],
        "magnet-0012": [364.23, 1158.37, -121.329],
    }
    return {
        "schema": ("parastell.public_reference_overlap_quantification/v1.0.0"),
        "intersections": [
            {
                "magnet_id": magnet_id,
                "dagmc_witness_cm": witness,
                "classification": "TRUE_SOURCE_CAD_PHYSICAL_INTERSECTION",
            }
            for magnet_id, witness in witnesses.items()
        ],
    }


def _incidence(semantic_id, handle, point, material):
    point = np.asarray(point, dtype=float)
    triangle = np.asarray(
        [point, point + [1.0, 0.0, 0.0], point + [0.0, 1.0, 0.0]]
    )
    return TriangleIncidence(
        triangle_handle=handle,
        surface_global_id=handle + 100,
        volume_global_id=handle + 200,
        semantic_id=semantic_id,
        material=material,
        coordinates_cm=triangle,
    )


def test_former_zones_bind_exact_witness_and_separate_roles():
    report = _public_report()
    incidences = []
    for index, row in enumerate(report["intersections"]):
        witness = row["dagmc_witness_cm"]
        incidences.append(
            _incidence(row["magnet_id"], 10 + index, witness, "magnets")
        )
        incidences.append(
            _incidence("vac_vessel", 20 + index, witness, "vac_vessel")
        )
    rows = evidence._collect_former_overlap_zones(
        incidences,
        level="coarse",
        public_report=report,
        public_report_sha256=(evidence.EXPECTED_PUBLIC_OVERLAP_REPORT_SHA256),
    )
    assert [row["zone_id"] for row in rows] == list(
        evidence.REQUIRED_FORMER_ZONE_IDS
    )
    assert all(row["sphere_radius_cm"] == 40.0 for row in rows)
    assert all(row["magnet"]["triangle_count"] == 1 for row in rows)
    assert all(row["vac_vessel"]["triangle_count"] >= 1 for row in rows)
    assert all(row["zone_coverage_pass"] is True for row in rows)
    assert all(
        row["public_overlap_report_sha256"]
        == evidence.EXPECTED_PUBLIC_OVERLAP_REPORT_SHA256
        for row in rows
    )
    refined = []
    for row in rows:
        updated = {**row, "level": "refined"}
        updated["magnet"] = {
            **row["magnet"],
            "triangle_count": row["magnet"]["triangle_count"] + 1,
            "maximum_edge_cm": row["magnet"]["maximum_edge_cm"] / 2.0,
        }
        updated["vac_vessel"] = {
            **row["vac_vessel"],
            "triangle_count": row["vac_vessel"]["triangle_count"] + 1,
            "maximum_edge_cm": row["vac_vessel"]["maximum_edge_cm"] / 2.0,
        }
        refined.append(updated)
    combined = evidence.combine_former_overlap_zone_evidence(rows, refined)
    assert combined[0]["witness_cm"] == rows[0]["witness_cm"]
    assert combined[0]["coarse"]["magnet"] == rows[0]["magnet"]
    assert combined[0]["refined"]["magnet"] == refined[0]["magnet"]


def test_public_witness_inventory_fails_closed():
    report = _public_report()
    report["intersections"].pop()
    with pytest.raises(ValueError, match="inventory"):
        evidence._validate_public_overlap_report(
            report, evidence.EXPECTED_PUBLIC_OVERLAP_REPORT_SHA256
        )
    with pytest.raises(ValueError, match="not preregistered"):
        evidence._validate_public_overlap_report(_public_report(), "0" * 64)


def test_protocol_contract_rejects_drift():
    protocol = {
        "schema": "parastell.faceting_comparison_protocol/v1.0.0",
        "levels": {
            "coarse": {
                "minimum_mesh_size_cm": 5.0,
                "maximum_mesh_size_cm": 20.0,
                "algorithm": 1,
            }
        },
        "facet_deviation_certificate": {
            "direction": "dagmc_facets_to_source_cad_boundary",
            "source_boundary": "compound_of_all_brep_faces",
            "maximum_cover_radius_cm": 0.25,
            "solver_margin_cm": 1.0e-7,
            "include_maximum_brep_entity_tolerance": True,
            "require_all_physical_volume_triangles": True,
            "require_all_magnet_triangles": True,
            "projection_errors_allowed": 0,
        },
    }
    assert evidence._protocol_level(protocol, "coarse") == {
        "minimum_size_cm": 5.0,
        "maximum_size_cm": 20.0,
        "algorithm": 1,
    }
    protocol["facet_deviation_certificate"]["maximum_cover_radius_cm"] = 0.3
    with pytest.raises(ValueError, match="not supported"):
        evidence._protocol_level(protocol, "coarse")


class _FakeMesh:
    def __init__(self):
        self.vertices = {
            1: [0.0, 0.0, 0.0],
            2: [1.0, 0.0, 0.0],
            3: [0.0, 1.0, 0.0],
            4: [0.0, 0.0, 1.0],
        }
        self.triangles = {
            101: [2, 3, 4],
            102: [1, 4, 3],
            103: [1, 2, 4],
            104: [1, 3, 2],
        }

    def get_connectivity(self, handles):
        assert isinstance(handles, np.ndarray)
        assert handles.dtype == np.uint64
        assert handles.shape == (1,)
        return self.triangles[int(handles[0])]

    def get_coords(self, handles):
        assert isinstance(handles, np.ndarray)
        assert handles.dtype == np.uint64
        return np.asarray(
            [self.vertices[int(value)] for value in handles]
        ).ravel()


def _one_volume_native():
    surface_handles = [11, 12, 13, 14]
    triangle_handles = [101, 102, 103, 104]
    return evidence.NativeMesh(
        mesh=_FakeMesh(),
        root_triangle_handles=set(triangle_handles),
        surface_triangles=dict(
            zip(surface_handles, ([v] for v in triangle_handles))
        ),
        surface_global_ids=dict(zip(surface_handles, [1, 2, 3, 4])),
        surface_senses={value: (10, 0) for value in surface_handles},
        volume_global_ids={10: 1},
        volume_materials={1: "magnets"},
        volume_surfaces={10: surface_handles},
    )


def _one_target():
    return evidence.SourceTarget(
        volume_global_id=1,
        semantic_id="magnet-0000",
        material="magnets",
        source_step="magnet_set.step",
        source_step_sha256="a" * 64,
        source_solid_index=0,
        boundary_shape=object(),
        maximum_entity_tolerance_cm=1.0e-7,
        brep_entity_count=12,
    )


def test_semantic_snapshot_reconciles_native_triangles_and_closure():
    snapshot, incidences = evidence._collect_snapshot(
        _one_volume_native(),
        {1: _one_target()},
        level="coarse",
        protocol_sha256=evidence.EXPECTED_PROTOCOL_SHA256,
        h5m_sha256="b" * 64,
        criteria_sha256="a" * 64,
        physical_input_identity={
            "source_mesh_canonical_fingerprint": "d" * 64
        },
        physical_input_identity_sha256="c" * 64,
        source_inventory=[],
        projection_targets=[],
        faceting_settings={
            "minimum_size_cm": 5.0,
            "maximum_size_cm": 20.0,
            "algorithm": 1,
        },
    )
    assert len(incidences) == 4
    assert snapshot["physical_triangle_count"] == 4
    assert snapshot["covered_physical_triangle_count"] == 4
    assert snapshot["all_physical_triangles_covered"] is True
    assert snapshot["native_topology_gate_pass"] is True
    assert snapshot["schema"] == "parastell.faceting_snapshot/v1.0.0"
    assert snapshot["criteria_sha256"] == "a" * 64
    assert snapshot["construct_only"] is False
    assert snapshot["build_status"] == ("BUILD_COMPLETE_PENDING_QUALIFICATION")
    assert snapshot["physical_input_identity"]
    assert snapshot["source_mesh_canonical_fingerprint"] == "d" * 64
    assert snapshot["semantic_volumes"][0]["volume_cm3"] == pytest.approx(
        1.0 / 6.0
    )
    assert snapshot["semantic_volumes"][0][
        "signed_volume_cm3"
    ] == pytest.approx(1.0 / 6.0)


def test_semantic_topology_signature_is_raw_id_order_independent():
    volumes = [
        {
            "semantic_id": "b",
            "material": "m2",
            "boundary_surface_count": 2,
        },
        {
            "semantic_id": "a",
            "material": "m1",
            "boundary_surface_count": 1,
        },
    ]
    adjacency = [
        {"surface_global_id": 99, "semantic_pair": ["b", "a"]},
        {"surface_global_id": 2, "semantic_pair": ["a", "b"]},
    ]
    first = evidence._semantic_topology_signature(volumes, adjacency, 3)
    second = evidence._semantic_topology_signature(
        list(reversed(volumes)), list(reversed(adjacency)), 3
    )
    assert first == second


def test_deviation_certificate_fails_closed_on_projection_error(monkeypatch):
    snapshot, incidences = evidence._collect_snapshot(
        _one_volume_native(),
        {1: _one_target()},
        level="coarse",
        protocol_sha256=evidence.EXPECTED_PROTOCOL_SHA256,
        h5m_sha256="b" * 64,
        criteria_sha256="a" * 64,
        physical_input_identity={
            "source_mesh_canonical_fingerprint": "d" * 64
        },
        physical_input_identity_sha256="c" * 64,
        source_inventory=[],
        projection_targets=[],
        faceting_settings={
            "minimum_size_cm": 5.0,
            "maximum_size_cm": 20.0,
            "algorithm": 1,
        },
    )

    calls = 0

    def fail_one(_, point):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic failure")
        return 0.0

    monkeypatch.setattr(evidence, "_project_to_boundary", fail_one)
    certificate = evidence._collect_deviation_certificate(
        incidences,
        {1: _one_target()},
        snapshot,
        level="coarse",
        protocol_sha256=evidence.EXPECTED_PROTOCOL_SHA256,
        h5m_sha256="b" * 64,
        physical_input_identity_sha256="c" * 64,
        source_inventory=[],
        projection_targets=[],
    )
    assert certificate["projection_error_count"] > 0
    assert certificate["exact_triangle_reconciliation_pass"] is False
    assert certificate["certified_upper_bound"] is False


def test_deviation_certificate_binds_every_required_field(
    monkeypatch, tmp_path
):
    snapshot, incidences = evidence._collect_snapshot(
        _one_volume_native(),
        {1: _one_target()},
        level="refined",
        protocol_sha256=evidence.EXPECTED_PROTOCOL_SHA256,
        h5m_sha256="b" * 64,
        criteria_sha256="a" * 64,
        physical_input_identity={
            "source_mesh_canonical_fingerprint": "d" * 64
        },
        physical_input_identity_sha256="c" * 64,
        source_inventory=[
            {
                "source_step": "magnet_set.step",
                "sha256": "a" * 64,
                "size_bytes": 1,
                "solid_count": 1,
            }
        ],
        projection_targets=[],
        faceting_settings={
            "minimum_size_cm": 2.5,
            "maximum_size_cm": 10.0,
            "algorithm": 1,
        },
    )
    monkeypatch.setattr(
        evidence,
        "_project_to_boundary",
        lambda _boundary, point: abs(float(point[2])),
    )
    certificate = evidence._collect_deviation_certificate(
        incidences,
        {1: _one_target()},
        snapshot,
        level="refined",
        protocol_sha256=evidence.EXPECTED_PROTOCOL_SHA256,
        h5m_sha256="b" * 64,
        physical_input_identity_sha256="c" * 64,
        source_inventory=snapshot["source_solid_hash_inventory"],
        projection_targets=snapshot["semantic_projection_targets"],
        progress_path=tmp_path / "progress.json",
        progress_every_triangle_incidences=1,
    )
    assert certificate["schema"] == evidence.DEVIATION_SCHEMA
    assert certificate["level"] == "refined"
    assert certificate["direction"] == ("dagmc_facets_to_source_cad_boundary")
    assert certificate["faceting_protocol_sha256"] == (
        evidence.EXPECTED_PROTOCOL_SHA256
    )
    assert certificate["raw_h5m_sha256_before"] == "b" * 64
    assert certificate["physical_input_identity_sha256"] == "c" * 64
    assert certificate["actual_maximum_cover_radius_cm"] <= 0.25
    assert certificate["solver_margin_cm"] == 1.0e-7
    assert certificate["projection_error_count"] == 0
    assert certificate["all_physical_triangles_covered"] is True
    assert certificate["all_magnet_triangles_covered"] is True
    assert certificate["semantic_projection_target_pass"] is True
    assert certificate["exact_triangle_reconciliation_pass"] is True
    assert certificate["certified_upper_bound"] is True
    assert certificate["schema"] == (
        "parastell.facet_deviation_certificate/v1.0.0"
    )
    assert (
        certificate["covered_triangle_count"]
        == certificate["physical_triangle_count"]
    )
    assert certificate["coverage_reconciliation_pass"] is True
    assert certificate["projection_target_semantic_match_pass"] is True
    assert (
        certificate["triangle_id_digest_sha256"]
        == certificate["expected_triangle_incidence_digest_sha256"]
    )
    assert certificate["source_solid_sha256"] == {"magnet_set.step": "a" * 64}
    progress = evidence.json.loads(
        (tmp_path / "progress.json").read_text(encoding="utf-8")
    )
    assert progress["status"] == "TERMINAL"
    assert progress["completed_triangle_incidence_count"] == len(incidences)


def test_source_map_rejects_noncanonical_semantic_inventory(tmp_path):
    source_map = {
        "schema": evidence.SOURCE_MAP_SCHEMA,
        "coordinate_units": "cm",
        "volumes": [
            {
                "volume_global_id": 1,
                "semantic_id": "magnet-0000",
                "material": "magnets",
                "source_step": "magnet_set.step",
                "source_step_sha256": "a" * 64,
                "source_solid_index": 0,
            }
        ],
    }
    with pytest.raises(ValueError, match="semantic inventory"):
        evidence._load_source_targets(
            source_root=tmp_path,
            source_map=source_map,
            volume_materials={1: "magnets"},
        )


def test_source_map_rejects_magnet_global_id_permutation(tmp_path):
    semantic_ids = list(evidence.EXPECTED_SEMANTIC_MATERIALS)
    source_map_rows = []
    volume_materials = {}
    for index, semantic_id in enumerate(semantic_ids):
        global_id = index + 1
        source_step, solid_index = evidence.EXPECTED_SOURCE_TARGETS[
            semantic_id
        ]
        source_map_rows.append(
            {
                "volume_global_id": global_id,
                "semantic_id": semantic_id,
                "material": evidence.EXPECTED_SEMANTIC_MATERIALS[semantic_id],
                "source_step": source_step,
                "source_step_sha256": "a" * 64,
                "source_solid_index": solid_index,
            }
        )
        volume_materials[global_id] = evidence.EXPECTED_SEMANTIC_MATERIALS[
            semantic_id
        ]
    (
        source_map_rows[6]["volume_global_id"],
        source_map_rows[7]["volume_global_id"],
    ) = (
        source_map_rows[7]["volume_global_id"],
        source_map_rows[6]["volume_global_id"],
    )
    manifest = {
        "volume_mapping_evidence": [
            {
                "index": index,
                "semantic_role": semantic_id,
                "source_to_import_signature_pass": True,
                "signature_pass": True,
            }
            for index, semantic_id in enumerate(semantic_ids)
        ],
        "written_h5m_material_order_evidence": {
            "volume_global_ids": list(range(1, 25)),
            "pass": True,
        },
    }
    with pytest.raises(ValueError, match="sealed refaceting semantic mapping"):
        evidence._load_source_targets(
            source_root=tmp_path,
            source_map={
                "schema": evidence.SOURCE_MAP_SCHEMA,
                "coordinate_units": "cm",
                "volumes": source_map_rows,
            },
            volume_materials=volume_materials,
            refaceting_manifest=manifest,
        )


def test_source_map_rejects_reloaded_same_material_magnet_reordering(
    tmp_path, monkeypatch
):
    class _Center:
        def __init__(self, value):
            self.value = value

        def toTuple(self):
            return (self.value, self.value + 1.0, self.value + 2.0)

    class _Bounds:
        def __init__(self, value):
            self.xmin = value
            self.ymin = value + 1.0
            self.zmin = value + 2.0
            self.xmax = value + 3.0
            self.ymax = value + 4.0
            self.zmax = value + 5.0

    class _Solid:
        def __init__(self, value):
            self.value = float(value)

        def Volume(self):
            return 100.0 + self.value

        def Center(self):
            return _Center(self.value)

        def BoundingBox(self):
            return _Bounds(self.value)

    class _Selection:
        def __init__(self, solids):
            self._solids = solids

        def solids(self):
            return self

        def vals(self):
            return list(self._solids)

    component_files = {
        "chamber.step": [_Solid(0)],
        "first_wall.step": [_Solid(1)],
        "breeder.step": [_Solid(2)],
        "back_wall.step": [_Solid(3)],
        "shield.step": [_Solid(4)],
        "vacuum_vessel.step": [_Solid(5)],
    }
    canonical_magnets = [_Solid(index + 6) for index in range(18)]
    reloaded_magnets = list(canonical_magnets)
    reloaded_magnets[0], reloaded_magnets[1] = (
        reloaded_magnets[1],
        reloaded_magnets[0],
    )

    def _import_step(path):
        name = evidence.Path(path).name
        if name == "magnet_set.step":
            return _Selection(reloaded_magnets)
        return _Selection(component_files[name])

    fake_cadquery = types.SimpleNamespace(
        importers=types.SimpleNamespace(importStep=_import_step),
        Shape=types.SimpleNamespace(
            matrixOfInertia=lambda solid: [
                [solid.value + row * 3 + column for column in range(3)]
                for row in range(3)
            ]
        ),
    )
    monkeypatch.setitem(sys.modules, "cadquery", fake_cadquery)
    monkeypatch.setattr(evidence, "sha256_file", lambda _path: "a" * 64)
    monkeypatch.setattr(
        evidence,
        "_source_boundary",
        lambda _solid: (object(), 0.0, 1),
    )

    canonical_by_semantic = {}
    for index, semantic_id in enumerate(evidence.EXPECTED_SEMANTIC_MATERIALS):
        solid = (
            list(component_files.values())[index][0]
            if index < 6
            else canonical_magnets[index - 6]
        )
        canonical_by_semantic[semantic_id] = {
            "mass_cm3": solid.Volume(),
            "center_cm": list(solid.Center().toTuple()),
            "bounding_box_cm": [
                solid.BoundingBox().xmin,
                solid.BoundingBox().ymin,
                solid.BoundingBox().zmin,
                solid.BoundingBox().xmax,
                solid.BoundingBox().ymax,
                solid.BoundingBox().zmax,
            ],
            "inertia_cm5": [
                value
                for row in fake_cadquery.Shape.matrixOfInertia(solid)
                for value in row
            ],
        }
    source_rows = []
    volume_materials = {}
    for index, semantic_id in enumerate(evidence.EXPECTED_SEMANTIC_MATERIALS):
        filename, solid_index = evidence.EXPECTED_SOURCE_TARGETS[semantic_id]
        (tmp_path / filename).touch(exist_ok=True)
        source_rows.append(
            {
                "volume_global_id": index + 1,
                "semantic_id": semantic_id,
                "material": evidence.EXPECTED_SEMANTIC_MATERIALS[semantic_id],
                "source_step": filename,
                "source_step_sha256": "a" * 64,
                "source_solid_index": solid_index,
            }
        )
        volume_materials[index + 1] = evidence.EXPECTED_SEMANTIC_MATERIALS[
            semantic_id
        ]
    manifest = {
        "volume_mapping_evidence": [
            {
                "index": index,
                "semantic_role": semantic_id,
                "source_to_import_signature_pass": True,
                "signature_pass": True,
                "source_solid_signature": canonical_by_semantic[semantic_id],
            }
            for index, semantic_id in enumerate(
                evidence.EXPECTED_SEMANTIC_MATERIALS
            )
        ],
        "written_h5m_material_order_evidence": {
            "volume_global_ids": list(range(1, 25)),
            "pass": True,
        },
    }
    with pytest.raises(ValueError, match="magnet-0000"):
        evidence._load_source_targets(
            source_root=tmp_path,
            source_map={
                "schema": evidence.SOURCE_MAP_SCHEMA,
                "coordinate_units": "cm",
                "volumes": source_rows,
            },
            volume_materials=volume_materials,
            refaceting_manifest=manifest,
        )


def test_refaceting_manifest_binds_physical_identity_and_h5m():
    identity = {
        "vmec_sha256": "d" * 64,
        "source_mesh": "e" * 64,
        "acceptance_criteria_sha256": "a" * 64,
    }
    identity_hash = evidence._json_sha256(identity)
    manifest = {
        "schema": "parastell.refaceted_source_cad_candidate/v1.0.0",
        "refaceting_only": True,
        "status": "BUILD_COMPLETE_PENDING_QUALIFICATION",
        "construct_only": False,
        "inherited_source_cad_physical_gate_pass": True,
        "faceting_protocol": {"sha256": evidence.EXPECTED_PROTOCOL_SHA256},
        "acceptance_criteria": {"sha256": "a" * 64},
        "source_cad_qualification": {
            "source_manifest_sha256": "1" * 64,
            "source_audit_sha256": "2" * 64,
            "source_audit_seal_sha256": "3" * 64,
            "physical_change_report_sha256": "4" * 64,
            "reference_manifest_sha256": "5" * 64,
        },
        "source_topology_contract": {"pass": True},
        "gmsh_pre_mesh_topology_contract": {"pass": True},
        "written_h5m_topology_contract": {"pass": True},
        "written_h5m_material_order_evidence": {"pass": True},
        "faceting": {
            "level": "coarse",
            "minimum_mesh_size_cm": 5.0,
            "maximum_mesh_size_cm": 20.0,
            "algorithm": 1,
        },
        "physical_input_identity": identity,
        "physical_input_identity_sha256": identity_hash,
        "artifacts": {"dagmc.h5m": {"sha256": "f" * 64}},
        "input_immutability_pass": True,
    }
    evidence._validate_refaceting_manifest(
        manifest,
        level="coarse",
        settings={
            "minimum_size_cm": 5.0,
            "maximum_size_cm": 20.0,
            "algorithm": 1,
        },
        protocol_sha256=evidence.EXPECTED_PROTOCOL_SHA256,
        h5m_sha256="f" * 64,
        expected_physical_identity_sha256=identity_hash,
    )
    wrong_level = dict(manifest)
    wrong_level["faceting"] = {**manifest["faceting"], "level": "refined"}
    with pytest.raises(ValueError, match="level"):
        evidence._validate_refaceting_manifest(
            wrong_level,
            level="coarse",
            settings={
                "minimum_size_cm": 5.0,
                "maximum_size_cm": 20.0,
                "algorithm": 1,
            },
            protocol_sha256=evidence.EXPECTED_PROTOCOL_SHA256,
            h5m_sha256="f" * 64,
            expected_physical_identity_sha256=identity_hash,
        )
    manifest["physical_input_identity"]["source_mesh"] = "0" * 64
    with pytest.raises(ValueError, match="content hash"):
        evidence._validate_refaceting_manifest(
            manifest,
            level="coarse",
            settings={
                "minimum_size_cm": 5.0,
                "maximum_size_cm": 20.0,
                "algorithm": 1,
            },
            protocol_sha256=evidence.EXPECTED_PROTOCOL_SHA256,
            h5m_sha256="f" * 64,
            expected_physical_identity_sha256=identity_hash,
        )


def test_refaceting_seal_requires_every_upstream_geometry_gate():
    seal = {
        "schema": "parastell.refaceted_source_cad_candidate_seal/v1.0.0",
        "candidate_build_manifest_sha256": "a" * 64,
        "dagmc_h5m_sha256": "b" * 64,
        "physical_input_identity_sha256": "c" * 64,
        "inherited_source_cad_physical_gate_pass": True,
        "input_immutability_pass": True,
        "source_topology_contract_pass": True,
        "gmsh_pre_mesh_topology_contract_pass": True,
        "written_h5m_topology_contract_pass": True,
    }
    evidence._validate_refaceting_seal(
        seal,
        manifest_sha256="a" * 64,
        h5m_sha256="b" * 64,
        physical_input_identity_sha256="c" * 64,
    )
    seal["written_h5m_topology_contract_pass"] = False
    with pytest.raises(ValueError, match="failed gates"):
        evidence._validate_refaceting_seal(
            seal,
            manifest_sha256="a" * 64,
            h5m_sha256="b" * 64,
            physical_input_identity_sha256="c" * 64,
        )


def test_native_qualification_requires_all_overlap_precisions():
    report = {
        "schema": "parastell.dagmc_native_qualification/v1.0.0",
        "raw_h5m_sha256_before": "a" * 64,
        "raw_h5m_sha256_after": "a" * 64,
        "h5m_unchanged": True,
        "native_dagmc_gate_pass": True,
        "check_watertight": {
            "pass": True,
            "unmatched_edge_count": 0,
            "unsealed_surface_count": 0,
            "unsealed_volume_count": 0,
        },
        "overlap_checks": [
            {
                "points_per_edge": value,
                "pass": True,
                "terminal_overlap_location_count": 0,
                "parsed_overlap_location_count": 0,
                "row_count_matches_terminal": True,
                "precision_header_pass": True,
            }
            for value in (1, 2, 4)
        ],
    }
    evidence._validate_native_qualification(report, "a" * 64)
    report["overlap_checks"].pop()
    with pytest.raises(ValueError, match="failed"):
        evidence._validate_native_qualification(report, "a" * 64)


def test_native_qualification_rejects_missing_hashes_or_terminal_gate():
    report = {
        "schema": "parastell.dagmc_native_qualification/v1.0.0",
        "raw_h5m_sha256_before": "a" * 64,
        "raw_h5m_sha256_after": "a" * 64,
        "h5m_unchanged": True,
        "native_dagmc_gate_pass": True,
        "check_watertight": {
            "pass": True,
            "unmatched_edge_count": 0,
            "unsealed_surface_count": 0,
            "unsealed_volume_count": 0,
        },
        "overlap_checks": [
            {
                "points_per_edge": value,
                "pass": True,
                "terminal_overlap_location_count": 0,
                "parsed_overlap_location_count": 0,
                "row_count_matches_terminal": True,
                "precision_header_pass": True,
            }
            for value in (1, 2, 4)
        ],
    }
    for key in (
        "raw_h5m_sha256_before",
        "raw_h5m_sha256_after",
        "native_dagmc_gate_pass",
    ):
        malformed = dict(report)
        malformed.pop(key)
        with pytest.raises(ValueError, match="failed"):
            evidence._validate_native_qualification(malformed, "a" * 64)


def test_real_occ_boundary_projection_uses_faces_not_solid_interior():
    import cadquery as cq

    solid = cq.Workplane("XY").box(2.0, 2.0, 2.0).val()
    boundary, tolerance, entity_count = evidence._source_boundary(solid)
    assert tolerance >= 0.0
    assert entity_count > 0
    assert evidence._project_to_boundary(boundary, (1.0, 0.0, 0.0)) == (
        pytest.approx(0.0)
    )
    assert evidence._project_to_boundary(boundary, (0.0, 0.0, 0.0)) == (
        pytest.approx(1.0)
    )


@pytest.mark.parametrize("mutate_controls", [True, False])
def test_terminal_seal_immutability_check_rejects_drift(mutate_controls):
    controls = {"h5m_sha256": "a" * 64, "protocol_sha256": "b" * 64}
    source = [{"source_step": "one.step", "sha256": "c" * 64}]
    changed_controls = dict(controls)
    changed_source = [dict(row) for row in source]
    if mutate_controls:
        changed_controls["h5m_sha256"] = "0" * 64
    else:
        changed_source[0]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="before terminal sealing"):
        evidence._assert_input_immutability(
            controls,
            changed_controls,
            source,
            changed_source,
            phase="before terminal sealing",
        )
