import json
import copy
import math

import pytest

from parastell.reference_geometry import sha256_file
from parastell.source_cad_refaceting import _faceting_level
from parastell.source_cad_refaceting import _fragment_ordered_volumes
from parastell.source_cad_refaceting import _topology_contract_from_gmsh
from parastell.source_cad_refaceting import (
    _topology_contract_from_native_report,
)
from parastell.source_cad_refaceting import _valid_volume_signature
from parastell.source_cad_refaceting import _source_import_signature_match
from parastell.source_cad_refaceting import _volume_signature_match
from parastell.source_cad_refaceting import MATERIAL_TAGS
from parastell.source_cad_refaceting import validate_source_cad_packet


def test_material_order_matches_original_public_builder():
    assert MATERIAL_TAGS[:6] == (
        "Vacuum",
        "first_wall",
        "breeder",
        "back_wall",
        "shield",
        "vac_vessel",
    )
    assert MATERIAL_TAGS[6:] == ("magnets",) * 18


def _valid_native_report():
    owners = {
        1: [1, 2],
        2: [2, 3],
        3: [3, 4],
        4: [4, 5],
        5: [5, 6],
    }
    for surface_id in range(6, 143):
        owners[surface_id] = [((surface_id - 6) % 24) + 1]
    surfaces_by_volume = {volume_id: [] for volume_id in range(1, 25)}
    for surface_id, volume_ids in owners.items():
        for volume_id in volume_ids:
            surfaces_by_volume[volume_id].append(surface_id)
    volume_materials = []
    for volume_id, material in enumerate(MATERIAL_TAGS, start=1):
        volume_materials.append(
            {
                "volume_global_id": volume_id,
                "material_groups": [material],
                "pass": True,
            }
        )
    incidence = []
    closures = []
    for volume_id, surface_ids in surfaces_by_volume.items():
        incidence.append(
            {
                "volume_global_id": volume_id,
                "classified_child_global_ids": list(surface_ids),
                "sense_surface_global_ids": list(surface_ids),
                "pass": True,
            }
        )
        closures.append(
            {
                "volume_global_id": volume_id,
                "surface_global_ids": list(surface_ids),
                "pass": True,
            }
        )
    return {
        "schema": "parastell.native_moab_topology/v1.0.0",
        "raw_h5m_sha256_before": "a" * 64,
        "raw_h5m_sha256_after": "a" * 64,
        "h5m_unchanged": True,
        "native_topology_gate_pass": True,
        "material_group_gate_pass": True,
        "volume_global_ids": list(range(1, 25)),
        "surface_global_ids": list(range(1, 143)),
        "surface_senses": [
            {
                "surface_global_id": surface_id,
                "sense_volume_global_ids": volume_ids,
                "pass": True,
            }
            for surface_id, volume_ids in owners.items()
        ],
        "volume_incidence": incidence,
        "volume_closures": closures,
        "volume_materials": volume_materials,
    }


def test_native_topology_contract_cross_reconciles_all_representations():
    report = _valid_native_report()
    contract = _topology_contract_from_native_report(report)
    assert contract["pass"] is True
    assert contract["volume_count"] == 24
    assert contract["surface_count"] == 142
    assert contract["internal_surface_count"] == 5
    assert contract["exterior_surface_count"] == 137
    assert contract["total_incidence_count"] == 147
    assert len(contract["canonical_contract_sha256"]) == 64

    missing_sense = copy.deepcopy(report)
    missing_sense["surface_senses"].pop()
    with pytest.raises(ValueError, match="surface-sense"):
        _topology_contract_from_native_report(missing_sense)

    closure_drift = copy.deepcopy(report)
    closure_drift["volume_closures"][0]["surface_global_ids"].pop()
    with pytest.raises(ValueError, match="representations disagree"):
        _topology_contract_from_native_report(closure_drift)

    wrong_pair = copy.deepcopy(report)
    wrong_pair["surface_senses"][0]["sense_volume_global_ids"] = [1, 3]
    with pytest.raises(ValueError, match="ownership disagree"):
        _topology_contract_from_native_report(wrong_pair)


def test_volume_signature_rejects_nonfinite_or_malformed_values():
    signature = {
        "mass_cm3": 1.0,
        "center_cm": [1.0, 2.0, 3.0],
        "bounding_box_cm": [0.0, 0.0, 0.0, 2.0, 3.0, 4.0],
        "inertia_cm5": [1.0] * 9,
    }
    assert _valid_volume_signature(signature) is True
    assert _volume_signature_match(signature, copy.deepcopy(signature)) is True
    malformed = copy.deepcopy(signature)
    malformed["center_cm"] = [1.0, 2.0]
    assert _valid_volume_signature(malformed) is False
    assert _volume_signature_match(signature, malformed) is False
    nonfinite = copy.deepcopy(signature)
    nonfinite["mass_cm3"] = math.nan
    assert _valid_volume_signature(nonfinite) is False


def test_source_import_signature_uses_inertia_not_unlocated_cq_bounds():
    source = {
        "mass_cm3": 8.0,
        "center_cm": [1.0, 2.0, 3.0],
        "bounding_box_cm": [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        "inertia_cm5": [4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0],
    }
    imported = copy.deepcopy(source)
    imported["bounding_box_cm"] = [-200.0, -100.0, 0.0, 200.0, 100.0, 2.0]

    assert _source_import_signature_match(source, imported) is True
    assert _volume_signature_match(source, imported) is False

    imported["inertia_cm5"][0] = 5.0
    assert _source_import_signature_match(source, imported) is False


class _FakeTopologyModel:
    def __init__(self, report, *, orphan_surface=False, non_surface=False):
        self._surfaces_by_volume = {
            int(row["volume_global_id"]): list(row["sense_surface_global_ids"])
            for row in report["volume_incidence"]
        }
        self._owners = {
            int(row["surface_global_id"]): list(row["sense_volume_global_ids"])
            for row in report["surface_senses"]
        }
        self._surface_ids = list(report["surface_global_ids"])
        if orphan_surface:
            self._surface_ids.append(999)
        self._non_surface = non_surface

    def getBoundary(self, volumes, combined, oriented, recursive):
        assert combined is False and oriented is False and recursive is False
        volume_id = int(volumes[0][1])
        dimension = 1 if self._non_surface and volume_id == 1 else 2
        return [
            (dimension, surface_id)
            for surface_id in self._surfaces_by_volume[volume_id]
        ]

    def getEntities(self, dimension):
        assert dimension == 2
        return [(2, surface_id) for surface_id in self._surface_ids]

    def getAdjacencies(self, dimension, surface_id):
        assert dimension == 2
        return (self._owners.get(surface_id, []), [])


class _FakeTopologyGmsh:
    def __init__(self, report, **kwargs):
        self.model = _FakeTopologyModel(report, **kwargs)


def test_gmsh_topology_contract_rejects_orphan_and_non_surface_entities():
    report = _valid_native_report()
    ordered_volumes = [(3, volume_id) for volume_id in range(1, 25)]
    contract = _topology_contract_from_gmsh(
        _FakeTopologyGmsh(report), ordered_volumes
    )
    assert contract["pass"] is True
    with pytest.raises(RuntimeError, match="orphan"):
        _topology_contract_from_gmsh(
            _FakeTopologyGmsh(report, orphan_surface=True), ordered_volumes
        )
    with pytest.raises(RuntimeError, match="non-surface"):
        _topology_contract_from_gmsh(
            _FakeTopologyGmsh(report, non_surface=True), ordered_volumes
        )


def test_faceting_level_requires_protocol_criteria_agreement():
    criteria = {
        "faceting": {
            "coarse_mesh_size_cm": [5.0, 20.0],
            "refined_mesh_size_cm": [2.5, 10.0],
        }
    }
    protocol = {
        "levels": {
            "coarse": {
                "minimum_mesh_size_cm": 5.0,
                "maximum_mesh_size_cm": 20.0,
                "algorithm": 1,
            },
            "refined": {
                "minimum_mesh_size_cm": 2.5,
                "maximum_mesh_size_cm": 10.0,
                "algorithm": 1,
            },
        }
    }
    assert (
        _faceting_level(criteria, protocol, "refined")["maximum_mesh_size_cm"]
        == 10.0
    )
    protocol["levels"]["refined"]["maximum_mesh_size_cm"] = 9.0
    with pytest.raises(ValueError, match="disagree"):
        _faceting_level(criteria, protocol, "refined")


class _FakeOcc:
    def __init__(self, mapping):
        self.mapping = mapping
        self.synchronized = False

    def fragment(self, objects, tools, removeObject, removeTool):
        assert len(objects) == 1
        assert len(tools) == 23
        assert removeObject is True
        assert removeTool is True
        return (objects + tools, self.mapping)

    def synchronize(self):
        self.synchronized = True


class _FakeModel:
    def __init__(self, mapping):
        self.occ = _FakeOcc(mapping)
        self._volumes = [row[0] for row in mapping]

    def getEntities(self, dimension):
        assert dimension == 3
        return self._volumes


class _FakeGmsh:
    def __init__(self, mapping):
        self.model = _FakeModel(mapping)


def test_fragment_mapping_preserves_one_unique_volume_per_material():
    imported = [(3, index) for index in range(1, 25)]
    mapping = [[value] for value in reversed(imported)]
    gmsh = _FakeGmsh(mapping)

    assert _fragment_ordered_volumes(gmsh, imported) == [
        value for value in reversed(imported)
    ]
    assert gmsh.model.occ.synchronized is True


def test_fragment_mapping_rejects_split_or_merged_material_ownership():
    imported = [(3, index) for index in range(1, 25)]
    split = [[value] for value in imported]
    split[0] = [(3, 1), (3, 25)]
    with pytest.raises(RuntimeError, match="maps to 2"):
        _fragment_ordered_volumes(_FakeGmsh(split), imported)

    merged = [[value] for value in imported]
    merged[1] = merged[0]
    with pytest.raises(RuntimeError, match="multiple source"):
        _fragment_ordered_volumes(_FakeGmsh(merged), imported)


def test_source_packet_rejects_artifact_or_seal_drift(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    names = (
        "chamber.step",
        "first_wall.step",
        "breeder.step",
        "back_wall.step",
        "shield.step",
        "vacuum_vessel.step",
        "magnet_set.step",
        "source_mesh.h5m",
        "dagmc.h5m",
    )
    for index, name in enumerate(names):
        (source / name).write_bytes(f"artifact-{index}".encode())
    artifacts = {
        name: {
            "size_bytes": (source / name).stat().st_size,
            "sha256": sha256_file(source / name),
        }
        for name in names
    }
    manifest = {
        "construct_only": False,
        "status": "BUILD_COMPLETE_PENDING_QUALIFICATION",
        "artifacts": artifacts,
    }
    manifest_path = source / "candidate_build_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    physical_path = tmp_path / "physical.json"
    physical_path.write_text(
        json.dumps({"physical_change_gate_pass": True}), encoding="utf-8"
    )
    physical_hash = sha256_file(physical_path)
    audit = {
        "schema": "parastell.source_cad_candidate_audit/v1.4.0",
        "source_cad_gate_pass": True,
        "source_cad_physical_gate_pass": True,
        "postflight": {"input_immutability_pass": True},
        "pair_identity_contract": {"pass": True},
        "physical_change_binding": {
            "pass": True,
            "sha256": physical_hash,
        },
        "physical_change_support_evidence": {"pass": True},
        "candidate_integrity": {
            "pass": True,
            "manifest_sha256": sha256_file(manifest_path),
        },
        "candidate_h5m_sha256": sha256_file(source / "dagmc.h5m"),
        "reference_integrity": {
            "pass": True,
            "manifest_sha256": "r" * 64,
        },
        "criteria": {"acceptance_criteria_sha256": "c" * 64},
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    audit_hash = sha256_file(audit_path)
    seal = {
        "schema": "parastell.source_cad_candidate_audit_seal/v1.0.0",
        "source_cad_audit_sha256": audit_hash,
        "source_cad_gate_pass": True,
        "input_immutability_pass": True,
    }
    seal_path = tmp_path / "seal.json"
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    packet = validate_source_cad_packet(
        source,
        audit_path,
        seal_path,
        physical_path,
        expected_source_manifest_sha256=sha256_file(manifest_path),
        expected_source_audit_sha256=audit_hash,
        expected_source_audit_seal_sha256=sha256_file(seal_path),
        expected_physical_change_report_sha256=physical_hash,
        expected_reference_manifest_sha256="r" * 64,
        expected_acceptance_criteria_sha256="c" * 64,
    )
    assert packet["pass"] is True

    (source / "shield.step").write_bytes(b"changed")
    drift = validate_source_cad_packet(
        source,
        audit_path,
        seal_path,
        physical_path,
        expected_source_manifest_sha256=sha256_file(manifest_path),
        expected_source_audit_sha256=audit_hash,
        expected_source_audit_seal_sha256=sha256_file(seal_path),
        expected_physical_change_report_sha256=physical_hash,
        expected_reference_manifest_sha256="r" * 64,
        expected_acceptance_criteria_sha256="c" * 64,
    )
    assert drift["pass"] is False
    assert drift["gates"]["source_artifacts_match_manifest"] is False
