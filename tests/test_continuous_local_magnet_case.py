from __future__ import annotations

import copy
import hashlib
import json

import pytest

from parastell.coil_filament_inventory import (
    build_coil_filament_inventory,
    write_inventory_create_only,
)
from parastell.continuous_local_magnet_case import (
    CASE_SCHEMA,
    MAP_SCHEMA,
    resolve_continuous_local_magnet_case,
)
from parastell.continuous_radial_contract import COMPONENT_ORDER
from parastell.material_identity import read_fusion_material_db


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _bind(path):
    return {"path": str(path), "sha256": _sha(path)}


def _source_manifest(coil_sha):
    stages = []
    previous = "0" * 64
    for index, component in enumerate(COMPONENT_ORDER[1:]):
        outer = f"{index + 1:x}" * 64
        stages.append(
            {
                "stage_index": index,
                "component": component,
                "cumulative_inner_matrix_sha256": previous,
                "layer_matrix_sha256": "a" * 64,
                "cumulative_outer_matrix_sha256": outer,
            }
        )
        previous = outer
    return {
        "schema": "parastell.parametric_source_cad/v1.0.0",
        "status": "SOURCE_CAD_BUILT_GEOMETRY_GATES_PENDING",
        "transport_eligible": False,
        "plan": {
            "device": "WISTELL-D",
            "extent_degrees": 90.0,
            "field_period_degrees": 90.0,
            "grid_shape": [80, 90],
            "magnet_representation": "radial_envelope",
            "radial_build_mode": "isolated_cumulative_shells",
            "component_order": list(COMPONENT_ORDER),
            "inputs": {"coils": {"sha256": coil_sha}},
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
        "actual_invessel_component_order": list(COMPONENT_ORDER),
        "radial_build_mode": "isolated_cumulative_shells",
        "magnet_inventory": {
            "representation": "radial_envelope",
            "material": "homogenized_magnet",
        },
        "component_evidence": {
            role: {"brep_valid": True, "solid_count": 1, "volume_cm3": 1.0}
            for role in COMPONENT_ORDER
        },
        "radial_build_construction_stages": stages,
        "resolved_layers": {
            role: {
                "shape": [80, 90],
                "minimum_cm": 1.0,
                "finite": True,
                "strictly_positive": True,
                "poloidally_closed": True,
            }
            for role in COMPONENT_ORDER[1:]
        },
    }


def _fixture(tmp_path):
    coil = tmp_path / "coils.test"
    coil.write_text(
        "title\nperiods 4\nbegin\n"
        "1 0 0 1\n0 1 0 1\n-1 0 0 1\n0 -1 0 0\nend\n",
        encoding="utf-8",
    )
    inventory = build_coil_filament_inventory(
        coil, start_line=3, scale_to_cm=100.0, sector_extent_degrees=90.0
    )
    inventory_path = tmp_path / "coil-inventory.json"
    write_inventory_create_only(inventory_path, inventory)
    selected = inventory["filaments"][0]
    source_path = _write(
        tmp_path / "source.json", _source_manifest(_sha(coil))
    )
    h5m = tmp_path / "dagmc.h5m"
    h5m.write_bytes(b"continuous accepted h5m")
    surface = {
        "schema": "parastell.continuous_magnet_surface_manifest/v1.0.0",
        "status": "PASS",
        "dagmc_sha256": _sha(h5m),
        "geometry": {
            "volume_count": 9,
            "continuous_magnet_volume_id": 9,
            "explicit_swept_coils": False,
        },
        "capture_bank": {
            "surface_ids": [10, 11, 12],
            "surface_count": 3,
            "scope": "complete_closed_continuous_magnet_volume_boundary",
            "records_all_entering_and_leaving_crossings": True,
            "closed": True,
        },
        "incoming_replay_interface": {
            "coupling_interface": "continuous_magnet_layer_inner_boundary",
            "surface_ids": [10],
            "surface_count": 1,
            "adjacency": {"from_volume_id": 8, "to_volume_id": 9},
            "closed_manifold_claim": False,
        },
        "physical_h5m_mutation": False,
    }
    surface_path = _write(tmp_path / "surface.json", surface)
    root = {
        "schema": "parastell.root_accepted_continuous_geometry/v1.0.0",
        "acceptance_authority": "ROOT_GEOMETRY_GATE_ACCEPTED",
        "dagmc_sha256": _sha(h5m),
        "source_manifest_sha256": _sha(source_path),
        "surface_manifest_sha256": _sha(surface_path),
        "canonical_geometry_fingerprint": "c" * 64,
        "global_magnet_representation": "continuous_radial_envelope",
    }
    root_path = _write(tmp_path / "root.json", root)
    bank = tmp_path / "bank.h5"
    bank.write_bytes(b"surface phase space")
    phase_path = _write(
        tmp_path / "phase.json",
        {
            "schema": "parastell.openmc16_surface_phase_space/v1.0.0",
            "openmc_surface_bank_fields_required": [
                "r",
                "u",
                "E",
                "time",
                "wgt",
                "delayed_group",
                "surf_id",
                "particle",
            ],
            "source_histories": 1000,
            "requested_surface_ids": [10, 11, 12],
            "source_files": [{"path": str(bank), "sha256": _sha(bank)}],
            "normalization": {
                "canonical_weight": "openmc_weight / source_histories",
                "tally_conditioning": False,
                "sampling_or_resampling": False,
            },
            "raw_phase_space_pass": True,
            "native_field_limitations": {
                "parent_history_id": "not stored by OpenMC 0.16"
            },
        },
    )
    mapping = {
        "schema": MAP_SCHEMA,
        "status": "QUALIFIED_FOR_BOUNDED_REPLAY",
        "global_source": {
            "dagmc_sha256": _sha(h5m),
            "surface_manifest_sha256": _sha(surface_path),
            "bank_sha256": _sha(bank),
            "coupling_interface": "continuous_magnet_layer_inner_boundary",
            "surface_ids": [10],
        },
        "local_target": {
            "coil_inventory_sha256": _sha(inventory_path),
            "filament_id": selected["filament_id"],
            "coordinates_sha256": selected["coordinates_sha256"],
            "receiving_interface": "outer_casing_external",
        },
        "station_selection": {
            "metric": "heating",
            "global_position_cm": [100.0, 0.0, 0.0],
            "arc_length_cm": 5.0,
            "periodic_tiling_applied": False,
        },
        "transform": {
            "rotation_global_to_local": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "translation_cm": [0.0, 0.0, 0.0],
        },
        "phase_space_contract": {
            "position_mapping": "declared_spatial_map",
            "direction_mapping": "proper_rotation",
            "preserved_fields": [
                "particle_pdg",
                "energy_eV",
                "time_s",
                "openmc_weight",
                "weight_per_source_history",
                "delayed_group",
                "source_file_index",
                "source_record_index",
            ],
            "parent_history_id_available": False,
            "canonical_weights_reconditioned": False,
        },
    }
    map_path = _write(tmp_path / "map.json", mapping)
    material_db = _write(
        tmp_path / "materials.json",
        {
            "case": {
                "density": 8.0,
                "comp": {"Fe56": 1.0},
                "metadata": {"name": "case", "citation": "test"},
            },
            "winding": {
                "density": 6.0,
                "comp": {"Cu63": 1.0},
                "metadata": {"name": "winding", "citation": "test"},
            },
        },
    )
    materials = read_fusion_material_db(material_db, temperature_K=20.0)
    case = {
        "schema": CASE_SCHEMA,
        "case_id": "local-test",
        "global_transport_geometry": {
            "source_manifest": _bind(source_path),
            "surface_manifest": _bind(surface_path),
            "dagmc_h5m": _bind(h5m),
            "root_acceptance_receipt": _bind(root_path),
        },
        "local_magnet_geometry": {
            "coil_filament_inventory": _bind(inventory_path),
            "filament_id": selected["filament_id"],
            "filament_coordinates_sha256": selected["coordinates_sha256"],
            "outer_cross_section_cm": {"width": 40.0, "thickness": 50.0},
            "case_thickness_cm": 2.0,
            "sample_mod": 1,
            "dimension_authority": "user_supplied_local_model",
        },
        "components": [
            {
                "component_id": "case",
                "role": "outer_casing",
                "material_id": "case",
            },
            {
                "component_id": "winding",
                "role": "winding_pack",
                "material_id": "winding",
            },
        ],
        "materials": materials,
        "surface_source": {
            "bank": _bind(bank),
            "phase_space_manifest": _bind(phase_path),
            "source_histories": 1000,
            "spatial_coupling_map": _bind(map_path),
        },
    }
    case_path = _write(tmp_path / "case.json", case)
    return case_path, case, mapping


def test_resolves_continuous_global_and_separate_local_filament(tmp_path):
    case_path, _, _ = _fixture(tmp_path)
    plan = resolve_continuous_local_magnet_case(case_path)
    assert (
        plan["global_transport_geometry"]["representation"]
        == "continuous_radial_envelope"
    )
    assert plan["local_geometry_request"]["outer_width_cm"] == 40.0
    assert (
        plan["local_geometry_request"]["global_volume_parity_claim"] is False
    )
    assert plan["surface_source"]["transform"]["right_handed"] is True
    assert plan["claims"]["global_swept_coils"] is False


def test_rejects_swept_global_even_when_case_is_rehashed(tmp_path):
    case_path, case, _ = _fixture(tmp_path)
    source_path = tmp_path / "source.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["plan"]["magnet_representation"] = "swept_filaments"
    _write(source_path, source)
    case["global_transport_geometry"]["source_manifest"] = _bind(source_path)
    _write(case_path, case)
    with pytest.raises(ValueError, match="radial envelope"):
        resolve_continuous_local_magnet_case(case_path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda mapping: mapping["station_selection"].update(
                metric="crossing_count"
            ),
            "coupling map",
        ),
        (
            lambda mapping: mapping["transform"].update(
                rotation_global_to_local=[[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
            ),
            "right-handed",
        ),
        (
            lambda mapping: mapping["global_source"].update(surface_ids=[999]),
            "coupling map",
        ),
        (
            lambda mapping: mapping["phase_space_contract"].update(
                canonical_weights_reconditioned=True
            ),
            "coupling map",
        ),
        (
            lambda mapping: mapping["phase_space_contract"].update(
                parent_history_id_available=True
            ),
            "coupling map",
        ),
    ],
)
def test_rejects_unqualified_or_cross_bound_spatial_map(
    tmp_path, mutation, match
):
    case_path, case, mapping = _fixture(tmp_path)
    mutation(mapping)
    map_path = _write(tmp_path / "map.json", mapping)
    case["surface_source"]["spatial_coupling_map"] = _bind(map_path)
    _write(case_path, case)
    with pytest.raises(ValueError, match=match):
        resolve_continuous_local_magnet_case(case_path)


def test_rejects_local_dimension_authority_drift(tmp_path):
    case_path, case, _ = _fixture(tmp_path)
    changed = copy.deepcopy(case)
    changed["local_magnet_geometry"]["outer_cross_section_cm"] = {
        "width": 30.0,
        "thickness": 30.0,
    }
    changed["local_magnet_geometry"][
        "dimension_authority"
    ] = "inherited_global_layer"
    _write(case_path, changed)
    with pytest.raises(ValueError, match="dimensions"):
        resolve_continuous_local_magnet_case(case_path)


def test_rejects_selected_filament_identity_drift(tmp_path):
    case_path, case, _ = _fixture(tmp_path)
    case["local_magnet_geometry"]["filament_coordinates_sha256"] = "0" * 64
    _write(case_path, case)
    with pytest.raises(ValueError, match="filament identity"):
        resolve_continuous_local_magnet_case(case_path)


def test_rejects_reconditioned_phase_space_manifest(tmp_path):
    case_path, case, _ = _fixture(tmp_path)
    phase_path = tmp_path / "phase.json"
    phase = json.loads(phase_path.read_text(encoding="utf-8"))
    phase["normalization"]["tally_conditioning"] = True
    _write(phase_path, phase)
    case["surface_source"]["phase_space_manifest"] = _bind(phase_path)
    _write(case_path, case)
    with pytest.raises(ValueError, match="phase-space manifest"):
        resolve_continuous_local_magnet_case(case_path)
