"""OpenMC-0.16-only consumer of a ParaStell geometry transport seal.

This split-runtime stage deliberately does not import or inspect CAD, MOAB, or
DAGMC Python APIs.  The geometry runtime has already sealed their conclusions;
this module revalidates every bound byte and constructs only the OpenMC model.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

SEAL_SCHEMA = "parastell.parametric_openmc16_geometry_transport_seal/v1.0.0"
RECEIPT_SCHEMA = "parastell.parametric_openmc16_sealed_model/v1.0.0"
SEAL_FILENAME = "GEOMETRY_TRANSPORT_SEAL.json"
STRENGTHS_FILENAME = "source_strengths.npy"
RECEIPT_FILENAME = "PARAMETRIC_OPENMC16_SEALED_MODEL_RECEIPT.json"
MODEL_FILENAME = "model.xml"
EXPECTED_INPUTS = {
    "dagmc_h5m",
    "source_mesh_h5m",
    "materials_xml",
    "cross_sections_xml",
    "geometry_gate_receipt",
    "dagmc_export_receipt",
    "premesh_topology",
    "source_domain_receipt",
    "h5m_writeback",
    "source_physics_manifest",
}
MATERIAL_NAMES = {
    "first_wall",
    "breeder",
    "back_wall",
    "high_temperature_shield",
    "vacuum_vessel",
    "low_temperature_shield",
    "magnets",
}
CONTINUOUS_MATERIAL_NAMES = (MATERIAL_NAMES - {"magnets"}) | {
    "homogenized_magnet"
}
FORBIDDEN_RUNTIME_IMPORTS = (
    "pymoab",
    "pydagmc",
    "cadquery",
    "gmsh",
)


def _statepoint_batches(run: Mapping[str, Any]) -> list[int]:
    batches = int(run["batches"])
    interval = run.get("statepoint_interval_batches")
    if interval is None:
        return [batches]
    if (
        isinstance(interval, bool)
        or not isinstance(interval, int)
        or interval <= 0
        or interval > batches
    ):
        raise ValueError("statepoint_interval_batches is invalid")
    result = list(range(interval, batches + 1, interval))
    if result[-1] != batches:
        result.append(batches)
    return result


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _bound_file(row: Mapping[str, Any], label: str) -> tuple[Path, dict]:
    if not isinstance(row, Mapping) or set(row) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise ValueError(f"{label} binding is incomplete")
    path = Path(str(row["path"]))
    if not path.is_absolute():
        raise ValueError(f"{label} path is not absolute")
    path = path.resolve(strict=True)
    digest = str(row["sha256"])
    size = row["size_bytes"]
    if (
        not _valid_sha256(digest)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or path.stat().st_size != size
        or _sha256(path) != digest
    ):
        raise ValueError(f"{label} binding hash or size mismatch")
    return path, {"path": str(path), "sha256": digest, "size_bytes": size}


def _validate_native_ids(
    geometry: Mapping[str, Any], dagmc_sha256: str
) -> dict[str, Any]:
    native = geometry.get("native_id_inventory")
    if not isinstance(native, Mapping):
        raise TypeError("native DAGMC ID inventory is absent")
    try:
        surface_ids = [int(value) for value in native["surface_ids"]]
        volume_ids = [int(value) for value in native["volume_ids"]]
        maximum = int(native["maximum_native_id"])
        first_wrapper = int(geometry["first_available_wrapper_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("native DAGMC ID inventory is malformed") from exc
    try:
        physical_volume_count = int(geometry.get("physical_volume_count", 26))
    except (TypeError, ValueError) as exc:
        raise ValueError("sealed physical volume count is invalid") from exc
    geometry_mode = str(geometry.get("geometry_mode", "swept_magnets_legacy"))
    if geometry_mode not in {
        "continuous_radial_envelope",
        "swept_magnets_legacy",
    } or physical_volume_count not in {9, 26}:
        raise ValueError("sealed geometry mode or volume count is invalid")
    if (
        native.get("native_id_gate_pass") is not True
        or native.get("raw_h5m_sha256") != dagmc_sha256
        or not surface_ids
        or volume_ids != list(range(1, physical_volume_count + 1))
        or len(surface_ids) != len(set(surface_ids))
        or any(value <= 0 for value in surface_ids)
        or maximum != max(surface_ids + volume_ids)
        or first_wrapper != maximum + 1
    ):
        raise ValueError(
            "native DAGMC IDs/counts or wrapper floor are invalid"
        )
    return {
        "surface_ids": surface_ids,
        "volume_ids": volume_ids,
        "surface_count": len(surface_ids),
        "volume_count": len(volume_ids),
        "physical_volume_count": physical_volume_count,
        "geometry_mode": geometry_mode,
        "maximum_native_id": maximum,
        "first_available_wrapper_id": first_wrapper,
    }


def _validate_clearance(geometry: Mapping[str, Any]) -> dict[str, float]:
    try:
        maximum_radius = float(geometry["maximum_h5m_vertex_radius_cm"])
        wrapper_radius = float(geometry["external_vacuum_radius_cm"])
        margin = float(geometry["containment_margin_cm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("wrapper clearance values are malformed") from exc
    expected_margin = wrapper_radius - maximum_radius
    if (
        not all(
            math.isfinite(value)
            for value in (maximum_radius, wrapper_radius, margin)
        )
        or maximum_radius <= 0.0
        or wrapper_radius <= maximum_radius
        or margin <= 0.0
        or not math.isclose(
            margin, expected_margin, rel_tol=0.0, abs_tol=1.0e-12
        )
    ):
        raise ValueError("sealed external wrapper clearance is invalid")
    return {
        "maximum_h5m_vertex_radius_cm": maximum_radius,
        "external_vacuum_radius_cm": wrapper_radius,
        "containment_margin_cm": margin,
    }


def _validate_strengths(
    seal_path: Path, source: Mapping[str, Any], source_mesh_sha256: str
) -> tuple[np.ndarray, Path, dict[str, Any]]:
    if (
        source.get("source_mesh_sha256") != source_mesh_sha256
        or source.get("strengths_filename") != STRENGTHS_FILENAME
        or source.get("strength_dtype") != "<f8"
        or source.get("volume_normalized") is not False
    ):
        raise ValueError("sealed source identity or semantics are invalid")
    path = (seal_path.parent / STRENGTHS_FILENAME).resolve(strict=True)
    if path.parent != seal_path.parent:
        raise ValueError("source-strength path escapes the seal directory")
    if _sha256(path) != source.get("strengths_file_sha256"):
        raise ValueError("source-strength file hash mismatch")
    values = np.load(path, allow_pickle=False)
    if (
        values.ndim != 1
        or values.dtype.str != "<f8"
        or not values.flags.c_contiguous
        or not len(values)
        or np.any(~np.isfinite(values))
        or np.any(values < 0.0)
        or not np.any(values > 0.0)
    ):
        raise ValueError("source-strength array is invalid")
    raw_hash = hashlib.sha256(values.tobytes(order="C")).hexdigest()
    total = float(np.sum(values, dtype=np.float64))
    audit = source.get("tetrahedron_audit")
    count = source.get("strength_count")
    sealed_total = source.get("physical_rate_n_per_s_for_modeled_extent")
    if (
        raw_hash != source.get("strengths_raw_little_endian_f8_sha256")
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count != len(values)
        or not isinstance(audit, Mapping)
        or audit.get("tetrahedron_data_gate_pass") is not True
        or audit.get("tetrahedron_count") != len(values)
        or audit.get("invalid_tetrahedron_count") != 0
        or audit.get("negative_source_strength_count") != 0
        or not isinstance(sealed_total, (int, float))
        or isinstance(sealed_total, bool)
        or float(sealed_total) != total
        or float(audit.get("total_source_strength_n_per_s", math.nan)) != total
    ):
        raise ValueError("source-strength count, raw hash, or sum is invalid")
    return (
        values,
        path,
        {
            "strength_count": len(values),
            "strengths_file_sha256": source["strengths_file_sha256"],
            "strengths_raw_little_endian_f8_sha256": raw_hash,
            "physical_rate_n_per_s_for_modeled_extent": total,
            "volume_normalized": False,
        },
    )


def _validate_nuclear_data(
    value: Mapping[str, Any], cross_sections_sha256: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not isinstance(value, Mapping):
        raise TypeError("nuclear-data manifest is absent")
    candidate = dict(value)
    claimed = candidate.pop("manifest_sha256", None)
    if (
        not _valid_sha256(claimed)
        or _canonical_sha(candidate) != claimed
        or candidate.get("cross_sections_xml_sha256") != cross_sections_sha256
    ):
        raise ValueError("nuclear-data manifest seal is invalid")
    nuclides = candidate.get("required_nuclides")
    libraries = candidate.get("libraries")
    if (
        not isinstance(nuclides, list)
        or not nuclides
        or len(nuclides) != len(set(nuclides))
        or any(not str(value).strip() for value in nuclides)
        or not isinstance(libraries, list)
        or not libraries
    ):
        raise ValueError("nuclear-data inventory is incomplete")
    bound = {}
    identities = set()
    for index, row in enumerate(libraries):
        if not isinstance(row, Mapping) or set(row) != {
            "type",
            "material",
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("nuclear-data library binding is malformed")
        identity = (str(row["type"]), str(row["material"]))
        if identity[0] not in {"neutron", "photon"} or identity in identities:
            raise ValueError("nuclear-data library identity is invalid")
        path, evidence = _bound_file(
            {
                "path": row["path"],
                "sha256": row["sha256"],
                "size_bytes": row["size_bytes"],
            },
            f"nuclear-data library {index}",
        )
        identities.add(identity)
        bound[str(path)] = evidence
    elements = set()
    for nuclide in nuclides:
        match = re.match(r"^([A-Z][a-z]?)", str(nuclide))
        if match is None:
            raise ValueError("nuclear-data nuclide name is invalid")
        elements.add(match.group(1))
    expected_identities = {
        *(("neutron", str(nuclide)) for nuclide in nuclides),
        *(("photon", element) for element in elements),
    }
    if identities != expected_identities:
        raise ValueError("neutron/photoatomic library coverage is incomplete")
    return dict(value), bound


def _validate_seal(
    seal_path: Path, expected_seal_sha256: str
) -> tuple[dict[str, Any], dict[str, Path], np.ndarray, dict[str, Any], dict]:
    seal_path = seal_path.resolve(strict=True)
    if (
        seal_path.name != SEAL_FILENAME
        or _sha256(seal_path) != expected_seal_sha256
    ):
        raise ValueError("geometry transport seal file hash or name mismatch")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    candidate = dict(seal)
    claimed = candidate.pop("seal_content_sha256", None)
    if (
        seal.get("schema") != SEAL_SCHEMA
        or seal.get("status") != "GEOMETRY_TRANSPORT_SEAL_PASS"
        or seal.get("claim") != "BOUNDED_SMOKE_MODEL_INPUT_ONLY"
        or not _valid_sha256(claimed)
        or _canonical_sha(candidate) != claimed
        or seal.get("inputs_immutable") is not True
        or seal.get("physical_h5m_mutation") is not False
        or seal.get("source_mesh_mutation") is not False
        or seal.get("openmc_model_export_status") != "NOT_RUN"
        or seal.get("openmc_geometry_debug_status") != "NOT_RUN"
        or seal.get("production_run_authorized") is not False
    ):
        raise ValueError("geometry transport seal is not accepted")
    control_path, control_binding = _bound_file(
        seal.get("control", {}), "control"
    )
    inputs = seal.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != EXPECTED_INPUTS:
        raise ValueError("sealed input inventory is incomplete")
    paths = {}
    evidence = {}
    for name in sorted(EXPECTED_INPUTS):
        path, row = _bound_file(inputs[name], name)
        paths[name], evidence[name] = path, row
    geometry = seal.get("geometry")
    if (
        not isinstance(geometry, Mapping)
        or geometry.get("dagmc_sha256") != evidence["dagmc_h5m"]["sha256"]
        or geometry.get("modeled_extent_degrees") != 90.0
        or geometry.get("n_field_periods") != 4
    ):
        raise ValueError("sealed direct-period geometry identity is invalid")
    ids = _validate_native_ids(geometry, evidence["dagmc_h5m"]["sha256"])
    clearance = _validate_clearance(geometry)
    expected_magnets = (
        {"continuous-magnet-layer": 9}
        if geometry.get("geometry_mode") == "continuous_radial_envelope"
        else {f"magnet-{index:04d}": index + 9 for index in range(18)}
    )
    if seal.get("magnet_cell_ids") != expected_magnets:
        raise ValueError("sealed magnet cell IDs are invalid")
    expected_components = {
        "chamber": 1,
        "first_wall": 2,
        "breeder": 3,
        "back_wall": 4,
        "high_temperature_shield": 5,
        "vacuum_vessel": 6,
        "low_temperature_shield": 7,
        "vacuum_gap": 8,
    }
    if geometry.get("geometry_mode") == "continuous_radial_envelope":
        expected_components["magnets"] = 9
    declared_components = geometry.get("component_cell_ids")
    if (
        declared_components is not None
        and declared_components != expected_components
    ) or (
        geometry.get("geometry_mode") == "continuous_radial_envelope"
        and declared_components is None
    ):
        raise ValueError("sealed component cell IDs are invalid")
    source_values, strengths_path, source = _validate_strengths(
        seal_path,
        seal.get("source", {}),
        evidence["source_mesh_h5m"]["sha256"],
    )
    materials = seal.get("materials")
    expected_names = sorted(
        CONTINUOUS_MATERIAL_NAMES
        if geometry.get("geometry_mode") == "continuous_radial_envelope"
        else MATERIAL_NAMES
    )
    if (
        not isinstance(materials, Mapping)
        or materials.get("names") != expected_names
    ):
        raise ValueError("sealed material names are invalid")
    nuclear_data, libraries = _validate_nuclear_data(
        materials.get("nuclear_data_manifest", {}),
        evidence["cross_sections_xml"]["sha256"],
    )
    run = seal.get("run")
    if not isinstance(run, Mapping) or set(run) not in (
        {"particles", "batches", "seed"},
        {"particles", "batches", "seed", "statepoint_interval_batches"},
    ):
        raise ValueError("sealed run controls are invalid")
    if any(
        isinstance(run.get(key), bool)
        or not isinstance(run.get(key), int)
        or run.get(key) <= 0
        for key in run
    ):
        raise ValueError("sealed run controls are invalid")
    _statepoint_batches(run)
    source_physics = json.loads(
        paths["source_physics_manifest"].read_text(encoding="utf-8")
    )
    if (
        source_physics.get("schema")
        != "parastell.source_physics_manifest/v1.0.0"
        or source_physics.get("particle") != "neutron"
        or source_physics.get("angle") != "isotropic"
        or source_physics.get("energy_law") != "discrete"
        or source_physics.get("energy_eV") != [14_100_000.0]
        or source_physics.get("probability") != [1.0]
        or source_physics.get("claim") != "BOUNDED_SMOKE_ONLY"
    ):
        raise ValueError("sealed source-physics law is invalid")
    all_bindings = {
        "seal": {
            "path": str(seal_path),
            "sha256": expected_seal_sha256,
            "size_bytes": seal_path.stat().st_size,
        },
        "control": control_binding,
        "source_strengths": {
            "path": str(strengths_path),
            "sha256": source["strengths_file_sha256"],
            "size_bytes": strengths_path.stat().st_size,
        },
        **evidence,
        **{f"nuclear_data:{path}": row for path, row in libraries.items()},
    }
    summary = {
        "geometry": {**ids, **clearance},
        "source": source,
        "nuclear_data_manifest": nuclear_data,
        "material_names": expected_names,
        "source_physics": source_physics,
        "run": dict(run),
        "magnet_cell_ids": expected_magnets,
        "component_cell_ids": expected_components,
        "control_path": str(control_path),
    }
    return seal, paths, source_values, all_bindings, summary


def _revalidate_bindings(bindings: Mapping[str, Mapping[str, Any]]) -> None:
    for label, row in bindings.items():
        _bound_file(row, label)


def export_sealed_openmc16_model(
    seal_path: str | Path,
    expected_seal_sha256: str,
    output: str | Path,
):
    """Export model XML from a valid geometry seal without native CAD APIs."""
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"create-only output exists: {output_path}")
    if not _valid_sha256(expected_seal_sha256):
        raise ValueError("expected seal SHA-256 is invalid")
    seal, paths, strengths, bindings, summary = _validate_seal(
        Path(seal_path), expected_seal_sha256
    )
    _revalidate_bindings(bindings)

    import openmc

    if openmc.__version__ != "0.16.0":
        raise RuntimeError(f"OpenMC 0.16.0 required, got {openmc.__version__}")
    first = summary["geometry"]["first_available_wrapper_id"]
    extent_radians = math.radians(90.0)
    start = openmc.YPlane(surface_id=first, boundary_type="periodic")
    end = openmc.Plane(
        surface_id=first + 1,
        a=math.sin(extent_radians),
        b=-math.cos(extent_radians),
        c=0.0,
        d=0.0,
        boundary_type="periodic",
    )
    start.periodic_surface = end
    vacuum = openmc.Sphere(
        surface_id=first + 2,
        r=summary["geometry"]["external_vacuum_radius_cm"],
        boundary_type="vacuum",
    )
    dagmc = openmc.DAGMCUniverse(
        filename=str(paths["dagmc_h5m"]),
        universe_id=first + 5,
        auto_geom_ids=False,
    )
    region = -vacuum & +start & +end
    wrapper_cell = openmc.Cell(cell_id=first + 3, fill=dagmc, region=region)
    root = openmc.Universe(universe_id=first + 4, cells=[wrapper_cell])
    geometry = openmc.Geometry(root)

    materials = openmc.Materials.from_xml(path=paths["materials_xml"])
    observed_names = sorted(str(row.name).strip() for row in materials)
    if observed_names != summary["material_names"]:
        raise ValueError(
            "OpenMC material names differ from the sealed inventory"
        )
    materials.cross_sections = str(paths["cross_sections_xml"])
    mesh = openmc.UnstructuredMesh(str(paths["source_mesh_h5m"]), "moab")
    law = summary["source_physics"]
    source = openmc.IndependentSource(
        space=openmc.stats.MeshSpatial(
            mesh, strengths=strengths.tolist(), volume_normalized=False
        ),
        angle=openmc.stats.Isotropic(),
        energy=openmc.stats.Discrete(law["energy_eV"], law["probability"]),
        particle="neutron",
    )
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.particles = summary["run"]["particles"]
    settings.batches = summary["run"]["batches"]
    settings.seed = summary["run"]["seed"]
    statepoint_batches = _statepoint_batches(summary["run"])
    settings.statepoint = {"batches": statepoint_batches}
    settings.source = source
    settings.photon_transport = True
    model = openmc.Model(
        geometry=geometry, materials=materials, settings=settings
    )

    output_path.mkdir(parents=True, exist_ok=False)
    model_path = output_path / MODEL_FILENAME
    model.export_to_model_xml(model_path)
    if not model_path.is_file() or model_path.stat().st_size <= 0:
        raise RuntimeError("OpenMC model XML was not created")
    _revalidate_bindings(bindings)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "MODEL_EXPORTED_TRANSPORT_PENDING",
        "claim": "BOUNDED_SMOKE_ONLY",
        "run_mode": "fixed source",
        "openmc_runtime": {
            "version": openmc.__version__,
            "python_version": platform.python_version(),
            "python_executable": str(Path(sys.executable).resolve()),
        },
        "geometry_transport_seal": bindings["seal"],
        "geometry": {
            "dagmc_sha256": bindings["dagmc_h5m"]["sha256"],
            "modeled_extent_degrees": 90.0,
            "n_field_periods": 4,
            **summary["geometry"],
        },
        "transport_periodic_wrapper": {
            "start_surface_id": first,
            "end_surface_id": first + 1,
            "vacuum_surface_id": first + 2,
            "wrapper_cell_id": first + 3,
            "root_universe_id": first + 4,
            "dagmc_universe_id": first + 5,
            "periodic_planes_paired": True,
            "period_degrees": 90.0,
            "auto_geom_ids": False,
        },
        "source": summary["source"],
        "materials": {
            "names": summary["material_names"],
            "nuclear_data_manifest": summary["nuclear_data_manifest"],
        },
        "run": summary["run"],
        "statepoint_policy": {
            "explicit": True,
            "batches": statepoint_batches,
            "interval_batches": summary["run"].get(
                "statepoint_interval_batches"
            ),
            "final_batch_included": statepoint_batches[-1]
            == summary["run"]["batches"],
        },
        "magnet_cell_ids": summary["magnet_cell_ids"],
        "component_cell_ids": summary["component_cell_ids"],
        "validated_input_bindings": {
            key: bindings[key] for key in sorted(bindings)
        },
        "model_xml": {
            "path": str(model_path),
            "sha256": _sha256(model_path),
            "size_bytes": model_path.stat().st_size,
        },
        "runtime_dependency_boundary": {
            "required": ["openmc==0.16.0", "numpy"],
            "forbidden": list(FORBIDDEN_RUNTIME_IMPORTS),
        },
        "photon_transport": True,
        "prompt_photons": "transported when produced by nuclear data",
        "delayed_photons": "separate activation-derived source",
        "all_bound_inputs_immutable": True,
        "physical_h5m_mutation": False,
        "source_mesh_mutation": False,
        "transport_executed": False,
        "production_run_authorized": False,
    }
    receipt["receipt_content_sha256"] = _canonical_sha(receipt)
    receipt_path = output_path / RECEIPT_FILENAME
    with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    _revalidate_bindings(bindings)
    if seal.get("seal_content_sha256") != json.loads(
        Path(seal_path).read_text(encoding="utf-8")
    ).get("seal_content_sha256"):
        raise RuntimeError("geometry transport seal changed during export")
    return model, receipt
