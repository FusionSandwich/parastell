"""Create the geometry-runtime half of a split OpenMC 0.16 model build.

The qualified ParaStell/PyMOAB runtime and the qualified OpenMC Python runtime
use different Python minor versions on Bateman.  This module deliberately does
all native MOAB inspection without importing OpenMC, then writes a hash-bound
seal that an OpenMC-only process can consume later.
"""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping
import xml.etree.ElementTree as ET

import numpy as np

from .parametric_openmc16_model import MATERIAL_NAMES
from .parametric_openmc16_model import _canonical_sha
from .parametric_openmc16_model import _load_control
from .parametric_openmc16_model import _maximum_vertex_radius
from .parametric_openmc16_model import _nuclear_data_manifest
from .parametric_openmc16_model import _validate_evidence
from .reference_geometry import native_dagmc_id_inventory, sha256_file
from .source_domain import _source_arrays, audit_source_tetrahedra_arrays


SEAL_SCHEMA = "parastell.parametric_openmc16_geometry_transport_seal/v1.0.0"
STRENGTHS_FILENAME = "source_strengths.npy"
SEAL_FILENAME = "GEOMETRY_TRANSPORT_SEAL.json"


def _material_names_from_xml(path: Path) -> list[str]:
    root = ET.parse(path).getroot()
    names = [
        str(row.get("name", "")).strip() for row in root.findall("./material")
    ]
    folded = [name.casefold() for name in names]
    if any(not name for name in names) or len(folded) != len(set(folded)):
        raise ValueError("materials XML names must be nonempty and unique")
    expected = MATERIAL_NAMES | {"Vacuum"}
    if set(names) != expected:
        raise ValueError(
            "materials XML names do not match DAGMC material tags"
        )
    return sorted(names)


def _runtime_identity() -> dict[str, Any]:
    """Bind the exact Python/native MOAB modules that derived the seal."""
    from pymoab import core, types

    modules = {}
    for name, module in (("pymoab.core", core), ("pymoab.types", types)):
        path = Path(module.__file__).resolve(strict=True)
        modules[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    executable = Path(sys.executable).resolve(strict=True)
    return {
        "python_executable": str(executable),
        "python_executable_sha256": sha256_file(executable),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "native_modules": modules,
    }


def _canonical_strengths(strengths: Any) -> tuple[np.ndarray, bytes, str]:
    values = np.ascontiguousarray(
        np.asarray(strengths, dtype="<f8").reshape(-1)
    )
    if (
        not len(values)
        or np.any(~np.isfinite(values))
        or np.any(values < 0.0)
        or not np.any(values > 0.0)
    ):
        raise ValueError(
            "source strengths must be finite, nonnegative, and nonzero"
        )
    raw_sha256 = hashlib.sha256(values.tobytes()).hexdigest()
    stream = BytesIO()
    np.save(stream, values, allow_pickle=False)
    return values, stream.getvalue(), raw_sha256


def _input_hashes(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "path": str(path.resolve(strict=True)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in sorted(paths.items())
    }


def create_geometry_transport_seal(
    control_path: Path,
    expected_control_sha256: str,
    output: Path,
) -> dict[str, Any]:
    """Validate geometry inputs and write a create-only OpenMC handoff seal."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"create-only output exists: {output}")

    control_path = control_path.resolve(strict=True)
    control = _load_control(control_path, expected_control_sha256)
    control_binding = {
        "path": str(control_path),
        "sha256": expected_control_sha256,
        "size_bytes": control_path.stat().st_size,
    }
    paths = _validate_evidence(control)
    before = _input_hashes(paths)
    dagmc_hash = before["dagmc_h5m"]["sha256"]
    source_mesh_hash = before["source_mesh_h5m"]["sha256"]

    native_ids = native_dagmc_id_inventory(paths["dagmc_h5m"])
    if (
        native_ids.get("native_id_gate_pass") is not True
        or native_ids.get("raw_h5m_sha256") != dagmc_hash
        or not native_ids.get("surface_ids")
        or not native_ids.get("volume_ids")
    ):
        raise ValueError("native DAGMC ID inventory is not accepted")
    maximum_native_id = int(native_ids["maximum_native_id"])
    if maximum_native_id != max(
        [int(value) for value in native_ids["surface_ids"]]
        + [int(value) for value in native_ids["volume_ids"]]
    ):
        raise ValueError("maximum native DAGMC ID is inconsistent")
    expected_magnet_volume_ids = set(range(9, 27))
    if not expected_magnet_volume_ids.issubset(
        {int(value) for value in native_ids["volume_ids"]}
    ):
        raise ValueError(
            "native DAGMC inventory omits a declared magnet volume"
        )

    maximum_radius = float(_maximum_vertex_radius(paths["dagmc_h5m"]))
    wrapper_radius = float(control["external_vacuum_radius_cm"])
    if (
        not np.isfinite(maximum_radius)
        or maximum_radius <= 0.0
        or wrapper_radius <= maximum_radius
    ):
        raise ValueError("external vacuum sphere does not enclose the H5M")

    tetrahedra, tagged_volumes, tagged_strengths = _source_arrays(
        paths["source_mesh_h5m"]
    )
    source_audit = audit_source_tetrahedra_arrays(
        tetrahedra, tagged_volumes, tagged_strengths
    )
    if source_audit.get("tetrahedron_data_gate_pass") is not True:
        raise ValueError("source-mesh tetrahedron data are invalid")
    strengths, strengths_npy, strengths_raw_sha256 = _canonical_strengths(
        tagged_strengths
    )
    if not np.isclose(
        float(np.sum(strengths)),
        float(source_audit["total_source_strength_n_per_s"]),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            "source-strength total disagrees with tetrahedron audit"
        )

    materials = _material_names_from_xml(paths["materials_xml"])
    nuclear_data = _nuclear_data_manifest(
        paths["materials_xml"], paths["cross_sections_xml"]
    )
    runtime = _runtime_identity()
    after = _input_hashes(paths)
    if after != before:
        raise RuntimeError(
            "one or more geometry-seal inputs changed during audit"
        )
    if (
        sha256_file(control_path) != control_binding["sha256"]
        or control_path.stat().st_size != control_binding["size_bytes"]
    ):
        raise RuntimeError("build control changed during geometry-seal audit")

    strengths_file_sha256 = hashlib.sha256(strengths_npy).hexdigest()
    seal = {
        "schema": SEAL_SCHEMA,
        "status": "GEOMETRY_TRANSPORT_SEAL_PASS",
        "claim": "BOUNDED_SMOKE_MODEL_INPUT_ONLY",
        "control": control_binding,
        "inputs": before,
        "runtime": runtime,
        "geometry": {
            "dagmc_sha256": dagmc_hash,
            "modeled_extent_degrees": float(control["modeled_extent_degrees"]),
            "n_field_periods": int(control["n_field_periods"]),
            "native_id_inventory": native_ids,
            "first_available_wrapper_id": maximum_native_id + 1,
            "maximum_h5m_vertex_radius_cm": maximum_radius,
            "external_vacuum_radius_cm": wrapper_radius,
            "containment_margin_cm": wrapper_radius - maximum_radius,
        },
        "source": {
            "source_mesh_sha256": source_mesh_hash,
            "strengths_filename": STRENGTHS_FILENAME,
            "strengths_file_sha256": strengths_file_sha256,
            "strengths_raw_little_endian_f8_sha256": strengths_raw_sha256,
            "strength_dtype": "<f8",
            "strength_count": int(len(strengths)),
            "physical_rate_n_per_s_for_modeled_extent": float(
                np.sum(strengths)
            ),
            "tetrahedron_audit": source_audit,
            "volume_normalized": False,
        },
        "materials": {
            "names": materials,
            "nuclear_data_manifest": nuclear_data,
        },
        "run": dict(control["run"]),
        "magnet_cell_ids": {
            f"magnet-{index:04d}": index + 9 for index in range(18)
        },
        "inputs_immutable": True,
        "physical_h5m_mutation": False,
        "source_mesh_mutation": False,
        "openmc_model_export_status": "NOT_RUN",
        "openmc_geometry_debug_status": "NOT_RUN",
        "production_run_authorized": False,
    }
    seal["seal_content_sha256"] = _canonical_sha(seal)
    encoded_seal = (
        json.dumps(seal, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")

    output.mkdir(parents=True, exist_ok=False)
    with (output / STRENGTHS_FILENAME).open("xb") as stream:
        stream.write(strengths_npy)
    if sha256_file(output / STRENGTHS_FILENAME) != strengths_file_sha256:
        raise RuntimeError("written source-strength artifact hash mismatch")
    if _input_hashes(paths) != before:
        raise RuntimeError("one or more inputs changed while writing the seal")
    if (
        sha256_file(control_path) != control_binding["sha256"]
        or control_path.stat().st_size != control_binding["size_bytes"]
    ):
        raise RuntimeError("build control changed while writing the seal")
    with (output / SEAL_FILENAME).open("xb") as stream:
        stream.write(encoded_seal)
    return seal
