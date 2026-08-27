"""Immutable reference-geometry support for ParaStell radiation models.

This module deliberately treats an accepted DAGMC file as an input.  It can
inventory and fingerprint the file, discover the original homogenized magnet
volumes, prove that their faceted outer boundaries close, and create an OpenMC
``DAGMCUniverse`` without editing the H5M.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REFERENCE_SCHEMA = "parastell.reference_geometry/v1.0.0"
PARITY_SCHEMA = "parastell.reference_geometry_parity/v1.0.0"
FIGURE_MANIFEST_SCHEMA = "parastell.geometry_figure_manifest/v1.0.0"
GEOMETRY_MODE = "reuse_reference_h5m"
DEFAULT_COORDINATE_QUANTUM_CM = 1.0e-6


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _triangles(values: Any) -> np.ndarray:
    coordinates = np.asarray(values, dtype=float)
    if coordinates.ndim == 3 and coordinates.shape[1:] == (3, 3):
        result = coordinates
    elif coordinates.ndim == 2 and coordinates.shape[1] == 3:
        if len(coordinates) % 3:
            raise ValueError("triangle coordinate rows are not divisible by 3")
        result = coordinates.reshape((-1, 3, 3))
    else:
        raise ValueError(
            "triangle coordinates must have shape (n,3) or (n,3,3)"
        )
    if len(result) == 0 or not np.all(np.isfinite(result)):
        raise ValueError("triangle coordinates must be finite and nonempty")
    return result


def _quantize(values: Any, quantum_cm: float) -> np.ndarray:
    quantum = float(quantum_cm)
    if not np.isfinite(quantum) or quantum <= 0.0:
        raise ValueError("coordinate quantum must be positive and finite")
    coordinates = np.asarray(values, dtype=float)
    scaled = coordinates / quantum
    limit = np.nextafter(float(np.iinfo(np.int64).max), 0.0)
    if not np.all(np.isfinite(scaled)) or np.any(np.abs(scaled) > limit):
        raise ValueError("coordinates exceed the canonical integer grid")
    return np.rint(scaled).astype(np.int64)


def _volume_id(value: Any) -> int | None:
    return None if value is None else int(value.id)


def _outward_sign(surface: Any, volume_id: int) -> int:
    """Map the native DAGMC facet normal to the volume-outward normal."""
    forward = _volume_id(getattr(surface, "forward_volume", None))
    reverse = _volume_id(getattr(surface, "reverse_volume", None))
    if forward == int(volume_id):
        return 1
    if reverse == int(volume_id):
        return -1
    raise ValueError(
        f"surface {surface.id} has no forward/reverse sense for volume {volume_id}"
    )


def _material(value: Any) -> str:
    return str(value or "").strip()


def _component_name(volume: Any) -> str:
    return str(
        getattr(volume, "component_id", "")
        or getattr(volume, "name", "")
        or ""
    ).strip()


def _surface_geometry_signature(surface: Any, quantum_cm: float) -> str:
    quantized = _quantize(_triangles(surface.triangle_coords), quantum_cm)
    canonical = []
    for triangle in quantized:
        vertices = sorted(tuple(int(item) for item in row) for row in triangle)
        canonical.append(vertices)
    return _json_hash(sorted(canonical))


def _surface_metrics(surface: Any) -> dict[str, Any]:
    triangles = _triangles(surface.triangle_coords)
    area_vectors = 0.5 * np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    areas = np.linalg.norm(area_vectors, axis=1)
    area = float(areas.sum())
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError(f"surface {surface.id} has nonpositive area")
    centroid = np.sum(triangles.mean(axis=1) * areas[:, None], axis=0) / area
    return {
        "area_cm2": area,
        "centroid_cm": [float(item) for item in centroid],
        "triangle_count": int(len(triangles)),
    }


def audit_volume_envelope(
    volume: Any,
    *,
    coordinate_quantum_cm: float = DEFAULT_COORDINATE_QUANTUM_CM,
    vector_area_relative_tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    """Audit the complete, topology-sensed outer boundary of one volume."""
    edge_counts: dict[tuple[Any, Any], int] = {}
    vector_area = np.zeros(3)
    total_area = 0.0
    signed_six_volume = 0.0
    surface_rows = []
    all_vertices = []
    volume_id = int(volume.id)
    for surface in sorted(volume.surfaces, key=lambda item: int(item.id)):
        triangles = _triangles(surface.triangle_coords)
        sign = _outward_sign(surface, volume_id)
        quantized = _quantize(triangles, coordinate_quantum_cm)
        for triangle in quantized:
            vertices = [tuple(int(item) for item in row) for row in triangle]
            for first, second in ((0, 1), (1, 2), (2, 0)):
                edge = tuple(sorted((vertices[first], vertices[second])))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        all_vertices.append(triangles.reshape((-1, 3)))
        native_area_vectors = 0.5 * np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        areas = np.linalg.norm(native_area_vectors, axis=1)
        surface_area = float(areas.sum())
        total_area += surface_area
        vector_area += sign * native_area_vectors.sum(axis=0)
        signed_six_volume += sign * float(
            np.sum(
                triangles[:, 0] * np.cross(triangles[:, 1], triangles[:, 2])
            )
        )
        surface_rows.append(
            {
                "surface_id": int(surface.id),
                "forward_volume_id": _volume_id(
                    getattr(surface, "forward_volume", None)
                ),
                "reverse_volume_id": _volume_id(
                    getattr(surface, "reverse_volume", None)
                ),
                "outward_normal_sign": sign,
                "area_cm2": surface_area,
                "triangle_count": int(len(triangles)),
            }
        )
    if not surface_rows:
        raise ValueError(f"volume {volume_id} has no boundary surfaces")
    multiplicity_histogram: dict[str, int] = {}
    for count in edge_counts.values():
        key = str(int(count))
        multiplicity_histogram[key] = multiplicity_histogram.get(key, 0) + 1
    errors = sum(count != 2 for count in edge_counts.values())
    relative_closure = float(np.linalg.norm(vector_area) / total_area)
    vertices = np.concatenate(all_vertices)
    return {
        "volume_id": volume_id,
        "material_tag": _material(getattr(volume, "material", "")),
        "component_name": _component_name(volume),
        "surface_ids": [row["surface_id"] for row in surface_rows],
        "surfaces": surface_rows,
        "edge_count": int(len(edge_counts)),
        "edge_multiplicity_histogram": multiplicity_histogram,
        "edge_multiplicity_error_count": int(errors),
        "edge_closure_pass": errors == 0,
        "vector_area_cm2": [float(item) for item in vector_area],
        "vector_area_closure_relative": relative_closure,
        "vector_area_closure_tolerance": float(vector_area_relative_tolerance),
        "vector_area_closure_pass": relative_closure
        <= float(vector_area_relative_tolerance),
        "volume_cm3": abs(signed_six_volume) / 6.0,
        "bounding_box_cm": {
            "lower": [float(item) for item in vertices.min(axis=0)],
            "upper": [float(item) for item in vertices.max(axis=0)],
        },
        "closed": errors == 0
        and relative_closure <= float(vector_area_relative_tolerance),
    }


def canonical_geometry_fingerprint(
    dagmc_path: str | Path,
    *,
    coordinate_quantum_cm: float = DEFAULT_COORDINATE_QUANTUM_CM,
    faceting_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an entity-order-independent faceted geometry fingerprint."""
    import pydagmc

    path = Path(dagmc_path).resolve()
    model = pydagmc.Model(str(path))
    surface_objects: dict[int, Any] = {}
    for volume in model.volumes_by_id.values():
        for surface in volume.surfaces:
            surface_objects[int(surface.id)] = surface

    surface_geometry = {
        surface_id: _surface_geometry_signature(surface, coordinate_quantum_cm)
        for surface_id, surface in surface_objects.items()
    }
    provisional_volumes: dict[int, dict[str, Any]] = {}
    provisional_signatures: dict[int, str] = {}
    for volume_id, volume in model.volumes_by_id.items():
        envelope = audit_volume_envelope(
            volume, coordinate_quantum_cm=coordinate_quantum_cm
        )
        row = {
            "material_tag": _material(getattr(volume, "material", "")),
            "component_name": _component_name(volume),
            "surface_geometry": sorted(
                surface_geometry[int(surface.id)]
                for surface in volume.surfaces
            ),
            "bounding_box_quantized": _quantize(
                [
                    envelope["bounding_box_cm"]["lower"],
                    envelope["bounding_box_cm"]["upper"],
                ],
                coordinate_quantum_cm,
            ).tolist(),
            "volume_quantized_cm3": int(
                round(envelope["volume_cm3"] / coordinate_quantum_cm**3)
            ),
        }
        provisional_volumes[int(volume_id)] = row
        provisional_signatures[int(volume_id)] = _json_hash(row)

    surfaces = []
    for surface_id, surface in surface_objects.items():
        forward = _volume_id(getattr(surface, "forward_volume", None))
        reverse = _volume_id(getattr(surface, "reverse_volume", None))
        surfaces.append(
            {
                "geometry": surface_geometry[surface_id],
                "forward_volume": (
                    provisional_signatures[forward]
                    if forward in provisional_signatures
                    else None
                ),
                "reverse_volume": (
                    provisional_signatures[reverse]
                    if reverse in provisional_signatures
                    else None
                ),
                **_surface_metrics(surface),
            }
        )
    canonical = {
        "canonicalization_version": 1,
        "coordinate_units": "cm",
        "coordinate_quantum_cm": float(coordinate_quantum_cm),
        "faceting_settings": dict(faceting_settings or {}),
        "volumes": sorted(
            provisional_volumes.values(),
            key=lambda row: json.dumps(row, sort_keys=True),
        ),
        "surfaces": sorted(
            surfaces, key=lambda row: json.dumps(row, sort_keys=True)
        ),
    }
    return {
        "algorithm": "sha256",
        "canonical_fingerprint": _json_hash(canonical),
        "raw_h5m_sha256": sha256_file(path),
        "coordinate_quantum_cm": float(coordinate_quantum_cm),
        "volume_count": len(provisional_volumes),
        "surface_count": len(surface_objects),
        "canonical_geometry": canonical,
    }


def geometry_inventory(
    dagmc_path: str | Path,
    *,
    coordinate_quantum_cm: float = DEFAULT_COORDINATE_QUANTUM_CM,
) -> dict[str, Any]:
    """Return the native volume/surface/material inventory for a DAGMC file."""
    import pydagmc

    path = Path(dagmc_path).resolve()
    model = pydagmc.Model(str(path))
    volumes = []
    surface_ids: set[int] = set()
    for volume_id, volume in sorted(model.volumes_by_id.items()):
        envelope = audit_volume_envelope(
            volume, coordinate_quantum_cm=coordinate_quantum_cm
        )
        surface_ids.update(envelope["surface_ids"])
        volumes.append(envelope)
    material_counts: dict[str, int] = {}
    for row in volumes:
        tag = row["material_tag"]
        material_counts[tag] = material_counts.get(tag, 0) + 1
    return {
        "raw_h5m_sha256": sha256_file(path),
        "coordinate_units": "cm",
        "volume_count": len(volumes),
        "surface_count": len(surface_ids),
        "material_counts": dict(sorted(material_counts.items())),
        "volumes_without_material": [
            int(volume.id)
            for volume in getattr(model, "volumes_without_material", ())
        ],
        "volumes": volumes,
    }


def discover_original_magnets(
    dagmc_path: str | Path,
    *,
    accepted_material_tags: Sequence[str] = ("magnets",),
    coordinate_quantum_cm: float = DEFAULT_COORDINATE_QUANTUM_CM,
) -> dict[str, Any]:
    """Discover and close the original one-solid ParaStell magnets."""
    inventory = geometry_inventory(
        dagmc_path, coordinate_quantum_cm=coordinate_quantum_cm
    )
    accepted = {str(item).strip().lower() for item in accepted_material_tags}
    rows = [
        row
        for row in inventory["volumes"]
        if row["material_tag"].strip().lower() in accepted
    ]

    def key(row: Mapping[str, Any]) -> tuple[float, ...]:
        lower = np.asarray(row["bounding_box_cm"]["lower"], dtype=float)
        upper = np.asarray(row["bounding_box_cm"]["upper"], dtype=float)
        center = (lower + upper) / 2.0
        angle = math.atan2(center[1], center[0]) % (2.0 * math.pi)
        return (angle, float(center[2]), float(center[0]), float(center[1]))

    rows.sort(key=key)
    magnets = []
    for index, row in enumerate(rows):
        magnets.append(
            {
                "magnet_id": f"magnet-{index:04d}",
                "volume_id": row["volume_id"],
                "material_tag": row["material_tag"],
                "component_name": row["component_name"],
                "outer_envelope_surface_ids": row["surface_ids"],
                "surface_count": len(row["surface_ids"]),
                "bounding_box_cm": row["bounding_box_cm"],
                "volume_cm3": row["volume_cm3"],
                "edge_multiplicity_error_count": row[
                    "edge_multiplicity_error_count"
                ],
                "vector_area_closure_relative": row[
                    "vector_area_closure_relative"
                ],
                "edge_closure_pass": row["edge_closure_pass"],
                "vector_area_closure_pass": row["vector_area_closure_pass"],
                "closed": row["closed"],
                "physical_role": "original_homogenized_magnet_solid",
                "casing_winding_split": False,
            }
        )
    return {
        "schema": "parastell.original_magnet_envelopes/v1.0.0",
        "raw_h5m_sha256": inventory["raw_h5m_sha256"],
        "accepted_material_tags": sorted(accepted),
        "magnet_count": len(magnets),
        "all_envelopes_close": bool(magnets)
        and all(item["closed"] for item in magnets),
        "magnets": magnets,
    }


@dataclass(frozen=True)
class ReferenceGeometry:
    """An immutable handle to an accepted H5M file."""

    path: Path
    accepted_sha256: str
    geometry_mode: str = GEOMETRY_MODE

    @classmethod
    def open(
        cls, path: str | Path, *, expected_sha256: str | None = None
    ) -> "ReferenceGeometry":
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        digest = sha256_file(resolved)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(
                f"reference H5M hash mismatch: expected {expected_sha256}, got {digest}"
            )
        return cls(resolved, digest)

    def verify_unchanged(self) -> bool:
        current = sha256_file(self.path)
        if current != self.accepted_sha256:
            raise RuntimeError(
                "accepted reference H5M was modified after instrumentation"
            )
        return True

    def dagmc_universe(self, **kwargs: Any) -> Any:
        """Create an OpenMC universe that references, but never edits, the H5M."""
        import openmc

        self.verify_unchanged()
        return openmc.DAGMCUniverse(filename=str(self.path), **kwargs)

    def openmc_geometry(
        self, *, external_vacuum_radius_cm: float | None = None
    ) -> Any:
        """Create OpenMC geometry, optionally with an external CSG vacuum sphere."""
        import openmc

        if external_vacuum_radius_cm is None:
            universe = self.dagmc_universe()
            return openmc.Geometry(universe)
        # The wrapper cell is OpenMC CSG and therefore has an ID in the same
        # namespace as DAGMC cells.  Remap DAGMC entity IDs only in OpenMC's
        # in-memory model to avoid a collision; the H5M remains byte-identical.
        universe = self.dagmc_universe(auto_geom_ids=True)
        radius = float(external_vacuum_radius_cm)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError(
                "external vacuum radius must be positive and finite"
            )
        boundary = openmc.Sphere(r=radius, boundary_type="vacuum")
        cell = openmc.Cell(fill=universe, region=-boundary)
        return openmc.Geometry([cell])

    def evidence(self) -> dict[str, Any]:
        self.verify_unchanged()
        stat = self.path.stat()
        return {
            "geometry_mode": self.geometry_mode,
            "path": str(self.path),
            "raw_h5m_sha256": self.accepted_sha256,
            "size_bytes": stat.st_size,
            "mtime_ns_at_inspection": stat.st_mtime_ns,
            "h5m_mutated": False,
        }


def compare_reference_geometry(
    reference_path: str | Path,
    instrumented_path: str | Path,
    *,
    coordinate_quantum_cm: float = DEFAULT_COORDINATE_QUANTUM_CM,
    faceting_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify reference/instrumented geometry parity without tolerating drift."""
    reference = canonical_geometry_fingerprint(
        reference_path,
        coordinate_quantum_cm=coordinate_quantum_cm,
        faceting_settings=faceting_settings,
    )
    instrumented = canonical_geometry_fingerprint(
        instrumented_path,
        coordinate_quantum_cm=coordinate_quantum_cm,
        faceting_settings=faceting_settings,
    )
    if reference["raw_h5m_sha256"] == instrumented["raw_h5m_sha256"]:
        classification = "BYTE_IDENTICAL_REUSE"
    elif (
        reference["canonical_fingerprint"]
        == instrumented["canonical_fingerprint"]
    ):
        classification = "CANONICAL_GEOMETRY_EQUIVALENT"
    else:
        classification = "FAIL_PHYSICAL_GEOMETRY_DIFFERENCE"
    return {
        "schema": PARITY_SCHEMA,
        "classification": classification,
        "pass": classification
        in {"BYTE_IDENTICAL_REUSE", "CANONICAL_GEOMETRY_EQUIVALENT"},
        "reference": {
            key: reference[key]
            for key in (
                "raw_h5m_sha256",
                "canonical_fingerprint",
                "volume_count",
                "surface_count",
            )
        },
        "instrumented": {
            key: instrumented[key]
            for key in (
                "raw_h5m_sha256",
                "canonical_fingerprint",
                "volume_count",
                "surface_count",
            )
        },
        "same_path": Path(reference_path).resolve()
        == Path(instrumented_path).resolve(),
        "h5m_mutation_performed": False,
        "extra_physical_h5m_volumes": max(
            0, instrumented["volume_count"] - reference["volume_count"]
        ),
    }


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_reference_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed on a missing or geometry-changing reference contract."""
    required = {
        "schema",
        "reference_identity_classification",
        "fork_main_sha",
        "upstream_main_sha",
        "inputs",
        "example_script",
        "environment",
        "raw_h5m_sha256",
        "canonical_geometry_fingerprint",
        "coordinate_units",
        "modeled_toroidal_extent_deg",
        "field_period_semantics",
        "volume_inventory",
        "surface_inventory",
        "material_component_tags",
        "magnet_ids",
        "magnet_representation",
        "source_mesh",
        "faceting_settings",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"reference manifest lacks {sorted(missing)}")
    if manifest["schema"] != REFERENCE_SCHEMA:
        raise ValueError("unsupported reference-geometry schema")
    if manifest["magnet_representation"] != "one_homogenized_solid_each":
        raise ValueError("reference magnets must remain homogenized solids")
    if manifest.get("ports_present") is not False:
        raise ValueError("reference manifest must explicitly prove no ports")
    if manifest.get("case_thickness_cm") != 0.0:
        raise ValueError("R1 case_thickness must be exactly zero")


def validate_figure_manifest(
    manifest: Mapping[str, Any], *, minimum_count: int = 70
) -> None:
    """Validate the reproducible parity-figure contract without image I/O."""
    if manifest.get("schema") != FIGURE_MANIFEST_SCHEMA:
        raise ValueError("unsupported geometry-figure manifest schema")
    figures = manifest.get("figures")
    if not isinstance(figures, list) or len(figures) < int(minimum_count):
        raise ValueError(
            f"geometry manifest requires at least {minimum_count} figures"
        )
    if manifest.get("figure_count") != len(figures):
        raise ValueError("figure_count disagrees with figure rows")
    filenames = [str(row.get("filename", "")) for row in figures]
    if len(filenames) != len(set(filenames)):
        raise ValueError("figure filenames must be unique")
    required = {
        "filename",
        "input_geometry_sha256",
        "canonical_geometry_fingerprint",
        "branch",
        "sha",
        "camera",
        "visible_components",
        "color_map",
        "coordinate_units",
        "status",
        "sha256",
    }
    for row in figures:
        missing = required - set(row)
        if missing:
            raise ValueError(f"figure row lacks {sorted(missing)}")
        if row["coordinate_units"] != "cm":
            raise ValueError("geometry figures must declare centimetres")
        if len(str(row["sha256"])) != 64:
            raise ValueError(
                "figure SHA-256 must contain 64 hexadecimal digits"
            )
    if manifest.get("matched_camera_and_color_policy") is not True:
        raise ValueError("matched camera and color policy must be explicit")
