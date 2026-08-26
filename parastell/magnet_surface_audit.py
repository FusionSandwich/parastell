"""Strict DAGMC audits for magnet coupling surfaces.

The outer-casing coupling interface is deliberately different from the
boundary of the casing/winding-pack union.  Only faces owned by the casing
whose opposite volume is an explicitly accepted external region are eligible.
The casing-to-winding interface and any winding-only face are never used to
close that interface artificially.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .dagmc_envelope import DEFAULT_CANONICAL_COORDINATE_QUANTUM_CM
from .dagmc_envelope import _openmc_to_outward_normal_sign
from .dagmc_envelope import _quantize_coordinates
from .dagmc_envelope import _triangles
from .dagmc_envelope import canonical_dagmc_fingerprint


COUPLING_INTERFACES = ("outer_casing_external", "winding_pack")


def validate_coupling_interface(value: Any) -> str:
    """Require one, and only one, local-model coupling interface."""
    if isinstance(value, str):
        selected = [value]
    elif isinstance(value, Mapping):
        selected = [name for name in COUPLING_INTERFACES if value.get(name)]
        unknown = set(value) - set(COUPLING_INTERFACES)
        if unknown:
            raise ValueError(f"unknown coupling interfaces: {sorted(unknown)}")
    elif isinstance(value, Sequence):
        selected = list(value)
    else:
        raise TypeError("coupling_interface must be a string or selection")
    if len(selected) != 1 or selected[0] not in COUPLING_INTERFACES:
        raise ValueError(
            "a local source bundle must select exactly one coupling_interface: "
            "outer_casing_external or winding_pack"
        )
    return str(selected[0])


def _volume_id(value: Any) -> int | None:
    return None if value is None else int(value.id)


def _surface_senses(surface: Any) -> tuple[int | None, int | None]:
    return (
        _volume_id(getattr(surface, "forward_volume", None)),
        _volume_id(getattr(surface, "reverse_volume", None)),
    )


def adjacent_volume_id(surface: Any, volume_id: int) -> int | None:
    """Return the volume on the other side of a DAGMC surface."""
    forward_id, reverse_id = _surface_senses(surface)
    if forward_id == int(volume_id):
        return reverse_id
    if reverse_id == int(volume_id):
        return forward_id
    raise ValueError(
        f"surface {int(surface.id)} has no sense for volume {int(volume_id)}"
    )


@dataclass(frozen=True)
class SurfaceSubsetAudit:
    surface_ids: tuple[int, ...]
    surface_area_cm2: float
    triangle_count: int
    unique_edge_count: int
    bad_edge_count: int
    bad_edge_multiplicities: Mapping[str, int]
    edge_closure_pass: bool
    vector_area_global_cm2: tuple[float, float, float]
    vector_area_closure_relative: float | None
    vector_area_closure_pass: bool

    @property
    def closed(self) -> bool:
        return self.edge_closure_pass and self.vector_area_closure_pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_ids": list(self.surface_ids),
            "surface_area_cm2": self.surface_area_cm2,
            "triangle_count": self.triangle_count,
            "unique_edge_count": self.unique_edge_count,
            "bad_edge_count": self.bad_edge_count,
            "bad_edge_multiplicities": dict(self.bad_edge_multiplicities),
            "edge_closure_pass": self.edge_closure_pass,
            "vector_area_global_cm2": list(self.vector_area_global_cm2),
            "vector_area_closure_relative": self.vector_area_closure_relative,
            "vector_area_closure_pass": self.vector_area_closure_pass,
            "closed": self.closed,
        }


def _edge_key(first: np.ndarray, second: np.ndarray) -> tuple[Any, Any]:
    one = tuple(int(value) for value in first)
    two = tuple(int(value) for value in second)
    return (one, two) if one < two else (two, one)


def audit_surface_subset(
    volume: Any,
    surface_ids: Iterable[int],
    *,
    coordinate_quantum_cm: float = DEFAULT_CANONICAL_COORDINATE_QUANTUM_CM,
    vector_area_tolerance: float = 1.0e-7,
) -> SurfaceSubsetAudit:
    """Test edge multiplicity and vector-area closure for a face subset."""
    selected = tuple(sorted(set(int(value) for value in surface_ids)))
    by_id = {int(surface.id): surface for surface in volume.surfaces}
    missing = set(selected) - set(by_id)
    if missing:
        raise ValueError(
            f"surfaces {sorted(missing)} do not bound volume {int(volume.id)}"
        )
    edge_counts: dict[tuple[Any, Any], int] = {}
    vector_area = np.zeros(3, dtype=float)
    total_area = 0.0
    triangle_count = 0
    for surface_id in selected:
        surface = by_id[surface_id]
        triangles = _triangles(surface.triangle_coords)
        quantized = _quantize_coordinates(triangles, coordinate_quantum_cm)
        cross = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        area_vectors = (
            0.5
            * cross
            * _openmc_to_outward_normal_sign(surface, int(volume.id))
        )
        areas = np.linalg.norm(area_vectors, axis=1)
        if np.any(~np.isfinite(areas)) or np.any(areas <= 0.0):
            raise ValueError(f"surface {surface_id} has a degenerate facet")
        vector_area += area_vectors.sum(axis=0)
        total_area += float(areas.sum())
        triangle_count += len(triangles)
        for triangle in quantized:
            for first, second in ((0, 1), (1, 2), (2, 0)):
                key = _edge_key(triangle[first], triangle[second])
                edge_counts[key] = edge_counts.get(key, 0) + 1
    bad = {key: count for key, count in edge_counts.items() if count != 2}
    relative = (
        float(np.linalg.norm(vector_area) / total_area)
        if total_area > 0.0
        else None
    )
    return SurfaceSubsetAudit(
        selected,
        total_area,
        triangle_count,
        len(edge_counts),
        len(bad),
        {
            json.dumps(key, separators=(",", ":")): int(value)
            for key, value in sorted(bad.items())[:100]
        },
        bool(selected) and not bad,
        tuple(float(value) for value in vector_area),
        relative,
        relative is not None and relative <= float(vector_area_tolerance),
    )


def classify_casing_surfaces(
    casing_volume: Any,
    winding_pack_volume: Any,
    *,
    accepted_external_volume_ids: Iterable[int],
) -> dict[str, Any]:
    """Classify casing faces using DAGMC topology, never coordinate signs."""
    accepted = {int(value) for value in accepted_external_volume_ids}
    if not accepted:
        raise ValueError("at least one accepted external volume is required")
    casing_id = int(casing_volume.id)
    winding_id = int(winding_pack_volume.id)
    selected = []
    internal = []
    excluded = []
    rows = []
    for surface in sorted(
        casing_volume.surfaces, key=lambda item: int(item.id)
    ):
        surface_id = int(surface.id)
        adjacent = adjacent_volume_id(surface, casing_id)
        forward_id, reverse_id = _surface_senses(surface)
        triangles = _triangles(surface.triangle_coords)
        cross = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        area = float((0.5 * np.linalg.norm(cross, axis=1)).sum())
        outward_sign = _openmc_to_outward_normal_sign(surface, casing_id)
        if adjacent == winding_id:
            role = "casing_to_winding_pack_internal"
            decision = "EXCLUDED_INTERNAL_INTERFACE"
            internal.append(surface_id)
        elif adjacent in accepted:
            role = "outer_casing_external"
            decision = "SELECTED_ACCEPTED_EXTERNAL_REGION"
            selected.append(surface_id)
        else:
            role = "casing_other_interface"
            decision = "EXCLUDED_UNACCEPTED_ADJACENT_REGION"
            excluded.append(surface_id)
        rows.append(
            {
                "surface_id": surface_id,
                "area_cm2": area,
                "forward_volume_id": forward_id,
                "reverse_volume_id": reverse_id,
                "casing_outward_normal_sign": outward_sign,
                "adjacent_non_casing_volume_id": adjacent,
                "role": role,
                "decision": decision,
            }
        )
    winding_surface_ids = {
        int(surface.id) for surface in winding_pack_volume.surfaces
    }
    shared = set(internal)
    if shared != (
        {int(surface.id) for surface in casing_volume.surfaces}
        & winding_surface_ids
    ):
        raise ValueError(
            "casing/winding topology and surface ownership disagree"
        )
    return {
        "casing_volume_id": casing_id,
        "winding_pack_volume_id": winding_id,
        "accepted_external_volume_ids": sorted(accepted),
        "outer_casing_external_surface_ids": selected,
        "casing_internal_surface_ids": internal,
        "casing_other_excluded_surface_ids": excluded,
        "surface_adjacencies": rows,
        "selection_rule": (
            "retain only casing-owned faces whose topology-adjacent non-casing "
            "volume is explicitly accepted as external"
        ),
        "coordinate_sign_heuristic_used": False,
    }


def _load_associations(path: str | Path) -> dict[int, dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = value.get("associations", value)
    if not isinstance(rows, Mapping):
        raise ValueError("association artifact has no associations mapping")
    result = {int(key): dict(row) for key, row in rows.items()}
    inventory = value.get("inventory", {})
    inventory_rows = (
        inventory.get("volumes", []) if isinstance(inventory, Mapping) else []
    )
    for row in inventory_rows:
        volume_id = int(row["volume_id"])
        if volume_id in result:
            for name in ("component_role", "magnet_id", "coil_id", "material"):
                if name in row:
                    result[volume_id][name] = row[name]
    return result


def _pair_associations(
    associations: Mapping[int, Mapping[str, Any]],
) -> list[tuple[str, str, int, int]]:
    grouped: dict[tuple[str, str], dict[str, int]] = {}
    for volume_id, row in associations.items():
        role = str(row.get("component_role", row.get("role", "")))
        if role not in {"magnet_casing", "winding_pack"}:
            continue
        magnet_id = str(row.get("magnet_id", "")).strip()
        coil_id = str(row.get("coil_id", "")).strip()
        if not magnet_id or not coil_id:
            raise ValueError(
                f"volume {volume_id} lacks explicit magnet identity"
            )
        key = (magnet_id, coil_id)
        if role in grouped.setdefault(key, {}):
            raise ValueError(f"duplicate {role} association for {key}")
        grouped[key][role] = int(volume_id)
    result = []
    for key, roles in sorted(grouped.items()):
        missing = {"magnet_casing", "winding_pack"} - set(roles)
        if missing:
            raise ValueError(f"magnet {key} lacks {sorted(missing)}")
        result.append(
            (key[0], key[1], roles["magnet_casing"], roles["winding_pack"])
        )
    return result


def audit_all_magnet_surfaces(
    dagmc_path: str | Path,
    associations_path: str | Path,
    *,
    accepted_external_volume_ids: Iterable[int],
    expected_magnet_count: int = 18,
    coordinate_quantum_cm: float = DEFAULT_CANONICAL_COORDINATE_QUANTUM_CM,
    faceting_tolerances: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit both candidate coupling interfaces for every associated magnet."""
    import pydagmc

    geometry_path = Path(dagmc_path).resolve()
    model = pydagmc.Model(str(geometry_path))
    association_artifact = json.loads(
        Path(associations_path).read_text(encoding="utf-8")
    )
    declared_policy = association_artifact.get("canonical_geometry_policy", {})
    declared_quantum = float(
        declared_policy.get("coordinate_quantum_cm", coordinate_quantum_cm)
    )
    if not np.isclose(
        declared_quantum, coordinate_quantum_cm, rtol=0.0, atol=0.0
    ):
        raise ValueError(
            "requested coordinate quantum disagrees with association artifact"
        )
    if faceting_tolerances is None:
        faceting_tolerances = dict(
            declared_policy.get("faceting_tolerances", {})
        )
    associations = _load_associations(associations_path)
    pairs = _pair_associations(associations)
    if len(pairs) != int(expected_magnet_count):
        raise ValueError(
            f"expected {expected_magnet_count} casing/winding pairs, found {len(pairs)}"
        )
    fingerprint = canonical_dagmc_fingerprint(
        geometry_path,
        coordinate_quantum_cm=coordinate_quantum_cm,
        faceting_tolerances=faceting_tolerances,
    )
    declared_fingerprint = str(
        association_artifact.get("canonical_geometry_fingerprint", "")
    )
    if declared_fingerprint and fingerprint["canonical_fingerprint"] != (
        declared_fingerprint
    ):
        raise ValueError(
            "recomputed canonical geometry fingerprint disagrees with "
            "association artifact"
        )
    magnets = []
    for magnet_id, coil_id, casing_id, winding_id in pairs:
        try:
            casing = model.volumes_by_id[casing_id]
            winding = model.volumes_by_id[winding_id]
        except KeyError as exc:
            raise ValueError(
                f"associated DAGMC volume is absent: {exc}"
            ) from exc
        classification = classify_casing_surfaces(
            casing,
            winding,
            accepted_external_volume_ids=accepted_external_volume_ids,
        )
        outer = audit_surface_subset(
            casing,
            classification["outer_casing_external_surface_ids"],
            coordinate_quantum_cm=coordinate_quantum_cm,
        )
        winding_ids = tuple(
            sorted(int(surface.id) for surface in winding.surfaces)
        )
        pack = audit_surface_subset(
            winding,
            winding_ids,
            coordinate_quantum_cm=coordinate_quantum_cm,
        )
        winding_rows = []
        for surface in sorted(winding.surfaces, key=lambda item: int(item.id)):
            winding_rows.append(
                {
                    "surface_id": int(surface.id),
                    "area_cm2": float(
                        (
                            0.5
                            * np.linalg.norm(
                                np.cross(
                                    _triangles(surface.triangle_coords)[:, 1]
                                    - _triangles(surface.triangle_coords)[
                                        :, 0
                                    ],
                                    _triangles(surface.triangle_coords)[:, 2]
                                    - _triangles(surface.triangle_coords)[
                                        :, 0
                                    ],
                                ),
                                axis=1,
                            )
                        ).sum()
                    ),
                    "forward_volume_id": _surface_senses(surface)[0],
                    "reverse_volume_id": _surface_senses(surface)[1],
                    "winding_outward_normal_sign": _openmc_to_outward_normal_sign(
                        surface, winding_id
                    ),
                    "adjacent_non_winding_volume_id": adjacent_volume_id(
                        surface, winding_id
                    ),
                }
            )
        magnets.append(
            {
                "magnet_id": magnet_id,
                "coil_id": coil_id,
                **classification,
                "winding_pack_surface_ids": list(winding_ids),
                "winding_pack_surface_adjacencies": winding_rows,
                "outer_casing_external_closure": outer.to_dict(),
                "winding_pack_closure": pack.to_dict(),
                "normal_sense_status": "PASS_DAGMC_FORWARD_REVERSE_TOPOLOGY",
            }
        )
    digest = hashlib.sha256(geometry_path.read_bytes()).hexdigest()
    return {
        "schema": "parastell.all_magnet_surface_inventory/v1",
        "dagmc_path": str(geometry_path),
        "raw_h5m_sha256": digest,
        "canonical_geometry_fingerprint": fingerprint["canonical_fingerprint"],
        "coordinate_quantum_cm": coordinate_quantum_cm,
        "faceting_tolerances": dict(faceting_tolerances),
        "accepted_external_volume_ids": sorted(
            int(value) for value in accepted_external_volume_ids
        ),
        "magnet_count": len(magnets),
        "magnets": magnets,
        "winding_pack_envelopes_pass": all(
            row["winding_pack_closure"]["closed"] for row in magnets
        ),
        "outer_casing_envelopes_pass": all(
            row["outer_casing_external_closure"]["closed"] for row in magnets
        ),
        "all_internal_casing_interfaces_excluded": all(
            not (
                set(row["outer_casing_external_surface_ids"])
                & set(row["casing_internal_surface_ids"])
            )
            for row in magnets
        ),
    }


def write_all_magnet_surface_audit(
    output_path: str | Path,
    dagmc_path: str | Path,
    associations_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    report = audit_all_magnet_surfaces(dagmc_path, associations_path, **kwargs)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
