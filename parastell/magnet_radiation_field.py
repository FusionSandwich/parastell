"""Backend-neutral public facade for the magnet-radiation producer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .coil_frame import CentrelineFrame, parallel_transport_frame
from .dagmc_envelope import (
    DEFAULT_CANONICAL_COORDINATE_QUANTUM_CM,
    DagmcEnvelope,
    MagnetPairRecord,
    MagnetVolumeInventory,
    canonical_geometry_policy,
    discover_magnet_volumes,
    extract_closed_component_union,
    extract_closed_envelope,
    select_magnet_pairs,
)


SCHEMA = "parastell.magnet_radiation_producer/v1.0.0"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cad_solid_boundary_signature(
    solid,
    *,
    linear_tolerance_cm: float = 0.5,
    angular_tolerance_rad: float = 0.1,
    coordinate_quantum_cm: float = 1.0e-6,
    refinement_factor: float = 0.25,
    maximum_refinement_relative_change: float = 0.0025,
) -> dict[str, Any]:
    """Measure a CAD solid from an independently tessellated closed boundary.

    ParaStell filament magnets contain periodic ruled B-spline faces for which
    OCC mass properties can disagree materially with two independent boundary
    tessellators.  Transport follows the boundary, so identity is bound to a
    closed, consistently oriented CadQuery boundary tessellation.  The OCC
    mass is retained as a diagnostic and is never silently substituted for
    the boundary volume.
    """
    linear = float(linear_tolerance_cm)
    angular = float(angular_tolerance_rad)
    quantum = float(coordinate_quantum_cm)
    refinement = float(refinement_factor)
    convergence_limit = float(maximum_refinement_relative_change)
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in (linear, angular, quantum, convergence_limit)
    ):
        raise ValueError("CAD boundary signature tolerances must be positive")
    if not np.isfinite(refinement) or not 0.0 < refinement < 1.0:
        raise ValueError("CAD boundary refinement_factor must be in (0, 1)")

    occ_mass = float(solid.Volume())
    if not np.isfinite(occ_mass) or occ_mass <= 0.0:
        raise ValueError("CAD solid OCC mass volume must be positive")
    if not hasattr(solid, "tessellate"):
        bounds = solid.BoundingBox()
        return {
            "volume_cm3": occ_mass,
            "bounding_box_cm": (
                (float(bounds.xmin), float(bounds.ymin), float(bounds.zmin)),
                (float(bounds.xmax), float(bounds.ymax), float(bounds.zmax)),
            ),
            "measurement_method": "occ_mass_fallback_no_tessellator",
            "occ_mass_volume_cm3": occ_mass,
            "occ_mass_boundary_relative_difference": 0.0,
            "triangle_count": None,
            "edge_count": None,
            "vector_area_closure_relative": None,
            "linear_tolerance_cm": linear,
            "angular_tolerance_rad": angular,
            "coordinate_quantum_cm": quantum,
        }

    if hasattr(solid, "isValid") and not bool(solid.isValid()):
        raise ValueError("CAD solid fails OCC BRep validity")

    def measure(deflection, angular_deflection):
        # CadQuery/OCC otherwise reuses a cached triangulation created at a
        # prior deflection, which would make a two-resolution convergence
        # claim meaningless.
        try:
            from OCP.BRepTools import BRepTools

            BRepTools.Clean_s(solid.wrapped)
        except (ImportError, AttributeError):
            # Lightweight test doubles do not expose an OCC wrapped shape.
            pass
        vertices, connectivity = solid.tessellate(
            deflection, angularTolerance=angular_deflection
        )
        coordinates = np.asarray(
            [
                (float(vertex.x), float(vertex.y), float(vertex.z))
                for vertex in vertices
            ],
            dtype=float,
        )
        indices = np.asarray(connectivity, dtype=int)
        if (
            coordinates.ndim != 2
            or coordinates.shape[1] != 3
            or indices.ndim != 2
            or indices.shape[1] != 3
            or len(indices) == 0
            or np.any(indices < 0)
            or np.any(indices >= len(coordinates))
        ):
            raise ValueError("CAD boundary tessellation is invalid")
        triangles = coordinates[indices]
        area_vectors = 0.5 * np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        areas = np.linalg.norm(area_vectors, axis=1)
        if np.any(~np.isfinite(areas)) or np.any(areas <= 0.0):
            raise ValueError("CAD boundary contains degenerate triangles")

        quantized = np.rint(triangles / quantum).astype(np.int64)
        edge_balance: dict[
            tuple[tuple[int, int, int], tuple[int, int, int]], list[Any]
        ] = {}
        vertex_triangles: dict[tuple[int, int, int], set[int]] = {}
        for triangle_index, triangle in enumerate(quantized):
            points = [
                tuple(int(value) for value in point) for point in triangle
            ]
            for point in points:
                vertex_triangles.setdefault(point, set()).add(triangle_index)
            for first, second in ((0, 1), (1, 2), (2, 0)):
                start = points[first]
                end = points[second]
                key = (start, end) if start < end else (end, start)
                direction = 1 if start < end else -1
                counts = edge_balance.setdefault(key, [0, 0, []])
                counts[0] += 1
                counts[1] += direction
                counts[2].append(triangle_index)
        invalid_edges = {
            edge: (counts[0], counts[1])
            for edge, counts in edge_balance.items()
            if counts[0] != 2 or counts[1] != 0
        }
        if invalid_edges:
            raise ValueError(
                "CAD boundary is not a consistently oriented closed "
                f"triangle shell: {list(invalid_edges.items())[:5]}"
            )

        adjacency = [set() for _ in range(len(triangles))]
        for counts in edge_balance.values():
            first, second = counts[2]
            adjacency[first].add(second)
            adjacency[second].add(first)
        for point, incident in vertex_triangles.items():
            pending = {next(iter(incident))}
            reached = set()
            while pending:
                current = pending.pop()
                if current in reached:
                    continue
                reached.add(current)
                pending.update(adjacency[current] & incident)
            if reached != incident:
                raise ValueError(
                    "CAD boundary has a non-manifold vertex link at "
                    f"{point}"
                )

        remaining = set(range(len(triangles)))
        components = []
        while remaining:
            pending = {next(iter(remaining))}
            component = set()
            while pending:
                current = pending.pop()
                if current in component:
                    continue
                component.add(current)
                pending.update(adjacency[current] - component)
            remaining.difference_update(component)
            selection = np.asarray(sorted(component), dtype=int)
            component_triangles = triangles[selection]
            component_areas = areas[selection]
            component_vectors = area_vectors[selection]
            component_closure = float(
                np.linalg.norm(np.sum(component_vectors, axis=0))
                / float(np.sum(component_areas))
            )
            if component_closure > 1.0e-7:
                raise ValueError(
                    "CAD boundary component vector-area closure failed: "
                    f"{component_closure}"
                )
            origin = np.mean(component_triangles.reshape((-1, 3)), axis=0)
            local = component_triangles - origin
            component_volume = float(
                np.einsum(
                    "ij,ij->i",
                    local[:, 0],
                    np.cross(local[:, 1], local[:, 2]),
                ).sum()
                / 6.0
            )
            if not np.isfinite(component_volume) or component_volume == 0.0:
                raise ValueError("CAD boundary component volume is invalid")
            components.append(
                {
                    "triangle_count": int(len(selection)),
                    "signed_volume_cm3": component_volume,
                    "vector_area_closure_relative": component_closure,
                }
            )
        positive = [row for row in components if row["signed_volume_cm3"] > 0]
        if len(positive) != 1:
            raise ValueError(
                "CAD boundary must contain one outward outer component and "
                "zero or more inward cavity components"
            )
        signed_volume = float(
            sum(row["signed_volume_cm3"] for row in components)
        )
        if not np.isfinite(signed_volume) or signed_volume <= 0.0:
            raise ValueError(
                "CAD boundary outward signed volume must be positive, got "
                f"{signed_volume} cm3"
            )
        total_area = float(np.sum(areas))
        closure = float(
            np.linalg.norm(np.sum(area_vectors, axis=0)) / total_area
        )
        canonical_triangles = sorted(
            tuple(
                sorted(tuple(int(value) for value in point) for point in row)
            )
            for row in quantized
        )
        boundary_hash = hashlib.sha256(
            json.dumps(canonical_triangles, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        lower = np.min(coordinates, axis=0)
        upper = np.max(coordinates, axis=0)
        return {
            "volume_cm3": signed_volume,
            "bounding_box_cm": (
                tuple(float(value) for value in lower),
                tuple(float(value) for value in upper),
            ),
            "triangle_count": int(len(triangles)),
            "edge_count": int(len(edge_balance)),
            "boundary_component_count": int(len(components)),
            "boundary_components": components,
            "vector_area_closure_relative": closure,
            "canonical_boundary_sha256": boundary_hash,
        }

    refinement_rows = [
        {
            "linear_tolerance_cm": linear,
            "angular_tolerance_rad": angular,
            "measurement": measure(linear, angular),
        },
        {
            "linear_tolerance_cm": linear * refinement,
            "angular_tolerance_rad": angular * refinement,
            "measurement": measure(linear * refinement, angular * refinement),
        },
    ]

    def relative_change(first, second):
        return abs(
            second["measurement"]["volume_cm3"]
            - first["measurement"]["volume_cm3"]
        ) / abs(second["measurement"]["volume_cm3"])

    convergence_change = relative_change(
        refinement_rows[-2], refinement_rows[-1]
    )
    if convergence_change > convergence_limit:
        refinement_rows.append(
            {
                "linear_tolerance_cm": linear * refinement**2,
                "angular_tolerance_rad": angular * refinement**2,
                "measurement": measure(
                    linear * refinement**2, angular * refinement**2
                ),
            }
        )
        convergence_change = relative_change(
            refinement_rows[-2], refinement_rows[-1]
        )
    coarse = refinement_rows[0]["measurement"]
    fine = refinement_rows[-1]["measurement"]
    if convergence_change > convergence_limit:
        raise ValueError(
            "CAD boundary tessellation volume did not converge: relative "
            f"change {convergence_change} exceeds {convergence_limit}; "
            f"coarse={coarse['volume_cm3']} cm3, "
            f"fine={fine['volume_cm3']} cm3, OCC mass={occ_mass} cm3"
        )
    return {
        **fine,
        "measurement_method": (
            "convergence_qualified_cadquery_closed_boundary_divergence"
        ),
        "occ_mass_volume_cm3": occ_mass,
        "occ_mass_boundary_relative_difference": (
            abs(occ_mass - fine["volume_cm3"]) / fine["volume_cm3"]
        ),
        "occ_brepcheck_valid": True,
        "coarse_boundary_volume_cm3": coarse["volume_cm3"],
        "refined_boundary_volume_cm3": fine["volume_cm3"],
        "refinement_relative_change": convergence_change,
        "refinement_level_count": len(refinement_rows),
        "refinement_history": [
            {
                "linear_tolerance_cm": row["linear_tolerance_cm"],
                "angular_tolerance_rad": row["angular_tolerance_rad"],
                "volume_cm3": row["measurement"]["volume_cm3"],
                "triangle_count": row["measurement"]["triangle_count"],
                "canonical_boundary_sha256": row["measurement"][
                    "canonical_boundary_sha256"
                ],
            }
            for row in refinement_rows
        ],
        "maximum_refinement_relative_change": convergence_limit,
        "coarse_linear_tolerance_cm": linear,
        "refined_linear_tolerance_cm": refinement_rows[-1][
            "linear_tolerance_cm"
        ],
        "coarse_angular_tolerance_rad": angular,
        "refined_angular_tolerance_rad": refinement_rows[-1][
            "angular_tolerance_rad"
        ],
        "coordinate_quantum_cm": quantum,
        "self_intersection_evidence": (
            "OCC_BREPCHECK_VALID_PLUS_CLOSED_MANIFOLD_BOUNDARY"
        ),
    }


def filament_associations(
    inventory: MagnetVolumeInventory,
    magnet_set,
    *,
    coils_path: str | Path,
    machine_id: str,
    sector_id: str,
    volume_relative_tolerance: float = 0.02,
    bounding_box_tolerance_cm: float = 25.0,
    cad_boundary_linear_tolerance_cm: float = 0.5,
    cad_boundary_angular_tolerance_rad: float = 0.1,
) -> dict[int, dict[str, Any]]:
    """Bind DAGMC pairs to the exact pre-export clipped CAD solid groups."""
    from scipy.optimize import linear_sum_assignment

    coils = tuple(magnet_set.magnet_coils)
    if len(coils) != len(inventory.pairs):
        raise ValueError(
            "source filament count does not match discovered winding packs"
        )
    if len(magnet_set.coil_solids) != len(coils):
        raise ValueError("CAD solid groups do not align with source filaments")
    if volume_relative_tolerance <= 0.0 or bounding_box_tolerance_cm <= 0.0:
        raise ValueError("association tolerances must be positive")

    def solid_signature(solids):
        if len(solids) not in {1, 2}:
            raise ValueError(
                "filament magnet groups require one or two solids"
            )
        roles = (
            ("magnet_casing", "winding_pack")
            if len(solids) == 2
            else ("winding_pack",)
        )
        result = {}
        for role, solid in zip(roles, solids):
            result[role] = cad_solid_boundary_signature(
                solid,
                linear_tolerance_cm=cad_boundary_linear_tolerance_cm,
                angular_tolerance_rad=cad_boundary_angular_tolerance_rad,
            )
        return result

    def pair_signature(pair):
        result = {
            "winding_pack": {
                "volume_cm3": float(pair.winding_pack.volume_cm3),
                "bounding_box_cm": pair.winding_pack.bounding_box_cm,
            }
        }
        if pair.casing is not None:
            result["magnet_casing"] = {
                "volume_cm3": float(pair.casing.volume_cm3),
                "bounding_box_cm": pair.casing.bounding_box_cm,
            }
        return result

    cad_signatures = [
        solid_signature(solids) for solids in magnet_set.coil_solids
    ]
    dagmc_signatures = [pair_signature(pair) for pair in inventory.pairs]
    errors = {}
    costs = np.full((len(coils), len(inventory.pairs)), np.inf, dtype=float)
    for cad_index, cad in enumerate(cad_signatures):
        for pair_index, dagmc in enumerate(dagmc_signatures):
            if set(cad) != set(dagmc):
                continue
            volume_error = 0.0
            box_error = 0.0
            for role in cad:
                expected_volume = float(dagmc[role]["volume_cm3"])
                if expected_volume <= 0.0:
                    raise ValueError("DAGMC magnet volume must be positive")
                volume_error = max(
                    volume_error,
                    abs(float(cad[role]["volume_cm3"]) - expected_volume)
                    / expected_volume,
                )
                cad_box = np.asarray(cad[role]["bounding_box_cm"], dtype=float)
                dagmc_box = np.asarray(
                    dagmc[role]["bounding_box_cm"], dtype=float
                )
                box_error = max(
                    box_error, float(np.max(np.abs(cad_box - dagmc_box)))
                )
            errors[(cad_index, pair_index)] = (volume_error, box_error)
            costs[cad_index, pair_index] = (
                volume_error / volume_relative_tolerance
                + box_error / bounding_box_tolerance_cm
            )
    cad_rows, pair_columns = linear_sum_assignment(costs)
    if len(cad_rows) != len(coils):
        raise ValueError("CAD-to-DAGMC magnet assignment is incomplete")
    pair_to_cad = {}
    for cad_index, pair_index in zip(cad_rows, pair_columns):
        volume_error, box_error = errors[(int(cad_index), int(pair_index))]
        if (
            not np.isfinite(costs[cad_index, pair_index])
            or volume_error > volume_relative_tolerance
            or box_error > bounding_box_tolerance_cm
        ):
            raise ValueError(
                "CAD-to-DAGMC magnet identity exceeds declared tolerances: "
                f"cad_group={cad_index}, pair={pair_index}, "
                f"volume_relative_error={volume_error}, "
                f"bounding_box_error_cm={box_error}"
            )
        competing = np.delete(costs[:, pair_index], int(cad_index))
        if len(competing) and np.isclose(
            float(np.min(competing)),
            float(costs[cad_index, pair_index]),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                f"CAD-to-DAGMC identity is ambiguous for pair {pair_index}"
            )
        pair_to_cad[int(pair_index)] = {
            "cad_group_index": int(cad_index),
            "volume_relative_error": volume_error,
            "bounding_box_error_cm": box_error,
        }

    result = {}
    coil_file_hash = _sha256(coils_path)
    for pair_index, pair in enumerate(inventory.pairs):
        match = pair_to_cad[pair_index]
        index = match["cad_group_index"]
        coil_id = f"coil-{index:04d}"
        magnet_id = f"{machine_id}-{sector_id}-{coil_id}"
        provenance = {
            "source_kind": "ParaStell MagnetSetFromFilaments",
            "coils_path": str(Path(coils_path).resolve()),
            "coils_sha256": coil_file_hash,
            "ordered_filament_index": index,
            "centreline_point_count": len(coils[index].coords),
            "cad_solid_group_index": index,
            "cad_to_dagmc_identity": {
                "method": (
                    "global_role_closed_boundary_volume_and_bounding_box_assignment"
                ),
                "volume_relative_tolerance": volume_relative_tolerance,
                "bounding_box_tolerance_cm": bounding_box_tolerance_cm,
                "volume_relative_error": match["volume_relative_error"],
                "bounding_box_error_cm": match["bounding_box_error_cm"],
                "cad_signature": cad_signatures[index],
            },
        }
        shared = {
            "magnet_id": magnet_id,
            "coil_id": coil_id,
            "source_coil_provenance": provenance,
        }
        result[pair.winding_pack.volume_id] = {
            **shared,
            "component_name": f"{magnet_id}-winding-pack",
            "component_id": f"{magnet_id}:winding_pack",
        }
        if pair.casing is not None:
            result[pair.casing.volume_id] = {
                **shared,
                "component_name": f"{magnet_id}-casing",
                "component_id": f"{magnet_id}:casing",
            }
    return result


@dataclass(frozen=True)
class ProducerSelection:
    magnet_selection: str | int | Sequence[str | int] = "all"
    particles: tuple[str, ...] = ("neutron", "photon")
    tally_profile: str = "magnet_damage_and_handoff"
    boundary_source: bool = True
    volume_flux: bool = True
    local_mesh: bool = True

    def __post_init__(self) -> None:
        if not self.particles or set(self.particles) - {"neutron", "photon"}:
            raise ValueError(
                "producer particles must be neutron and/or photon"
            )


class MagnetRadiationFieldProducer:
    """Coordinate discovery, envelopes, tallies, and neutral provenance only."""

    def __init__(
        self,
        dagmc_path: str | Path,
        *,
        selection: ProducerSelection | None = None,
        associations: Mapping[int, Mapping[str, Any]] | None = None,
        centreline_points_by_coil: (
            Mapping[str, Sequence[Sequence[float]]] | None
        ) = None,
        coordinate_quantum_cm: float = DEFAULT_CANONICAL_COORDINATE_QUANTUM_CM,
        faceting_tolerances: Mapping[str, Any] | None = None,
        expected_canonical_geometry_fingerprint: str | None = None,
    ):
        self.dagmc_path = Path(dagmc_path).resolve()
        if not self.dagmc_path.is_file():
            raise FileNotFoundError(self.dagmc_path)
        self.selection = selection or ProducerSelection()
        self.associations = dict(associations or {})
        self._centreline_points = dict(centreline_points_by_coil or {})
        self.canonical_geometry_policy = canonical_geometry_policy(
            coordinate_quantum_cm, faceting_tolerances
        )
        self.expected_canonical_geometry_fingerprint = (
            str(expected_canonical_geometry_fingerprint)
            if expected_canonical_geometry_fingerprint
            else None
        )
        self.inventory: MagnetVolumeInventory | None = None
        self.selected_pairs: tuple[MagnetPairRecord, ...] = ()
        self.centreline_frames: dict[str, CentrelineFrame] = {}
        self.envelopes: tuple[DagmcEnvelope, ...] = ()
        self.tally_inventory = None

    def discover(self) -> MagnetVolumeInventory:
        self.inventory = discover_magnet_volumes(
            self.dagmc_path,
            associations=self.associations,
            **self.canonical_geometry_policy,
        )
        if (
            self.expected_canonical_geometry_fingerprint is not None
            and self.inventory.canonical_geometry_fingerprint
            != self.expected_canonical_geometry_fingerprint
        ):
            raise ValueError(
                "canonical geometry fingerprint does not match the expected "
                "geometry identity policy"
            )
        self.selected_pairs = select_magnet_pairs(
            self.inventory, self.selection.magnet_selection
        )
        for pair in self.selected_pairs:
            if pair.coil_id in self._centreline_points:
                self.centreline_frames[pair.magnet_id] = (
                    parallel_transport_frame(
                        self._centreline_points[pair.coil_id], closed=True
                    )
                )
        return self.inventory

    def build_envelopes(
        self,
        *,
        frames_by_magnet: (
            Mapping[str, Mapping[str, Sequence[float]]] | None
        ) = None,
        spatial_bins: tuple[int, int] = (4, 4),
        boundary_roles: Sequence[str] = (
            "outer_magnet",
            "winding_pack",
        ),
    ) -> tuple[DagmcEnvelope, ...]:
        if self.inventory is None:
            self.discover()
        roles = tuple(str(value) for value in boundary_roles)
        supported_roles = {"outer_magnet", "winding_pack"}
        if (
            not roles
            or len(roles) != len(set(roles))
            or set(roles) - supported_roles
        ):
            raise ValueError(
                "boundary_roles must contain distinct outer_magnet and/or "
                "winding_pack values"
            )
        explicit = dict(frames_by_magnet or {})
        outputs = []
        for pair in self.selected_pairs:
            centreline = self.centreline_frames.get(pair.magnet_id)
            if pair.magnet_id in explicit:
                frame = explicit[pair.magnet_id]
            elif centreline is not None:
                sampled = centreline.sample(
                    pair.winding_pack.centroid_global_cm
                )
                frame = {
                    "plasma_direction_global": -np.asarray(
                        sampled["centreline_radial"]
                    ),
                    "toroidal_direction_global": sampled["centreline_tangent"],
                    "poloidal_direction_global": sampled[
                        "centreline_transverse"
                    ],
                }
            else:
                raise ValueError(
                    f"magnet {pair.magnet_id!r} requires an explicit orientation frame or centreline"
                )
            required = {
                "plasma_direction_global",
                "toroidal_direction_global",
                "poloidal_direction_global",
            }
            missing = required - set(frame)
            if missing:
                raise ValueError(
                    f"magnet {pair.magnet_id} frame is missing {sorted(missing)}"
                )
            common = {
                "dagmc_path": self.dagmc_path,
                "magnet_id": pair.magnet_id,
                "plasma_direction_global": frame["plasma_direction_global"],
                "toroidal_direction_global": frame[
                    "toroidal_direction_global"
                ],
                "poloidal_direction_global": frame[
                    "poloidal_direction_global"
                ],
                "spatial_bins": spatial_bins,
                **self.canonical_geometry_policy,
                "centreline_frame": centreline,
            }
            if "outer_magnet" in roles:
                casing = getattr(pair, "casing", None)
                if casing is not None:
                    outer = extract_closed_component_union(
                        volume_ids=(
                            casing.volume_id,
                            pair.winding_pack.volume_id,
                        ),
                        envelope_id=f"outer-magnet-{pair.magnet_id}",
                        **common,
                    )
                    if hasattr(outer, "envelope"):
                        # Preserve how the surface was constructed separately
                        # from the stable public interface role requested by
                        # this producer method.
                        outer.envelope.metadata["construction_kind"] = (
                            outer.envelope.metadata.get("boundary_role")
                        )
                        outer.envelope.metadata["boundary_role"] = (
                            "outer_magnet"
                        )
                    outputs.append(outer)
                elif "winding_pack" not in roles:
                    outer = extract_closed_envelope(
                        self.dagmc_path,
                        pair.winding_pack.volume_id,
                        envelope_id=f"outer-magnet-{pair.magnet_id}",
                        **{
                            key: value
                            for key, value in common.items()
                            if key != "dagmc_path"
                        },
                    )
                    if hasattr(outer, "envelope"):
                        outer.envelope.metadata["boundary_role"] = (
                            "outer_magnet"
                        )
                        outer.envelope.metadata["dagmc_volume_ids"] = [
                            pair.winding_pack.volume_id
                        ]
                    outputs.append(outer)
            if "winding_pack" in roles:
                winding = extract_closed_envelope(
                    self.dagmc_path,
                    pair.winding_pack.volume_id,
                    envelope_id=f"winding-pack-{pair.magnet_id}",
                    **{
                        key: value
                        for key, value in common.items()
                        if key != "dagmc_path"
                    },
                )
                if hasattr(winding, "envelope"):
                    winding.envelope.metadata["boundary_role"] = "winding_pack"
                    winding.envelope.metadata["dagmc_volume_ids"] = [
                        pair.winding_pack.volume_id
                    ]
                outputs.append(winding)
        self.envelopes = tuple(outputs)
        return self.envelopes

    def attach_openmc(
        self,
        model,
        *,
        neutron_edges_eV: Sequence[float],
        photon_edges_eV: Sequence[float],
        volume_flux_energy_axes=None,
        local_mesh_filters_by_cell=None,
        supported_responses: Sequence[str] | None = None,
    ):
        """Attach definitions to a caller-owned model; never execute transport."""
        from .openmc16 import add_envelope_tallies, configure_transport

        if not self.envelopes:
            raise RuntimeError(
                "closed magnet envelopes must be built before tallies"
            )
        if not hasattr(model, "settings") or model.settings is None:
            raise ValueError("OpenMC model must have settings")
        configure_transport(model.settings)
        surface_ids = sorted(
            {
                surface
                for envelope in self.envelopes
                for surface in envelope.envelope.surface_ids
            }
        )
        cell_ids = [
            component.volume_id
            for pair in self.selected_pairs
            for component in (pair.winding_pack, pair.casing)
            if component is not None
        ]
        self.tally_inventory = add_envelope_tallies(
            model,
            surface_ids=surface_ids,
            cell_ids=cell_ids,
            neutron_edges_eV=neutron_edges_eV,
            photon_edges_eV=photon_edges_eV,
            volume_flux_energy_axes=volume_flux_energy_axes,
            tally_profile=self.selection.tally_profile,
            local_mesh_filters_by_cell=(
                local_mesh_filters_by_cell
                if self.selection.local_mesh
                else None
            ),
            supported_responses=supported_responses,
        )
        return self.tally_inventory

    def manifest(self) -> dict[str, Any]:
        if self.inventory is None:
            raise RuntimeError("magnet discovery has not run")
        return {
            "schema": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dagmc": {
                "path": str(self.dagmc_path),
                "raw_h5m_sha256": _sha256(self.dagmc_path),
                "canonical_geometry_fingerprint": self.inventory.canonical_geometry_fingerprint,
                "canonical_geometry_policy": self.canonical_geometry_policy,
            },
            "selection": {
                "magnet_selection": self.selection.magnet_selection,
                "particles": list(self.selection.particles),
                "tally_profile": self.selection.tally_profile,
                "boundary_source": self.selection.boundary_source,
                "volume_flux": self.selection.volume_flux,
                "local_mesh": self.selection.local_mesh,
            },
            "magnet_inventory": self.inventory.to_dict(),
            "selected_magnet_ids": [
                pair.magnet_id for pair in self.selected_pairs
            ],
            "boundary_roles": sorted(
                {
                    str(
                        item.envelope.metadata.get(
                            "boundary_role", "winding_pack"
                        )
                    )
                    for item in self.envelopes
                }
            ),
            "envelopes": [item.envelope.to_dict() for item in self.envelopes],
            "centreline_frames": {
                name: frame.to_dict()
                for name, frame in self.centreline_frames.items()
            },
            "tallies": (
                self.tally_inventory.to_dict()
                if self.tally_inventory is not None
                else None
            ),
            "execution_performed": False,
        }

    def write_manifest(self, path: str | Path) -> dict[str, Any]:
        value = self.manifest()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return value
