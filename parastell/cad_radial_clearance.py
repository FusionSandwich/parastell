"""Source-CAD ray measurements for clearance-constrained radial builds."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def first_triangle_hit_distances(
    origins: np.ndarray,
    directions: np.ndarray,
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    maximum_distance_cm: float,
    determinant_tolerance: float = 1.0e-12,
    barycentric_tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Return the first positive ray/triangle hit distance for each ray.

    This is the Möller–Trumbore test evaluated one ray at a time to keep peak
    memory bounded for detailed magnet tessellations.  A ray with no hit inside
    ``maximum_distance_cm`` is represented by positive infinity.
    """

    ray_origins = np.asarray(origins, dtype=float)
    ray_directions = np.asarray(directions, dtype=float)
    points = np.asarray(vertices, dtype=float)
    facets = np.asarray(triangles, dtype=np.int64)
    if ray_origins.ndim != 2 or ray_origins.shape[1] != 3:
        raise ValueError("origins must have shape (N, 3)")
    if ray_directions.shape != ray_origins.shape:
        raise ValueError("directions must match origins")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("vertices must have shape (V, 3)")
    if facets.ndim != 2 or facets.shape[1] != 3:
        raise ValueError("triangles must have shape (T, 3)")
    if np.any(facets < 0) or np.any(facets >= len(points)):
        raise ValueError("triangle vertex index is out of range")
    if maximum_distance_cm <= 0.0 or not np.isfinite(maximum_distance_cm):
        raise ValueError("maximum_distance_cm must be finite and positive")

    norms = np.linalg.norm(ray_directions, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("directions must be finite and nonzero")
    unit_directions = ray_directions / norms[:, np.newaxis]

    vertex0 = points[facets[:, 0]]
    edge1 = points[facets[:, 1]] - vertex0
    edge2 = points[facets[:, 2]] - vertex0
    distances = np.full(len(ray_origins), np.inf, dtype=float)

    for index, (origin, direction) in enumerate(
        zip(ray_origins, unit_directions)
    ):
        pvec = np.cross(direction, edge2)
        determinant = np.einsum("ij,ij->i", edge1, pvec)
        valid = np.abs(determinant) > determinant_tolerance
        if not np.any(valid):
            continue
        inverse = np.zeros_like(determinant)
        inverse[valid] = 1.0 / determinant[valid]
        tvec = origin - vertex0
        u = np.einsum("ij,ij->i", tvec, pvec) * inverse
        valid &= u >= -barycentric_tolerance
        valid &= u <= 1.0 + barycentric_tolerance
        if not np.any(valid):
            continue
        qvec = np.cross(tvec, edge1)
        v = np.einsum("j,ij->i", direction, qvec) * inverse
        valid &= v >= -barycentric_tolerance
        valid &= u + v <= 1.0 + barycentric_tolerance
        if not np.any(valid):
            continue
        distance = np.einsum("ij,ij->i", edge2, qvec) * inverse
        valid &= distance > determinant_tolerance
        valid &= distance <= maximum_distance_cm
        if np.any(valid):
            distances[index] = float(np.min(distance[valid]))

    return distances


def tessellate_cadquery_solids(
    solids: Iterable,
    *,
    tolerance_cm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine CadQuery solid tessellations into indexed triangle arrays."""

    if tolerance_cm <= 0.0 or not np.isfinite(tolerance_cm):
        raise ValueError("tolerance_cm must be finite and positive")
    all_vertices = []
    all_triangles = []
    offset = 0
    for solid in solids:
        vertices, triangles = solid.tessellate(tolerance_cm)
        coordinates = np.array(
            [[vertex.x, vertex.y, vertex.z] for vertex in vertices],
            dtype=float,
        )
        indices = np.asarray(triangles, dtype=np.int64)
        if len(indices):
            all_vertices.append(coordinates)
            all_triangles.append(indices + offset)
            offset += len(coordinates)
    if not all_triangles:
        raise ValueError("CAD tessellation produced no triangles")
    return np.vstack(all_vertices), np.vstack(all_triangles)


def sample_first_wall_rays(
    ref_surf,
    toroidal_angles_deg: np.ndarray,
    poloidal_angles_deg: np.ndarray,
    wall_s: float,
    *,
    scale: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reference-wall loci and topology-independent outward rays."""

    from . import invessel_build as ivb

    toroidal = np.asarray(toroidal_angles_deg, dtype=float)
    poloidal = np.asarray(poloidal_angles_deg, dtype=float)
    zeros = np.zeros((len(toroidal), len(poloidal)))
    radial_build = ivb.RadialBuild(
        toroidal,
        poloidal,
        wall_s,
        {"chamber": {"thickness_matrix": zeros}},
        split_chamber=False,
    )
    build = ivb.InVesselBuild(
        ref_surf,
        radial_build,
        num_ribs=len(toroidal),
        num_rib_pts=len(poloidal),
        scale=scale,
    )
    build.populate_surfaces()
    build.calculate_loci()
    surface = build.Surfaces["chamber"]
    origins = surface.get_loci()
    directions = np.array([rib._normals() for rib in surface.Ribs])
    return origins, directions


def measure_first_wall_to_magnets(
    ref_surf,
    toroidal_angles_deg: np.ndarray,
    poloidal_angles_deg: np.ndarray,
    wall_s: float,
    magnet_solids: Iterable,
    *,
    tessellation_tolerance_cm: float = 0.1,
    maximum_distance_cm: float = 500.0,
    scale: float = 100.0,
) -> dict:
    """Measure first-wall-to-outer-magnet distance along build normals."""

    origins, directions = sample_first_wall_rays(
        ref_surf,
        toroidal_angles_deg,
        poloidal_angles_deg,
        wall_s,
        scale=scale,
    )
    vertices, triangles = tessellate_cadquery_solids(
        magnet_solids,
        tolerance_cm=tessellation_tolerance_cm,
    )
    distances = first_triangle_hit_distances(
        origins.reshape((-1, 3)),
        directions.reshape((-1, 3)),
        vertices,
        triangles,
        maximum_distance_cm=maximum_distance_cm,
    ).reshape(origins.shape[:-1])
    return {
        "origins_cm": origins,
        "directions": directions,
        "distances_cm": distances,
        "hit_mask": np.isfinite(distances),
        "tessellation_tolerance_cm": float(tessellation_tolerance_cm),
        "maximum_distance_cm": float(maximum_distance_cm),
        "triangle_count": int(len(triangles)),
        "vertex_count": int(len(vertices)),
    }


def project_dense_deficit_to_control_grid(
    dense_deficit_cm: np.ndarray,
    dense_toroidal_angles_deg: np.ndarray,
    dense_poloidal_angles_deg: np.ndarray,
    control_toroidal_angles_deg: np.ndarray,
    control_poloidal_angles_deg: np.ndarray,
) -> np.ndarray:
    """Conservatively project dense clearance deficits to build controls.

    Every positive dense deficit is assigned to all control-grid corners of
    its containing angular cell. The maximum is retained at each control
    point. This creates compact transition support while preventing a
    sub-control-grid violation from being averaged away.
    """

    dense = np.asarray(dense_deficit_cm, dtype=float)
    dense_phi = np.asarray(dense_toroidal_angles_deg, dtype=float)
    dense_theta = np.asarray(dense_poloidal_angles_deg, dtype=float)
    control_phi = np.asarray(control_toroidal_angles_deg, dtype=float)
    control_theta = np.asarray(control_poloidal_angles_deg, dtype=float)
    if dense.shape != (len(dense_phi), len(dense_theta)):
        raise ValueError("dense deficit shape does not match dense angles")
    for name, values in (
        ("dense toroidal", dense_phi),
        ("dense poloidal", dense_theta),
        ("control toroidal", control_phi),
        ("control poloidal", control_theta),
    ):
        if values.ndim != 1 or len(values) < 2 or np.any(np.diff(values) <= 0):
            raise ValueError(f"{name} angles must be strictly increasing")
    if dense_phi[0] < control_phi[0] or dense_phi[-1] > control_phi[-1]:
        raise ValueError("dense toroidal angles exceed the control domain")
    if (
        dense_theta[0] < control_theta[0]
        or dense_theta[-1] > control_theta[-1]
    ):
        raise ValueError("dense poloidal angles exceed the control domain")
    if np.any(~np.isfinite(dense)) or np.any(dense < 0.0):
        raise ValueError("dense deficits must be finite and nonnegative")

    def bracketing_indices(
        value: float, controls: np.ndarray
    ) -> tuple[int, ...]:
        exact = np.flatnonzero(
            np.isclose(controls, value, rtol=0.0, atol=1.0e-12)
        )
        if len(exact):
            return (int(exact[0]),)
        high = int(np.searchsorted(controls, value, side="right"))
        high = min(max(high, 1), len(controls) - 1)
        return (high - 1, high)

    control = np.zeros((len(control_phi), len(control_theta)), dtype=float)
    for i, phi in enumerate(dense_phi):
        phi_indices = bracketing_indices(phi, control_phi)
        for j, theta in enumerate(dense_theta):
            deficit = dense[i, j]
            if deficit <= 0.0:
                continue
            theta_indices = bracketing_indices(theta, control_theta)
            for control_i in phi_indices:
                for control_j in theta_indices:
                    control[control_i, control_j] = max(
                        control[control_i, control_j], deficit
                    )

    closure = np.maximum(control[:, 0], control[:, -1])
    control[:, 0] = closure
    control[:, -1] = closure
    return control
