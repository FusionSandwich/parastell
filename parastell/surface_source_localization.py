"""Localize OpenMC surface crossings on topology-oriented DAGMC facets."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _unit(values: np.ndarray, name: str) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError(f"{name} contains a zero or nonfinite vector")
    return values / norms[..., None]


def _barycentric(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    v0 = triangle[1] - triangle[0]
    v1 = triangle[2] - triangle[0]
    v2 = point - triangle[0]
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))
    denominator = d00 * d11 - d01 * d01
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("facet triangle is degenerate")
    second = (d11 * d20 - d01 * d21) / denominator
    third = (d00 * d21 - d01 * d20) / denominator
    return np.asarray([1.0 - second - third, second, third])


def localize_surface_crossings(
    phase_space: Mapping[str, np.ndarray],
    facet_catalog: Mapping[str, object],
    *,
    barycentric_tolerance: float = 1.0e-10,
    residual_tolerance_cm: float = 1.0e-8,
    grazing_tolerance: float = 1.0e-12,
) -> tuple[dict[str, np.ndarray], dict]:
    """Map crossings to exact facets and derive oriented local phase space.

    ``facet_catalog['normal_source']`` must state that normals came from the
    DAGMC forward/reverse surface topology.  This function deliberately does
    not guess outward sense from global coordinates.
    """

    required_phase = {
        "position_global_cm",
        "direction_global",
        "surface_id",
    }
    missing_phase = required_phase - set(phase_space)
    if missing_phase:
        raise ValueError(f"phase space omits {sorted(missing_phase)}")
    required_catalog = {
        "facet_id",
        "surface_id",
        "vertices_global_cm",
        "outward_normal_global",
        "normal_source",
    }
    missing_catalog = required_catalog - set(facet_catalog)
    if missing_catalog:
        raise ValueError(f"facet catalog omits {sorted(missing_catalog)}")
    if facet_catalog["normal_source"] != "dagmc_forward_reverse_topology":
        raise ValueError(
            "outward normals must come from DAGMC forward/reverse topology"
        )

    positions = np.asarray(phase_space["position_global_cm"], dtype=float)
    directions = _unit(
        np.asarray(phase_space["direction_global"], dtype=float),
        "direction_global",
    )
    record_surfaces = np.asarray(phase_space["surface_id"], dtype=np.int64)
    if (
        positions.shape != directions.shape
        or positions.ndim != 2
        or (positions.shape[1] != 3)
    ):
        raise ValueError(
            "positions and directions must have shape (records, 3)"
        )
    if len(record_surfaces) != len(positions):
        raise ValueError("surface IDs do not align with records")

    facet_ids = np.asarray(facet_catalog["facet_id"]).astype(str)
    facet_surfaces = np.asarray(
        facet_catalog["surface_id"], dtype=np.int64
    ).reshape(-1)
    triangles = np.asarray(facet_catalog["vertices_global_cm"], dtype=float)
    normals = _unit(
        np.asarray(facet_catalog["outward_normal_global"], dtype=float),
        "outward_normal_global",
    )
    count = len(facet_ids)
    if len(set(facet_ids.tolist())) != count:
        raise ValueError("facet IDs must be unique")
    if facet_surfaces.shape != (count,) or triangles.shape != (count, 3, 3):
        raise ValueError("facet catalog arrays are misaligned")
    if normals.shape != (count, 3):
        raise ValueError("facet normals must have shape (facets, 3)")
    geometric = _unit(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        "triangle geometric normal",
    )
    if np.any(np.abs(np.sum(geometric * normals, axis=1)) < 1.0 - 1.0e-10):
        raise ValueError("catalog normals are not parallel to facet geometry")

    selected_indices = np.empty(len(positions), dtype=np.int64)
    barycentrics = np.empty((len(positions), 3), dtype=float)
    residuals = np.empty(len(positions), dtype=float)
    reconstructions = np.empty_like(positions)
    local_positions = np.empty_like(positions)
    local_directions = np.empty_like(directions)
    local_frames = np.empty((len(positions), 3, 3), dtype=float)

    for record_index, (point, surface_id) in enumerate(
        zip(positions, record_surfaces)
    ):
        candidates = np.flatnonzero(facet_surfaces == surface_id)
        if not len(candidates):
            raise ValueError(f"surface {surface_id} has no catalog facets")
        matches = []
        for facet_index in candidates:
            triangle = triangles[facet_index]
            normal = normals[facet_index]
            signed_distance = float(np.dot(point - triangle[0], normal))
            projected = point - signed_distance * normal
            barycentric = _barycentric(projected, triangle)
            if (
                np.min(barycentric) >= -barycentric_tolerance
                and np.max(barycentric) <= 1.0 + barycentric_tolerance
                and abs(signed_distance) <= residual_tolerance_cm
            ):
                matches.append(
                    (abs(signed_distance), facet_index, barycentric, projected)
                )
        if not matches:
            raise ValueError(
                f"record {record_index} is not contained by a facet on "
                f"surface {surface_id}"
            )
        matches.sort(key=lambda item: (item[0], str(facet_ids[item[1]])))
        residual, facet_index, barycentric, reconstructed = matches[0]
        if np.linalg.norm(reconstructed - point) > residual_tolerance_cm:
            raise ValueError(
                "barycentric reconstruction residual is too large"
            )
        normal = normals[facet_index]
        tangent = triangles[facet_index, 1] - triangles[facet_index, 0]
        tangent = tangent - np.dot(tangent, normal) * normal
        tangent = _unit(tangent[None, :], "facet tangent")[0]
        bitangent = _unit(
            np.cross(normal, tangent)[None, :], "facet bitangent"
        )[0]
        frame = np.vstack([tangent, bitangent, normal])
        if not np.allclose(
            np.cross(frame[0], frame[1]), frame[2], atol=1.0e-12
        ):
            raise RuntimeError("derived facet-local frame is not right handed")

        selected_indices[record_index] = facet_index
        barycentrics[record_index] = barycentric
        residuals[record_index] = residual
        reconstructions[record_index] = reconstructed
        local_frames[record_index] = frame
        local_positions[record_index] = frame @ (
            point - triangles[facet_index, 0]
        )
        local_directions[record_index] = frame @ directions[record_index]

    record_normals = normals[selected_indices]
    mu = np.sum(directions * record_normals, axis=1)
    sense = np.where(
        mu < -grazing_tolerance,
        "incoming",
        np.where(mu > grazing_tolerance, "outgoing", "grazing"),
    )
    output = {
        **{name: np.asarray(value) for name, value in phase_space.items()},
        "facet_id": facet_ids[selected_indices],
        "facet_index": selected_indices,
        "barycentric_coordinates": barycentrics,
        "reconstructed_position_global_cm": reconstructions,
        "distance_to_facet_residual_cm": residuals,
        "outward_normal_global": record_normals,
        "mu": mu,
        "crossing_sense": sense,
        "position_local_cm": local_positions,
        "direction_local": local_directions,
        "local_frame_global": local_frames,
    }
    audit = {
        "schema": "parastell.surface_crossing_localization/v1.0.0",
        "status": "PASS",
        "record_count": int(len(positions)),
        "normal_source": facet_catalog["normal_source"],
        "barycentric_tolerance": float(barycentric_tolerance),
        "residual_tolerance_cm": float(residual_tolerance_cm),
        "grazing_tolerance_abs_mu": float(grazing_tolerance),
        "maximum_reconstruction_residual_cm": (
            float(np.max(residuals)) if len(residuals) else 0.0
        ),
        "incoming_records": int(np.count_nonzero(sense == "incoming")),
        "outgoing_records": int(np.count_nonzero(sense == "outgoing")),
        "grazing_records": int(np.count_nonzero(sense == "grazing")),
        "right_handed_local_frames": True,
    }
    return output, audit
