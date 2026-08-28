"""Order and normalization checks for an immutable ParaStell source mesh."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np


COORDINATE_QUANTUM_CM = 1.0e-8


def deduplicate_half_period_seam(
    original: Any,
    mate: Any,
    *,
    seam_degrees: float = 45.0,
    coordinate_quantum_cm: float = COORDINATE_QUANTUM_CM,
    angular_tolerance_degrees: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Merge only coincident vertices on the shared half-period seam."""
    left = np.asarray(original, dtype=float).reshape((-1, 3))
    right = np.asarray(mate, dtype=float).reshape((-1, 3))
    if left.shape != right.shape or not len(left):
        raise ValueError("original and mate coordinate arrays must agree")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("source coordinates must be finite")
    if coordinate_quantum_cm <= 0.0:
        raise ValueError("coordinate quantum must be positive")

    unique: list[np.ndarray] = []
    indices = np.empty(len(left) + len(right), dtype=np.int64)
    lookup: dict[tuple[int, int, int], tuple[int, int, int, np.ndarray]] = {}
    merged = []
    for side, coordinates in enumerate((left, right)):
        for local_index, coordinate in enumerate(coordinates):
            index = side * len(left) + local_index
            key = tuple(
                np.rint(coordinate / coordinate_quantum_cm).astype(np.int64)
            )
            prior = lookup.get(key)
            if prior is None:
                lookup[key] = (len(unique), side, local_index, coordinate)
                unique.append(coordinate)
                indices[index] = len(unique) - 1
                continue
            unique_index, prior_side, prior_index, prior_coordinate = prior
            if prior_side == side:
                raise ValueError(
                    "duplicate vertex within one half-period mesh"
                )
            if side != 1 or prior_side != 0:
                raise ValueError("ambiguous seam vertex ownership")
            residual = float(np.linalg.norm(coordinate - prior_coordinate))
            if residual > coordinate_quantum_cm:
                raise ValueError(
                    "rounded seam collision exceeds coordinate tolerance"
                )
            phi = float(
                np.mod(
                    np.degrees(np.arctan2(coordinate[1], coordinate[0])),
                    360.0,
                )
            )
            prior_phi = float(
                np.mod(
                    np.degrees(
                        np.arctan2(prior_coordinate[1], prior_coordinate[0])
                    ),
                    360.0,
                )
            )
            if not (
                np.isclose(
                    phi,
                    seam_degrees,
                    atol=angular_tolerance_degrees,
                    rtol=0.0,
                )
                and np.isclose(
                    prior_phi,
                    seam_degrees,
                    atol=angular_tolerance_degrees,
                    rtol=0.0,
                )
            ):
                raise ValueError(
                    "non-seam collision during half-period expansion"
                )
            indices[index] = unique_index
            merged.append(
                {
                    "original_vertex_index": int(prior_index),
                    "mate_vertex_index": int(local_index),
                    "phi_degrees": phi,
                    "residual_cm": residual,
                }
            )
    original_phi = np.mod(
        np.degrees(np.arctan2(left[:, 1], left[:, 0])), 360.0
    )
    mate_phi = np.mod(np.degrees(np.arctan2(right[:, 1], right[:, 0])), 360.0)
    original_seam_count = int(
        np.count_nonzero(
            np.isclose(
                original_phi,
                seam_degrees,
                atol=angular_tolerance_degrees,
                rtol=0.0,
            )
        )
    )
    mate_seam_count = int(
        np.count_nonzero(
            np.isclose(
                mate_phi,
                seam_degrees,
                atol=angular_tolerance_degrees,
                rtol=0.0,
            )
        )
    )
    if (
        not merged
        or len(merged) != original_seam_count
        or len(merged) != mate_seam_count
    ):
        raise ValueError(
            "half-period seam mapping is not a complete bijection"
        )
    return (
        np.asarray(unique, dtype=float),
        indices,
        {
            "merged_vertex_count": len(merged),
            "original_seam_vertex_count": original_seam_count,
            "mate_seam_vertex_count": mate_seam_count,
            "all_merged_vertices_on_seam": True,
            "seam_mapping_bijective": True,
            "maximum_merge_residual_cm": max(
                row["residual_cm"] for row in merged
            ),
            "seam_degrees": seam_degrees,
            "coordinate_quantum_cm": coordinate_quantum_cm,
            "angular_tolerance_degrees": angular_tolerance_degrees,
        },
    )


def audit_source_order_arrays(
    tagged_strengths: Any,
    external_strengths: Any,
    *,
    expected_rate_per_s: float,
) -> dict[str, Any]:
    """Prove an external strength vector matches MOAB tetrahedron order."""
    if (
        not math.isfinite(float(expected_rate_per_s))
        or float(expected_rate_per_s) <= 0.0
    ):
        raise ValueError("expected source rate must be finite and positive")
    tagged = np.asarray(tagged_strengths, dtype=np.float64).reshape(-1)
    external = np.asarray(external_strengths, dtype=np.float64).reshape(-1)
    if not len(tagged) or len(tagged) != len(external):
        raise ValueError("tagged and external strength counts disagree")
    if not np.all(np.isfinite(tagged)) or not np.all(np.isfinite(external)):
        raise ValueError("source strengths must be finite")
    if np.any(tagged < 0.0) or np.any(external < 0.0):
        raise ValueError("source strengths cannot be negative")
    tagged_bytes = np.asarray(tagged, dtype="<f8").tobytes()
    external_bytes = np.asarray(external, dtype="<f8").tobytes()
    rate = float(tagged.sum())
    rate_pass = math.isclose(
        rate, float(expected_rate_per_s), rel_tol=1.0e-14, abs_tol=0.0
    )
    return {
        "tetrahedron_count": int(len(tagged)),
        "element_order_exact_match": bool(np.array_equal(tagged, external)),
        "tagged_strength_vector_sha256": hashlib.sha256(
            tagged_bytes
        ).hexdigest(),
        "external_strength_vector_sha256": hashlib.sha256(
            external_bytes
        ).hexdigest(),
        "total_source_strength_per_s": rate,
        "expected_source_strength_per_s": float(expected_rate_per_s),
        "source_rate_closure_pass": rate_pass,
        "order_and_rate_gate_pass": bool(np.array_equal(tagged, external))
        and rate_pass,
    }


def audit_tetrahedral_seam_topology(
    coordinates: Any,
    connectivity: Any,
    *,
    seam_degrees: float = 45.0,
    angular_tolerance_degrees: float = 1.0e-8,
) -> dict[str, Any]:
    """Prove reflected source halves share a conformal internal seam."""
    points = np.asarray(coordinates, dtype=float).reshape((-1, 3))
    tets = np.asarray(connectivity, dtype=np.int64).reshape((-1, 4))
    if np.any(tets < 0) or np.any(tets >= len(points)):
        raise ValueError("tetrahedron connectivity is out of bounds")
    faces = np.sort(
        np.vstack(
            (
                tets[:, (0, 1, 2)],
                tets[:, (0, 1, 3)],
                tets[:, (0, 2, 3)],
                tets[:, (1, 2, 3)],
            )
        ),
        axis=1,
    )
    unique_faces, counts = np.unique(faces, axis=0, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("nonmanifold tetrahedral face multiplicity")
    boundary = unique_faces[counts == 1]
    boundary_points = points[boundary]
    phi = np.mod(
        np.degrees(
            np.arctan2(boundary_points[:, :, 1], boundary_points[:, :, 0])
        ),
        360.0,
    )
    seam_faces = np.all(
        np.isclose(
            phi,
            seam_degrees,
            atol=angular_tolerance_degrees,
            rtol=0.0,
        ),
        axis=1,
    )
    return {
        "unique_face_count": int(len(unique_faces)),
        "multiplicity_two_face_count": int(np.count_nonzero(counts == 2)),
        "boundary_face_count": int(len(boundary)),
        "maximum_face_multiplicity": int(counts.max()),
        "unmatched_internal_seam_face_count": int(
            np.count_nonzero(seam_faces)
        ),
        "conformal_internal_seam_pass": not np.any(seam_faces),
    }
