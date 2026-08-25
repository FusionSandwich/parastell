"""Continuous engineering frames attached to magnet coil centrelines.

The frame is deliberately labelled as an engineering centreline frame.  It
does not claim to reconstruct conductor twist or the orientation of individual
REBCO tapes inside a homogenized winding pack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


FRAME_TYPE = "coil_centerline_parallel_transport"


def _unit(vector: np.ndarray, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        raise ValueError(f"{label} must be finite and nonzero")
    return np.asarray(vector, dtype=float) / norm


def _rotate(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate one vector with Rodrigues' formula."""
    axis = _unit(axis, "rotation axis")
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * np.dot(axis, vector) * (1.0 - np.cos(angle))
    )


def _transport(vector: np.ndarray, first: np.ndarray, second: np.ndarray):
    cross = np.cross(first, second)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    if sine <= 1.0e-14:
        if cosine >= 0.0:
            return vector.copy()
        # An exact tangent reversal has no unique minimum rotation and signals
        # an invalid centreline discretization for this engineering frame.
        raise ValueError("adjacent centreline tangents reverse direction")
    return _rotate(vector, cross / sine, float(np.arctan2(sine, cosine)))


def _signed_angle(first: np.ndarray, second: np.ndarray, axis: np.ndarray):
    return float(
        np.arctan2(
            np.dot(axis, np.cross(first, second)), np.dot(first, second)
        )
    )


@dataclass(frozen=True)
class CentrelineFrame:
    """Piecewise-linear periodic centreline with a transported local frame."""

    points_cm: np.ndarray
    arclength_cm: np.ndarray
    tangents: np.ndarray
    radial_directions: np.ndarray
    transverse_directions: np.ndarray
    closed: bool
    closure_twist_rad: float
    quality_status: str

    def __post_init__(self) -> None:
        points = np.asarray(self.points_cm, dtype=float)
        count = len(points)
        if points.ndim != 2 or points.shape[1] != 3 or count < 3:
            raise ValueError("centreline points must have shape (N>=3, 3)")
        for name in (
            "tangents",
            "radial_directions",
            "transverse_directions",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != points.shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must align with centreline points")
        length = np.asarray(self.arclength_cm, dtype=float)
        if length.shape != (count + int(self.closed),):
            raise ValueError("arclength axis has an invalid shape")
        if length[0] != 0.0 or np.any(np.diff(length) <= 0.0):
            raise ValueError("arclength must start at zero and increase")
        t = np.asarray(self.tangents)
        r = np.asarray(self.radial_directions)
        b = np.asarray(self.transverse_directions)
        if not (
            np.allclose(np.linalg.norm(t, axis=1), 1.0, atol=1.0e-10)
            and np.allclose(np.linalg.norm(r, axis=1), 1.0, atol=1.0e-10)
            and np.allclose(np.linalg.norm(b, axis=1), 1.0, atol=1.0e-10)
            and np.allclose(np.sum(t * r, axis=1), 0.0, atol=1.0e-10)
            and np.allclose(np.cross(t, r), b, atol=1.0e-10)
        ):
            raise ValueError(
                "centreline frames must be right-handed orthonormal"
            )

    @property
    def total_length_cm(self) -> float:
        return float(self.arclength_cm[-1])

    def sample(self, positions_global_cm: Sequence[Sequence[float]]):
        """Map positions to the nearest centreline segment and local frame."""
        positions = np.asarray(positions_global_cm, dtype=float)
        scalar = positions.shape == (3,)
        positions = np.atleast_2d(positions)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if len(positions) == 0:
            empty_vectors = np.empty((0, 3), dtype=float)
            empty_scalars = np.empty(0, dtype=float)
            return {
                "nearest_centreline_global_cm": empty_vectors.copy(),
                "centreline_arclength_cm": empty_scalars.copy(),
                "normalized_arclength": empty_scalars.copy(),
                "centreline_tangent": empty_vectors.copy(),
                "centreline_radial": empty_vectors.copy(),
                "centreline_transverse": empty_vectors.copy(),
                "local_centreline_coordinates_cm": empty_vectors.copy(),
                "distance_to_centreline_cm": empty_scalars.copy(),
                "frame_type": FRAME_TYPE,
                "frame_quality_status": self.quality_status,
            }
        starts = self.points_cm
        ends = np.roll(starts, -1, axis=0) if self.closed else starts[1:]
        starts = starts if self.closed else starts[:-1]
        edges = ends - starts
        square = np.einsum("ij,ij->i", edges, edges)
        offset = positions[:, None, :] - starts[None, :, :]
        fraction = np.clip(
            np.einsum("nsi,si->ns", offset, edges) / square[None, :],
            0.0,
            1.0,
        )
        closest = starts[None, :, :] + fraction[:, :, None] * edges[None, :, :]
        distance2 = np.sum((positions[:, None, :] - closest) ** 2, axis=2)
        segment = np.argmin(distance2, axis=1)
        row = np.arange(len(positions))
        f = fraction[row, segment]
        nearest = closest[row, segment]
        next_index = (segment + 1) % len(self.points_cm)
        tangent = np.asarray(
            [_unit(value, "interpolated tangent") for value in edges[segment]]
        )
        radial = (1.0 - f[:, None]) * self.radial_directions[segment]
        radial += f[:, None] * self.radial_directions[next_index]
        radial -= np.sum(radial * tangent, axis=1)[:, None] * tangent
        radial = np.asarray(
            [_unit(value, "interpolated radial") for value in radial]
        )
        transverse = np.cross(tangent, radial)
        segment_length = np.sqrt(square[segment])
        arclength = self.arclength_cm[segment] + f * segment_length
        displacement = positions - nearest
        result: dict[str, Any] = {
            "nearest_centreline_global_cm": nearest,
            "centreline_arclength_cm": arclength,
            "normalized_arclength": arclength / self.total_length_cm,
            "centreline_tangent": tangent,
            "centreline_radial": radial,
            "centreline_transverse": transverse,
            "local_centreline_coordinates_cm": np.column_stack(
                (
                    arclength,
                    np.sum(displacement * radial, axis=1),
                    np.sum(displacement * transverse, axis=1),
                )
            ),
            "distance_to_centreline_cm": np.linalg.norm(displacement, axis=1),
            "frame_type": FRAME_TYPE,
            "frame_quality_status": self.quality_status,
        }
        if scalar:
            return {
                key: value[0] if isinstance(value, np.ndarray) else value
                for key, value in result.items()
            }
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_type": FRAME_TYPE,
            "closed": self.closed,
            "point_count": len(self.points_cm),
            "total_length_cm": self.total_length_cm,
            "closure_twist_rad": self.closure_twist_rad,
            "frame_quality_status": self.quality_status,
        }


def parallel_transport_frame(
    points_cm: Sequence[Sequence[float]],
    *,
    closed: bool = True,
    initial_radial: Sequence[float] | None = None,
) -> CentrelineFrame:
    """Construct a continuous, sign-stable parallel-transport frame."""
    points = np.asarray(points_cm, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("centreline points must have shape (N>=3, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("centreline points must be finite")
    if closed and np.linalg.norm(points[-1] - points[0]) <= 1.0e-10:
        points = points[:-1]
    if len(points) < 3:
        raise ValueError("centreline has too few distinct points")
    edges = (
        np.roll(points, -1, axis=0) - points
        if closed
        else np.diff(points, axis=0)
    )
    edge_lengths = np.linalg.norm(edges, axis=1)
    if np.any(edge_lengths <= 1.0e-12):
        raise ValueError("centreline contains duplicate adjacent points")
    unit_edges = edges / edge_lengths[:, None]
    if closed:
        tangents = np.asarray(
            [
                _unit(unit_edges[index - 1] + unit_edges[index], "tangent")
                for index in range(len(points))
            ]
        )
    else:
        tangents = np.empty_like(points)
        tangents[0], tangents[-1] = unit_edges[0], unit_edges[-1]
        for index in range(1, len(points) - 1):
            tangents[index] = _unit(
                unit_edges[index - 1] + unit_edges[index], "tangent"
            )
    if initial_radial is None:
        axes = np.eye(3)
        seed = axes[int(np.argmin(np.abs(axes @ tangents[0])))]
    else:
        seed = np.asarray(initial_radial, dtype=float)
    seed = seed - np.dot(seed, tangents[0]) * tangents[0]
    radial = np.empty_like(points)
    radial[0] = _unit(seed, "initial radial")
    for index in range(1, len(points)):
        moved = _transport(
            radial[index - 1], tangents[index - 1], tangents[index]
        )
        moved -= np.dot(moved, tangents[index]) * tangents[index]
        radial[index] = _unit(moved, "transported radial")
    closure_twist = 0.0
    if closed:
        returned = _transport(radial[-1], tangents[-1], tangents[0])
        closure_twist = _signed_angle(returned, radial[0], tangents[0])
        cumulative = np.concatenate(([0.0], np.cumsum(edge_lengths)))
        fractions = cumulative[:-1] / cumulative[-1]
        radial = np.asarray(
            [
                _rotate(value, tangent, closure_twist * fraction)
                for value, tangent, fraction in zip(
                    radial, tangents, fractions
                )
            ]
        )
        arclength = cumulative
    else:
        arclength = np.concatenate(([0.0], np.cumsum(edge_lengths)))
    radial -= np.sum(radial * tangents, axis=1)[:, None] * tangents
    radial = np.asarray([_unit(value, "radial") for value in radial])
    transverse = np.cross(tangents, radial)
    quality = (
        "QUALIFIED"
        if float(np.max(np.abs(np.diff(tangents, axis=0)))) < 0.75
        else "COARSE_CENTRELINE"
    )
    return CentrelineFrame(
        points,
        arclength,
        tangents,
        radial,
        transverse,
        closed,
        closure_twist,
        quality,
    )
