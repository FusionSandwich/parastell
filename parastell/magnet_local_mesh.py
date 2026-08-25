"""Stable centreline-oriented local tally meshes for homogenized magnets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


CANDIDATE_RESOLUTIONS_CM = (5.0, 2.0, 1.0, 0.5)


@dataclass(frozen=True)
class LocalMeshDefinition:
    magnet_id: str
    lower_left_local_cm: tuple[float, float, float]
    upper_right_local_cm: tuple[float, float, float]
    dimension: tuple[int, int, int]
    rotation_local_to_global: tuple[tuple[float, float, float], ...]
    translation_global_cm: tuple[float, float, float]
    requested_resolution_cm: float
    frame_type: str = "coil_centerline_parallel_transport"

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower_left_local_cm, dtype=float)
        upper = np.asarray(self.upper_right_local_cm, dtype=float)
        dimension = np.asarray(self.dimension, dtype=int)
        rotation = np.asarray(self.rotation_local_to_global, dtype=float)
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or np.any(upper <= lower)
            or np.any(dimension <= 0)
        ):
            raise ValueError("local mesh bounds and dimensions are invalid")
        if rotation.shape != (3, 3) or not np.allclose(
            rotation @ rotation.T, np.eye(3), atol=1.0e-10
        ):
            raise ValueError("local mesh rotation must be orthonormal")
        if np.linalg.det(rotation) < 1.0 - 1.0e-10:
            raise ValueError("local mesh rotation must be right handed")

    @property
    def widths_cm(self) -> np.ndarray:
        return (
            np.asarray(self.upper_right_local_cm)
            - np.asarray(self.lower_left_local_cm)
        ) / np.asarray(self.dimension)

    @property
    def bin_volume_cm3(self) -> float:
        return float(np.prod(self.widths_cm))

    @property
    def bin_count(self) -> int:
        return int(np.prod(self.dimension))

    def bin_metadata(
        self, centreline_frame: Any | None = None
    ) -> dict[str, Any]:
        """Return deterministic per-bin geometry and centreline linkage.

        OpenMC defines the mesh-bin order, while ``centreline_frame`` supplies
        the curved engineering frame evaluated independently at every bin
        centroid.  The latter is optional so manifests created before
        centreline linkage remain readable.  When it is absent, the returned
        status is explicit and no nearest-centreline quantities are invented.
        """
        lower = np.asarray(self.lower_left_local_cm, dtype=float)
        widths = self.widths_cm
        # OpenMC regular-mesh filter bins advance x first, then y, then z.
        indices = np.column_stack(
            np.unravel_index(
                np.arange(self.bin_count), self.dimension, order="F"
            )
        )
        local = lower + (indices + 0.5) * widths
        rotation = np.asarray(self.rotation_local_to_global)
        translation = np.asarray(self.translation_global_cm)
        global_centroid = local @ rotation.T + translation
        metadata = {
            "mesh_bin_ids": np.asarray(
                [
                    f"{self.magnet_id}-bin-{index:08d}"
                    for index in range(len(local))
                ],
                dtype=object,
            ),
            "local_centreline_coordinates_cm": local,
            "mesh_local_centroid_cm": local.copy(),
            "global_centroid_cm": global_centroid,
            "volume_cm3": np.full(len(local), self.bin_volume_cm3),
            "indices_ijk": indices,
            "centreline_linkage_available": centreline_frame is not None,
            "centreline_linkage_status": np.full(
                len(local),
                (
                    "LINKED_NEAREST_CENTRELINE_SEGMENT"
                    if centreline_frame is not None
                    else "UNAVAILABLE_CENTRELINE_FRAME_NOT_SUPPLIED"
                ),
                dtype=object,
            ),
        }
        if centreline_frame is None:
            return metadata

        sampled = centreline_frame.sample(global_centroid)
        vector_fields = (
            "nearest_centreline_global_cm",
            "centreline_tangent",
            "centreline_radial",
            "centreline_transverse",
            "local_centreline_coordinates_cm",
        )
        scalar_fields = (
            "centreline_arclength_cm",
            "normalized_arclength",
            "distance_to_centreline_cm",
        )
        for name in vector_fields:
            values = np.asarray(sampled[name], dtype=float)
            if values.shape != (self.bin_count, 3):
                raise ValueError(
                    f"centreline frame returned invalid {name} shape"
                )
            metadata[name] = values
        for name in scalar_fields:
            values = np.asarray(sampled[name], dtype=float)
            if values.shape != (self.bin_count,):
                raise ValueError(
                    f"centreline frame returned invalid {name} shape"
                )
            metadata[name] = values
        metadata["frame_type"] = np.full(
            self.bin_count, str(sampled["frame_type"]), dtype=object
        )
        metadata["frame_quality_status"] = np.full(
            self.bin_count,
            str(sampled["frame_quality_status"]),
            dtype=object,
        )
        return metadata

    def openmc_filter(self):
        """Construct OpenMC objects lazily; mesh coordinates remain local."""
        try:
            import openmc
        except ImportError as exc:
            raise RuntimeError(
                "OpenMC is required to construct a local mesh filter"
            ) from exc
        mesh = openmc.RegularMesh(name=f"pstl_local_{self.magnet_id}")
        mesh.lower_left = self.lower_left_local_cm
        mesh.upper_right = self.upper_right_local_cm
        mesh.dimension = self.dimension
        mesh_filter = openmc.MeshFilter(mesh)
        mesh_filter.rotation = np.asarray(self.rotation_local_to_global)
        mesh_filter.translation = np.asarray(self.translation_global_cm)
        return mesh_filter

    def to_dict(self) -> dict[str, Any]:
        return {
            "magnet_id": self.magnet_id,
            "lower_left_local_cm": list(self.lower_left_local_cm),
            "upper_right_local_cm": list(self.upper_right_local_cm),
            "dimension": list(self.dimension),
            "rotation_local_to_global": [
                list(value) for value in self.rotation_local_to_global
            ],
            "translation_global_cm": list(self.translation_global_cm),
            "requested_resolution_cm": self.requested_resolution_cm,
            "actual_bin_widths_cm": self.widths_cm.tolist(),
            "bin_count": self.bin_count,
            "bin_volume_cm3": self.bin_volume_cm3,
            "frame_type": self.frame_type,
        }


def _global_axis_aligned_bounds(
    mesh: LocalMeshDefinition,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(mesh.lower_left_local_cm, dtype=float)
    upper = np.asarray(mesh.upper_right_local_cm, dtype=float)
    corners = np.asarray(
        [
            [x, y, z]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ]
    )
    rotation = np.asarray(mesh.rotation_local_to_global, dtype=float)
    translation = np.asarray(mesh.translation_global_cm, dtype=float)
    global_corners = corners @ rotation.T + translation
    return global_corners.min(axis=0), global_corners.max(axis=0)


def qualify_local_mesh_nonoverlap(
    meshes: Sequence[LocalMeshDefinition],
    *,
    cell_filter_applied: bool,
    separation_tolerance_cm: float = 1.0e-9,
) -> dict[str, Any]:
    """Conservatively qualify disjoint, component-filtered R2S meshes.

    A separating global AABB axis proves that two rotated mesh volumes are
    disjoint.  Overlapping AABBs are reported as unqualified even when a more
    expensive oriented-box test might prove separation.  This one-sided test
    cannot create a false non-overlap claim.  Spatial mesh tallies that are not
    component-filtered remain unqualified regardless of geometric separation.
    """

    selected = tuple(meshes)
    if not selected:
        raise ValueError("at least one local mesh is required")
    if type(cell_filter_applied) is not bool:
        raise TypeError("cell_filter_applied must be boolean")
    tolerance = float(separation_tolerance_cm)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            "mesh separation tolerance must be finite and nonnegative"
        )
    magnet_ids = [mesh.magnet_id for mesh in selected]
    if len(magnet_ids) != len(set(magnet_ids)):
        raise ValueError("local mesh magnet IDs must be unique")
    bounds = {
        mesh.magnet_id: _global_axis_aligned_bounds(mesh) for mesh in selected
    }
    pair_rows = []
    all_pairs_separated = True
    for first_index, first in enumerate(selected):
        first_lower, first_upper = bounds[first.magnet_id]
        for second in selected[first_index + 1 :]:
            second_lower, second_upper = bounds[second.magnet_id]
            separated_axes = [
                axis
                for axis in range(3)
                if first_upper[axis] + tolerance < second_lower[axis]
                or second_upper[axis] + tolerance < first_lower[axis]
            ]
            separated = bool(separated_axes)
            all_pairs_separated = all_pairs_separated and separated
            pair_rows.append(
                {
                    "magnet_ids": [first.magnet_id, second.magnet_id],
                    "separated": separated,
                    "separating_global_axes": separated_axes,
                }
            )
    qualified = bool(cell_filter_applied and all_pairs_separated)
    blockers = []
    if not cell_filter_applied:
        blockers.append("LOCAL_MESH_TALLIES_ARE_NOT_COMPONENT_FILTERED")
    if not all_pairs_separated:
        blockers.append("PAIRWISE_GEOMETRIC_DISJOINTNESS_NOT_PROVEN")
    return {
        "method": "conservative_global_aabb_separating_axis",
        "separation_tolerance_cm": tolerance,
        "mesh_count": len(selected),
        "tested_pair_count": len(pair_rows),
        "pairwise_evidence": pair_rows,
        "geometric_pairwise_disjoint_proven": all_pairs_separated,
        "cell_filter_applied": cell_filter_applied,
        "nonoverlap_qualified": qualified,
        "status": "QUALIFIED" if qualified else "NOT_QUALIFIED",
        "blocking_reasons": blockers,
    }


def build_local_mesh_definition(
    magnet_id: str,
    *,
    bounding_box_global_cm: Sequence[Sequence[float]],
    centreline_sample: dict[str, Any],
    resolution_cm: float,
    padding_cm: float = 0.0,
) -> LocalMeshDefinition:
    """Bound a global magnet box in one local engineering frame."""
    if resolution_cm <= 0.0 or padding_cm < 0.0:
        raise ValueError(
            "local mesh resolution must be positive and padding nonnegative"
        )
    bounds = np.asarray(bounding_box_global_cm, dtype=float)
    if bounds.shape != (2, 3) or np.any(bounds[1] <= bounds[0]):
        raise ValueError("magnet bounding box must have shape (2, 3)")
    corners = np.asarray(
        [
            [x, y, z]
            for x in bounds[:, 0]
            for y in bounds[:, 1]
            for z in bounds[:, 2]
        ]
    )
    tangent = np.asarray(centreline_sample["centreline_tangent"], dtype=float)
    radial = np.asarray(centreline_sample["centreline_radial"], dtype=float)
    transverse = np.asarray(
        centreline_sample["centreline_transverse"], dtype=float
    )
    rotation = np.column_stack((tangent, radial, transverse))
    translation = np.asarray(
        centreline_sample["nearest_centreline_global_cm"], dtype=float
    )
    local = (corners - translation) @ rotation
    lower = local.min(axis=0) - padding_cm
    upper = local.max(axis=0) + padding_cm
    dimension = np.maximum(
        1, np.ceil((upper - lower) / resolution_cm).astype(int)
    )
    return LocalMeshDefinition(
        magnet_id,
        tuple(lower),
        tuple(upper),
        tuple(int(value) for value in dimension),
        tuple(tuple(float(item) for item in row) for row in rotation),
        tuple(translation),
        float(resolution_cm),
    )
