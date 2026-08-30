"""Qualification of ParaStell tetrahedral source meshes in DAGMC geometry."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .reference_geometry import _triangles, sha256_file
from .source_geometry_identity import source_mesh_fingerprint


SOURCE_QUADRATURE_BARYCENTRICS = np.asarray(
    [
        [0.25, 0.25, 0.25, 0.25],
        [0.5, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 0.5, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 1.0 / 6.0, 0.5, 1.0 / 6.0],
        [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 0.5],
    ],
    dtype=float,
)


def audit_source_tetrahedra_arrays(
    tetrahedron_coordinates_cm: Any,
    tagged_volumes_cm3: Any,
    source_strengths_n_per_s: Any,
    *,
    volume_relative_tolerance: float = 1.0e-10,
    volume_absolute_tolerance_cm3: float = 1.0e-12,
) -> dict[str, Any]:
    """Audit tetrahedron geometry/tags and select a canonical source point."""
    tetrahedra = np.asarray(tetrahedron_coordinates_cm, dtype=float)
    tagged = np.asarray(tagged_volumes_cm3, dtype=float).reshape(-1)
    strengths = np.asarray(source_strengths_n_per_s, dtype=float).reshape(-1)
    if tetrahedra.ndim != 3 or tetrahedra.shape[1:] != (4, 3):
        raise ValueError("tetrahedron coordinates must have shape (n,4,3)")
    if len(tetrahedra) != len(tagged) or len(tetrahedra) != len(strengths):
        raise ValueError("tetrahedron, volume, and strength counts disagree")
    if not len(tetrahedra):
        raise ValueError("source mesh has no tetrahedra")

    finite_tetrahedra = np.all(np.isfinite(tetrahedra), axis=(1, 2))
    finite_tags = np.isfinite(tagged) & np.isfinite(strengths)
    geometric = (
        np.abs(
            np.linalg.det(
                np.stack(
                    (
                        tetrahedra[:, 1] - tetrahedra[:, 0],
                        tetrahedra[:, 2] - tetrahedra[:, 0],
                        tetrahedra[:, 3] - tetrahedra[:, 0],
                    ),
                    axis=1,
                )
            )
        )
        / 6.0
    )
    positive_geometric = geometric > 0.0
    positive_tagged = tagged > 0.0
    volume_matches = np.isclose(
        geometric,
        tagged,
        rtol=float(volume_relative_tolerance),
        atol=float(volume_absolute_tolerance_cm3),
    )
    nonnegative_strength = strengths >= 0.0
    valid = (
        finite_tetrahedra
        & finite_tags
        & positive_geometric
        & positive_tagged
        & volume_matches
        & nonnegative_strength
    )

    positive_source = np.flatnonzero(valid & (strengths > 0.0))
    if not len(positive_source):
        raise ValueError(
            "source mesh has no valid positive-strength tetrahedron"
        )
    canonical_rows = []
    for index in positive_source:
        vertices = sorted(
            tuple(float(value) for value in row) for row in tetrahedra[index]
        )
        payload = np.asarray(vertices, dtype="<f8").tobytes()
        payload += np.asarray(
            [strengths[index], tagged[index]], dtype="<f8"
        ).tobytes()
        canonical_rows.append(
            (hashlib.sha256(payload).hexdigest(), int(index))
        )
    selected_digest, selected_index = min(canonical_rows)
    selected_point = tetrahedra[selected_index].mean(axis=0)

    invalid_indices = np.flatnonzero(~valid)
    return {
        "tetrahedron_count": int(len(tetrahedra)),
        "invalid_tetrahedron_count": int(len(invalid_indices)),
        "first_invalid_tetrahedron_indices": [
            int(value) for value in invalid_indices[:100]
        ],
        "nonfinite_tetrahedron_count": int((~finite_tetrahedra).sum()),
        "nonfinite_tag_count": int((~finite_tags).sum()),
        "nonpositive_geometric_volume_count": int((~positive_geometric).sum()),
        "nonpositive_tagged_volume_count": int((~positive_tagged).sum()),
        "tagged_volume_mismatch_count": int((~volume_matches).sum()),
        "negative_source_strength_count": int((strengths < 0.0).sum()),
        "total_source_strength_n_per_s": float(strengths.sum()),
        "total_geometric_volume_cm3": float(geometric.sum()),
        "total_tagged_volume_cm3": float(tagged.sum()),
        "selected_tetrahedron": {
            "canonical_payload_sha256": selected_digest,
            "source_mesh_order_index": selected_index,
            "source_strength_n_per_s": float(strengths[selected_index]),
            "tagged_volume_cm3": float(tagged[selected_index]),
        },
        "selected_source_point_cm": [float(value) for value in selected_point],
        "tetrahedron_data_gate_pass": bool(np.all(valid)),
    }


def _source_arrays(
    source_mesh: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from pymoab import core, types

    mesh = core.Core()
    mesh.load_file(str(source_mesh))
    tetrahedra = mesh.get_entities_by_type(mesh.get_root_set(), types.MBTET)
    connectivity = mesh.get_connectivity(tetrahedra)
    coordinates = np.asarray(
        mesh.get_coords(connectivity), dtype=float
    ).reshape((-1, 4, 3))
    volume_tag = mesh.tag_get_handle("Volume")
    strength_tag = mesh.tag_get_handle("Source Strength")
    volumes = np.asarray(
        mesh.tag_get_data(volume_tag, tetrahedra, flat=True), dtype=float
    )
    strengths = np.asarray(
        mesh.tag_get_data(strength_tag, tetrahedra, flat=True), dtype=float
    )
    return coordinates, volumes, strengths


def select_source_volume(
    volumes_by_id: Any, volume_id: int, expected_material: str
) -> Any:
    """Select the exact chamber volume and reject material-only ambiguity."""
    try:
        volume = volumes_by_id[int(volume_id)]
    except KeyError as exc:
        raise ValueError(
            f"DAGMC source volume {volume_id} does not exist"
        ) from exc
    material = str(getattr(volume, "material", "")).strip()
    if material != str(expected_material):
        raise ValueError(
            f"DAGMC source volume {volume_id} material {material!r} does not "
            f"match {expected_material!r}"
        )
    return volume


class _ClosedSurface:
    """Small backend-neutral closed-surface point classifier."""

    def __init__(self, triangles: np.ndarray):
        self.backend = "pyvista"
        try:
            import pyvista as pv

            points = triangles.reshape((-1, 3))
            faces = np.column_stack(
                (
                    np.full(len(triangles), 3, dtype=np.int64),
                    np.arange(len(points), dtype=np.int64).reshape((-1, 3)),
                )
            ).reshape(-1)
            self.surface = pv.PolyData(points, faces).clean(
                point_merging=True, tolerance=1.0e-8, absolute=True
            )
            self.n_open_edges = int(self.surface.n_open_edges)
            return
        except ModuleNotFoundError:
            self.backend = "vtk"

        import vtk
        from vtk.util.numpy_support import numpy_to_vtk

        points = np.asarray(triangles, dtype=float).reshape((-1, 3))
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_to_vtk(points, deep=True))
        cells = vtk.vtkCellArray()
        for start in range(0, len(points), 3):
            cell = vtk.vtkTriangle()
            for local in range(3):
                cell.GetPointIds().SetId(local, start + local)
            cells.InsertNextCell(cell)
        surface = vtk.vtkPolyData()
        surface.SetPoints(vtk_points)
        surface.SetPolys(cells)
        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(surface)
        clean.SetToleranceIsAbsolute(True)
        clean.SetAbsoluteTolerance(1.0e-8)
        clean.Update()
        self.surface = clean.GetOutput()
        edges = vtk.vtkFeatureEdges()
        edges.SetInputData(self.surface)
        edges.BoundaryEdgesOn()
        edges.FeatureEdgesOff()
        edges.ManifoldEdgesOff()
        edges.NonManifoldEdgesOff()
        edges.Update()
        self.n_open_edges = int(edges.GetOutput().GetNumberOfCells())

    def classify(
        self, points: np.ndarray, *, compute_all_distances: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(points, dtype=float).reshape((-1, 3))
        if self.backend == "pyvista":
            import pyvista as pv

            cloud = pv.PolyData(values)
            selected = np.asarray(
                cloud.select_enclosed_points(
                    self.surface, tolerance=1.0e-9, check_surface=True
                )["SelectedPoints"],
                dtype=np.uint8,
            )
            distance = np.asarray(
                cloud.compute_implicit_distance(self.surface, inplace=False)[
                    "implicit_distance"
                ],
                dtype=float,
            )
            return selected, distance

        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_to_vtk(values, deep=True))
        cloud = vtk.vtkPolyData()
        cloud.SetPoints(vtk_points)
        enclosed = vtk.vtkSelectEnclosedPoints()
        enclosed.SetInputData(cloud)
        enclosed.SetSurfaceData(self.surface)
        enclosed.SetTolerance(1.0e-9)
        enclosed.CheckSurfaceOn()
        enclosed.Update()
        selected = np.asarray(
            vtk_to_numpy(
                enclosed.GetOutput().GetPointData().GetArray("SelectedPoints")
            ),
            dtype=np.uint8,
        )
        distance = np.zeros(len(values), dtype=float)
        distance_indices = (
            np.arange(len(values), dtype=int)
            if compute_all_distances
            else np.flatnonzero(selected == 0)
        )
        if len(distance_indices):
            implicit = vtk.vtkImplicitPolyDataDistance()
            implicit.SetInput(self.surface)
            distance[distance_indices] = [
                implicit.EvaluateFunction(values[index])
                for index in distance_indices
            ]
        return selected, distance


def _source_volume_surface(
    dagmc_path: Path, volume_id: int, expected_material: str
):
    import pydagmc

    model = pydagmc.Model(str(dagmc_path))
    volume = select_source_volume(
        model.volumes_by_id, volume_id, expected_material
    )
    triangles = np.concatenate(
        [_triangles(surface.triangle_coords) for surface in volume.surfaces]
    )
    surface = _ClosedSurface(triangles)
    if surface.n_open_edges != 0:
        raise ValueError("declared source volume surface is not closed")
    return volume, surface


def audit_source_domain(
    dagmc_path: str | Path,
    source_mesh_path: str | Path,
    reference_source_mesh_path: str | Path,
    *,
    source_volume_id: int,
    source_component: str = "chamber",
    source_material: str = "Vacuum",
    source_strength_relative_tolerance: float = 1.0e-12,
    point_chunk_size: int = 10000,
    boundary_tolerance_cm: float = 1.0e-6,
) -> dict[str, Any]:
    """Audit every source tet, its vertices, and five integration points."""
    dagmc = Path(dagmc_path).resolve()
    source_mesh = Path(source_mesh_path).resolve()
    reference_source = Path(reference_source_mesh_path).resolve()
    tetrahedra, volumes, strengths = _source_arrays(source_mesh)
    arrays = audit_source_tetrahedra_arrays(tetrahedra, volumes, strengths)
    source_identity = source_mesh_fingerprint(source_mesh)
    reference_identity = source_mesh_fingerprint(reference_source)
    denominator = max(
        abs(reference_identity["total_source_strength_n_per_s"]), 1.0
    )
    relative_source_difference = (
        abs(
            source_identity["total_source_strength_n_per_s"]
            - reference_identity["total_source_strength_n_per_s"]
        )
        / denominator
    )
    semantic_identity_pass = (
        source_identity["canonical_fingerprint"]
        == reference_identity["canonical_fingerprint"]
    )

    if source_component != "chamber":
        raise ValueError("the source component role must be chamber")
    if int(source_volume_id) <= 0:
        raise ValueError("source volume ID must be a positive integer")
    source_volume, surface = _source_volume_surface(
        dagmc, int(source_volume_id), source_material
    )
    invalid_point_count = 0
    invalid_vertex_count = 0
    invalid_quadrature_count = 0
    first_invalid_points = []
    for start in range(0, len(tetrahedra), int(point_chunk_size)):
        stop = min(start + int(point_chunk_size), len(tetrahedra))
        vertices = tetrahedra[start:stop].reshape((-1, 3))
        quadrature = np.einsum(
            "pk,nkj->npj",
            SOURCE_QUADRATURE_BARYCENTRICS,
            tetrahedra[start:stop],
        ).reshape((-1, 3))
        points = np.concatenate((vertices, quadrature))
        selected, implicit_distance = surface.classify(points)
        invalid = np.flatnonzero(
            (selected == 0)
            & (np.abs(implicit_distance) > boundary_tolerance_cm)
        )
        invalid_point_count += int(len(invalid))
        vertex_point_count = 4 * (stop - start)
        invalid_vertex_count += int((invalid < vertex_point_count).sum())
        invalid_quadrature_count += int((invalid >= vertex_point_count).sum())
        for flat_index in invalid:
            if len(first_invalid_points) >= 100:
                break
            if flat_index < vertex_point_count:
                local_tet, local_index = divmod(int(flat_index), 4)
                sample_kind = "vertex"
            else:
                local_tet, local_index = divmod(
                    int(flat_index) - vertex_point_count, 5
                )
                sample_kind = "quadrature"
            first_invalid_points.append(
                {
                    "tetrahedron_index": start + local_tet,
                    "sample_kind": sample_kind,
                    "local_sample_index": local_index,
                    "point_cm": [float(value) for value in points[flat_index]],
                }
            )

    gate = (
        arrays["tetrahedron_data_gate_pass"]
        and invalid_point_count == 0
        and semantic_identity_pass
        and relative_source_difference
        <= float(source_strength_relative_tolerance)
    )
    return {
        "schema": "parastell.source_domain_audit/v1.0.0",
        "raw_h5m_sha256": sha256_file(dagmc),
        "source_mesh_sha256": sha256_file(source_mesh),
        "reference_source_mesh_sha256": sha256_file(reference_source),
        "source_material": str(source_material),
        "source_component": source_component,
        "source_volume_id": int(source_volume.id),
        "source_volume_surface_open_edge_count": int(surface.n_open_edges),
        "source_mesh_identity": source_identity,
        "reference_source_mesh_identity": reference_identity,
        "source_mesh_semantic_identity_pass": semantic_identity_pass,
        "source_strength_relative_difference": relative_source_difference,
        "source_strength_relative_tolerance": float(
            source_strength_relative_tolerance
        ),
        "quadrature_barycentric_coordinates": (
            SOURCE_QUADRATURE_BARYCENTRICS.tolist()
        ),
        "vertex_sample_count": int(4 * len(tetrahedra)),
        "quadrature_point_count": int(5 * len(tetrahedra)),
        "boundary_tolerance_cm": float(boundary_tolerance_cm),
        "invalid_vertex_sample_count": invalid_vertex_count,
        "invalid_quadrature_point_count": invalid_quadrature_count,
        "invalid_sample_point_count": invalid_point_count,
        "first_invalid_sample_points": first_invalid_points,
        **arrays,
        "source_domain_gate_pass": gate,
    }
