"""Point-cloud-derived aperture loops for surface-anchored engineering ports."""

from __future__ import annotations

from dataclasses import dataclass

import cadquery as cq
import numpy as np
from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing


@dataclass(frozen=True)
class ApertureBoundary:
    """A named triangulated radial boundary and its expected axial location."""

    name: str
    triangles: np.ndarray
    expected_w: float
    layers: tuple[str, ...] = ()


@dataclass(frozen=True)
class ApertureLoop:
    """Corresponding clear and liner-outer loops on one radial boundary."""

    boundary_name: str
    expected_w: float
    inner_points: np.ndarray
    outer_points: np.ndarray | None
    inner_uv: np.ndarray
    outer_uv: np.ndarray | None
    intersected_layers: tuple[str, ...] = ()

    @property
    def inner_closure_error(self):
        return float(
            np.linalg.norm(self.inner_points[-1] - self.inner_points[0])
        )

    @property
    def outer_closure_error(self):
        if self.outer_points is None:
            return 0.0
        return float(
            np.linalg.norm(self.outer_points[-1] - self.outer_points[0])
        )


@dataclass(frozen=True)
class PortApertureModel:
    """Authoritative surface loops plus CadQuery solids regenerated from them."""

    port_name: str
    loops: tuple[ApertureLoop, ...]
    axis: np.ndarray
    local_reference: np.ndarray
    local_normal: np.ndarray
    anchor: np.ndarray
    geometric_tolerance: float
    inner_solid: cq.Shape
    outer_solid: cq.Shape
    boolean_cutter: cq.Shape
    boolean_segments: tuple[tuple[float, float, cq.Shape], ...]
    liner_solid: cq.Shape | None

    @property
    def loop_point_counts(self):
        return tuple(len(loop.inner_points) - 1 for loop in self.loops)

    @property
    def maximum_loop_closure_error(self):
        return max(
            max(loop.inner_closure_error, loop.outer_closure_error)
            for loop in self.loops
        )

    def recovered_dimensions(self):
        inner = np.vstack([loop.inner_uv[:-1] for loop in self.loops])
        result = {
            "inner_width": float(np.ptp(inner[:, 0])),
            "inner_height": float(np.ptp(inner[:, 1])),
        }
        if self.loops[0].outer_uv is not None:
            outer = np.vstack([loop.outer_uv[:-1] for loop in self.loops])
            result.update(
                {
                    "outer_width": float(np.ptp(outer[:, 0])),
                    "outer_height": float(np.ptp(outer[:, 1])),
                    "liner_thickness_u": float(
                        (np.ptp(outer[:, 0]) - np.ptp(inner[:, 0])) / 2.0
                    ),
                    "liner_thickness_v": float(
                        (np.ptp(outer[:, 1]) - np.ptp(inner[:, 1])) / 2.0
                    ),
                }
            )
        return result


def sample_cross_section(cross_section, liner_thickness, tolerance):
    """Return corresponding CCW inner/outer samples in the local u/v plane."""
    liner_thickness = float(liner_thickness)
    tolerance = float(tolerance)
    if tolerance <= 0.0:
        raise ValueError("Aperture geometric tolerance must be positive")
    if cross_section.shape == "circle":
        outer_radius = cross_section.radius + liner_thickness
        ratio = max(-1.0, min(1.0, 1.0 - tolerance / outer_radius))
        denominator = max(np.arccos(ratio), 1e-6)
        count = max(16, int(np.ceil(np.pi / denominator)))
        count = int(np.ceil(count / 4.0) * 4)
        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        unit = np.column_stack((np.cos(angles), np.sin(angles)))
        inner = unit * cross_section.radius
        outer = unit * outer_radius if liner_thickness > 0.0 else None
    else:
        inner = np.array(
            [
                [-cross_section.width / 2.0, -cross_section.height / 2.0],
                [cross_section.width / 2.0, -cross_section.height / 2.0],
                [cross_section.width / 2.0, cross_section.height / 2.0],
                [-cross_section.width / 2.0, cross_section.height / 2.0],
            ],
            dtype=float,
        )
        outer = (
            np.array(
                [
                    [
                        -cross_section.width / 2.0 - liner_thickness,
                        -cross_section.height / 2.0 - liner_thickness,
                    ],
                    [
                        cross_section.width / 2.0 + liner_thickness,
                        -cross_section.height / 2.0 - liner_thickness,
                    ],
                    [
                        cross_section.width / 2.0 + liner_thickness,
                        cross_section.height / 2.0 + liner_thickness,
                    ],
                    [
                        -cross_section.width / 2.0 - liner_thickness,
                        cross_section.height / 2.0 + liner_thickness,
                    ],
                ],
                dtype=float,
            )
            if liner_thickness > 0.0
            else None
        )
    return inner, outer


def line_triangle_intersections(origin, direction, triangles, tolerance=1e-9):
    """Return de-duplicated signed line parameters for a triangle cloud."""
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    triangles = np.asarray(triangles, dtype=float)
    edge_1 = triangles[:, 1] - triangles[:, 0]
    edge_2 = triangles[:, 2] - triangles[:, 0]
    h = np.cross(np.broadcast_to(direction, edge_2.shape), edge_2)
    determinant = np.einsum("ij,ij->i", edge_1, h)
    valid = np.abs(determinant) > tolerance
    inverse = np.zeros_like(determinant)
    inverse[valid] = 1.0 / determinant[valid]
    displacement = origin - triangles[:, 0]
    barycentric_u = inverse * np.einsum("ij,ij->i", displacement, h)
    q = np.cross(displacement, edge_1)
    barycentric_v = inverse * np.einsum("j,ij->i", direction, q)
    parameter = inverse * np.einsum("ij,ij->i", edge_2, q)
    valid &= barycentric_u >= -tolerance
    valid &= barycentric_v >= -tolerance
    valid &= barycentric_u + barycentric_v <= 1.0 + tolerance
    values = sorted(float(value) for value in parameter[valid])
    unique = []
    for value in values:
        if not unique or abs(value - unique[-1]) > 1e-6:
            unique.append(value)
    return np.asarray(unique, dtype=float)


def _polygon_signed_area(points):
    points = np.asarray(points, dtype=float)
    return float(
        0.5
        * np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - points[:, 1] * np.roll(points[:, 0], -1)
        )
    )


def _orientation(a, b, c):
    left = b - a
    right = c - a
    return float(left[0] * right[1] - left[1] * right[0])


def _self_intersects(points, tolerance=1e-12):
    points = np.asarray(points, dtype=float)
    count = len(points)
    for left in range(count):
        a = points[left]
        b = points[(left + 1) % count]
        for right in range(left + 1, count):
            if right in {left, (left + 1) % count}:
                continue
            if left == 0 and right == count - 1:
                continue
            c = points[right]
            d = points[(right + 1) % count]
            o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
            o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
            if o1 * o2 < -tolerance and o3 * o4 < -tolerance:
                return True
    return False


def _validate_uv_loop(points, name, tolerance):
    if len(points) < 3:
        raise ValueError(f"Aperture loop {name!r} is disconnected")
    edges = np.roll(points, -1, axis=0) - points
    if np.any(np.linalg.norm(edges, axis=1) <= tolerance * 1e-6):
        raise ValueError(f"Aperture loop {name!r} has degenerate edges")
    if _polygon_signed_area(points) <= 0.0:
        raise ValueError(f"Aperture loop {name!r} has reversed ordering")
    if _self_intersects(points):
        raise ValueError(f"Aperture loop {name!r} is self-intersecting")


def _intersect_uv_loop(
    uv_points,
    boundary,
    anchor,
    axis,
    local_reference,
    local_normal,
    search_window,
):
    points = []
    for u_value, v_value in uv_points:
        line_origin = (
            anchor + u_value * local_reference + v_value * local_normal
        )
        candidates = line_triangle_intersections(
            line_origin, axis, boundary.triangles
        )
        nearby = candidates[
            np.abs(candidates - boundary.expected_w) <= search_window
        ]
        if len(nearby) == 0:
            raise ValueError(
                f"Aperture sample has no intersection with {boundary.name!r}"
            )
        if len(nearby) > 1:
            raise ValueError(
                f"Aperture sample has multiple far-side intersections with "
                f"{boundary.name!r}"
            )
        points.append(line_origin + nearby[0] * axis)
    points = np.asarray(points, dtype=float)
    return np.vstack((points, points[0]))


def _wire(points):
    vectors = [cq.Vector(*point) for point in np.asarray(points)[:-1]]
    return cq.Wire.makePolygon(vectors, close=True)


def _loft(loop_points, description):
    wires = [_wire(points) for points in loop_points]
    last_error = None
    try:
        segments = [
            cq.Solid.makeLoft([left, right], True)
            for left, right in zip(wires, wires[1:])
        ]
        solid = segments[0]
        for segment in segments[1:]:
            solid = solid.fuse(segment, glue=True, tol=1e-7)
        for candidate in (solid, solid.clean(), solid.fix()):
            if (
                candidate is not None
                and candidate.isValid()
                and candidate.Volume() > 0.0
            ):
                return candidate
    except Exception as error:
        last_error = error
    for ruled in (False, True):
        try:
            solid = cq.Solid.makeLoft(wires, ruled)
        except Exception as error:
            last_error = error
            continue
        for candidate in (solid, solid.clean(), solid.fix()):
            if (
                candidate is not None
                and candidate.isValid()
                and candidate.Volume() > 0.0
            ):
                return candidate
    raise ValueError(
        f"Could not loft valid {description} aperture loops"
    ) from last_error


def _triangle_face(first, second, third):
    wire = cq.Wire.makePolygon(
        [cq.Vector(*first), cq.Vector(*second), cq.Vector(*third)], close=True
    )
    return cq.Face.makeFromWires(wire)


def _solid_from_faces(faces, description):
    sewing = BRepBuilderAPI_Sewing(1e-6)
    for face in faces:
        sewing.Add(face.wrapped)
    sewing.Perform()
    sewn = cq.Shape.cast(sewing.SewedShape())
    shells = sewn.Shells()
    if sewn.ShapeType() == "Shell":
        shell = sewn
    elif len(shells) == 1:
        shell = shells[0]
    else:
        raise ValueError(f"Faceted {description} did not sew into one shell")
    solid = cq.Solid.makeSolid(shell)
    for candidate in (solid, solid.fix(), solid.clean().fix()):
        if candidate.isValid() and candidate.Volume() > 0.0:
            return candidate
    raise ValueError(
        f"Faceted {description} is not a valid solid: "
        f"sewn_type={sewn.ShapeType()}, shells={len(shells)}, "
        f"closed={shell.Closed()}, volume={solid.Volume()}"
    )


def _faceted_tube(loop_points, description):
    loops = [np.asarray(points, dtype=float)[:-1] for points in loop_points]
    count = len(loops[0])
    if any(len(loop) != count for loop in loops):
        raise ValueError(
            f"{description} loops do not have corresponding samples"
        )
    faces = []
    for left, right in zip(loops, loops[1:]):
        for index in range(count):
            next_index = (index + 1) % count
            faces.append(
                _triangle_face(
                    left[index], left[next_index], right[next_index]
                )
            )
            faces.append(
                _triangle_face(left[index], right[next_index], right[index])
            )
    first_center = loops[0].mean(axis=0)
    last_center = loops[-1].mean(axis=0)
    for index in range(count):
        next_index = (index + 1) % count
        faces.append(
            _triangle_face(first_center, loops[0][next_index], loops[0][index])
        )
        faces.append(
            _triangle_face(
                last_center, loops[-1][index], loops[-1][next_index]
            )
        )
    try:
        return _solid_from_faces(faces, description)
    except ValueError:
        # OCC can reject an otherwise closed triangulated shell when several
        # adjacent facets are almost coplanar. A loft through the same ordered
        # loops is an equivalent visualization/STEP regeneration of the
        # authoritative loop representation.
        return _loft(loop_points, description)


def _faceted_liner(inner_loop_points, outer_loop_points, description):
    inner = [
        np.asarray(points, dtype=float)[:-1] for points in inner_loop_points
    ]
    outer = [
        np.asarray(points, dtype=float)[:-1] for points in outer_loop_points
    ]
    count = len(inner[0])
    if any(len(loop) != count for loop in (*inner, *outer)):
        raise ValueError(
            f"{description} loops do not have corresponding samples"
        )
    faces = []
    for inner_left, inner_right, outer_left, outer_right in zip(
        inner, inner[1:], outer, outer[1:]
    ):
        for index in range(count):
            next_index = (index + 1) % count
            faces.extend(
                [
                    _triangle_face(
                        outer_left[index],
                        outer_left[next_index],
                        outer_right[next_index],
                    ),
                    _triangle_face(
                        outer_left[index],
                        outer_right[next_index],
                        outer_right[index],
                    ),
                    _triangle_face(
                        inner_left[index],
                        inner_right[index],
                        inner_right[next_index],
                    ),
                    _triangle_face(
                        inner_left[index],
                        inner_right[next_index],
                        inner_left[next_index],
                    ),
                ]
            )
    for inner_cap, outer_cap, reverse in (
        (inner[0], outer[0], False),
        (inner[-1], outer[-1], True),
    ):
        for index in range(count):
            next_index = (index + 1) % count
            if reverse:
                faces.extend(
                    [
                        _triangle_face(
                            outer_cap[index],
                            outer_cap[next_index],
                            inner_cap[next_index],
                        ),
                        _triangle_face(
                            outer_cap[index],
                            inner_cap[next_index],
                            inner_cap[index],
                        ),
                    ]
                )
            else:
                faces.extend(
                    [
                        _triangle_face(
                            outer_cap[index],
                            inner_cap[index],
                            inner_cap[next_index],
                        ),
                        _triangle_face(
                            outer_cap[index],
                            inner_cap[next_index],
                            outer_cap[next_index],
                        ),
                    ]
                )
    try:
        return _solid_from_faces(faces, description)
    except ValueError:
        outer_solid = _loft(outer_loop_points, f"{description} outer")
        inner_solid = _loft(inner_loop_points, f"{description} inner")
        liner = outer_solid.cut(inner_solid)
        for candidate in (liner, liner.clean(), liner.fix()):
            if candidate.isValid() and candidate.Volume() > 0.0:
                return candidate
        raise ValueError(
            f"Could not regenerate valid {description} from loops"
        )


def _prismatic_regeneration(
    uv_points, anchor, axis, local_reference, start_w, end_w, description
):
    """Regenerate a stable CAD comparison body from authoritative loop data."""
    plane = cq.Plane(
        origin=tuple(anchor + start_w * axis),
        xDir=tuple(local_reference),
        normal=tuple(axis),
    )
    solid = (
        cq.Workplane(plane)
        .polyline([tuple(point) for point in uv_points])
        .close()
        .extrude(end_w - start_w)
        .val()
    )
    if solid is None or not solid.isValid() or solid.Volume() <= 0.0:
        raise ValueError(
            f"Could not regenerate valid {description} from loops"
        )
    return solid


def build_aperture_loops(
    port,
    boundaries,
    geometric_tolerance=0.05,
):
    """Intersect corresponding rays without creating any CAD geometry."""
    if not boundaries:
        raise ValueError(f"Port {port.name!r} has no radial boundaries")
    anchor = np.asarray(port.placement.anchor, dtype=float)
    axis = np.asarray(port.placement.local_axis, dtype=float)
    local_reference = np.asarray(port.placement.local_reference, dtype=float)
    local_normal = np.asarray(port.placement.local_normal, dtype=float)
    liner_thickness = port.liner.thickness if port.liner.enabled else 0.0
    inner_uv, outer_uv = sample_cross_section(
        port.cross_section, liner_thickness, geometric_tolerance
    )
    _validate_uv_loop(inner_uv, f"{port.name} inner", geometric_tolerance)
    if outer_uv is not None:
        _validate_uv_loop(outer_uv, f"{port.name} outer", geometric_tolerance)
    maximum_dimension = float(
        max(np.ptp(inner_uv[:, 0]), np.ptp(inner_uv[:, 1]))
        + 2.0 * liner_thickness
    )
    search_window = max(5.0, 4.0 * maximum_dimension)
    loops = []
    previous_mean_w = -np.inf
    for boundary in boundaries:
        inner_points = _intersect_uv_loop(
            inner_uv,
            boundary,
            anchor,
            axis,
            local_reference,
            local_normal,
            search_window,
        )
        outer_points = (
            _intersect_uv_loop(
                outer_uv,
                boundary,
                anchor,
                axis,
                local_reference,
                local_normal,
                search_window,
            )
            if outer_uv is not None
            else None
        )
        mean_w = float(np.mean(np.dot(inner_points[:-1] - anchor, axis)))
        if mean_w <= previous_mean_w + geometric_tolerance * 1e-4:
            raise ValueError(
                f"Aperture boundary {boundary.name!r} is not ordered outward"
            )
        previous_mean_w = mean_w
        loops.append(
            ApertureLoop(
                boundary.name,
                boundary.expected_w,
                inner_points,
                outer_points,
                np.vstack((inner_uv, inner_uv[0])),
                (
                    np.vstack((outer_uv, outer_uv[0]))
                    if outer_uv is not None
                    else None
                ),
                boundary.layers,
            )
        )

    if port.extent.outer_extension > 0.0:
        last = loops[-1]
        translation = axis * port.extent.outer_extension
        loops.append(
            ApertureLoop(
                "outer_extension_end",
                last.expected_w + port.extent.outer_extension,
                last.inner_points + translation,
                (
                    last.outer_points + translation
                    if last.outer_points is not None
                    else None
                ),
                last.inner_uv.copy(),
                last.outer_uv.copy() if last.outer_uv is not None else None,
                (),
            )
        )

    return tuple(loops)


def build_aperture_model(
    port,
    boundaries,
    geometric_tolerance=0.05,
):
    """Regenerate optional CAD comparison solids from authoritative loops."""
    loops = build_aperture_loops(port, boundaries, geometric_tolerance)
    anchor = np.asarray(port.placement.anchor, dtype=float)
    axis = np.asarray(port.placement.local_axis, dtype=float)
    local_reference = np.asarray(port.placement.local_reference, dtype=float)
    local_normal = np.asarray(port.placement.local_normal, dtype=float)
    inner_uv = loops[0].inner_uv[:-1]
    outer_uv = (
        loops[0].outer_uv[:-1] if loops[0].outer_uv is not None else None
    )

    all_points = [
        (
            loop.outer_points
            if loop.outer_points is not None
            else loop.inner_points
        )
        for loop in loops
    ]
    start_w = float(np.min((all_points[0][:-1] - anchor) @ axis))
    end_w = float(np.max((all_points[-1][:-1] - anchor) @ axis))
    inner_solid = _prismatic_regeneration(
        inner_uv,
        anchor,
        axis,
        local_reference,
        start_w,
        end_w,
        f"{port.name} clear",
    )
    if outer_uv is None:
        outer_solid = inner_solid
        liner_solid = None
    else:
        outer_solid = _prismatic_regeneration(
            outer_uv,
            anchor,
            axis,
            local_reference,
            start_w,
            end_w,
            f"{port.name} outer",
        )
        liner_solid = outer_solid.cut(inner_solid)
        if not liner_solid.isValid() or liner_solid.Volume() <= 0.0:
            raise ValueError(
                f"Could not regenerate valid {port.name} liner from loops"
            )

    # Boolean caps are deliberately placed well beyond each curved radial
    # boundary. The cutter is applied only to its corresponding blanket layer,
    # so this axial guard cannot enlarge the aperture within that layer.
    guard = max(0.5, geometric_tolerance * 2.0)
    boolean_segments = []
    cutter_uv = outer_uv if outer_uv is not None else inner_uv
    for index, (left, right) in enumerate(zip(loops, loops[1:])):
        left_points = (
            left.outer_points
            if left.outer_points is not None
            else left.inner_points
        )
        right_points = (
            right.outer_points
            if right.outer_points is not None
            else right.inner_points
        )
        start_guard = guard
        if index == 0 and port.extent.start.reference == "layer":
            start_guard = 0.0
        end_guard = guard
        last_physical_segment = index == len(loops) - 2
        fractional_blind_end = (
            port.extent.end.reference == "layer"
            and port.extent.end.fraction not in {0.0, 1.0}
            and port.extent.outer_extension == 0.0
        )
        if last_physical_segment and fractional_blind_end:
            end_guard = 0.0
        start_w = (
            float(np.min((left_points[:-1] - anchor) @ axis)) - start_guard
        )
        end_w = float(np.max((right_points[:-1] - anchor) @ axis)) + end_guard
        plane = cq.Plane(
            origin=tuple(anchor + start_w * axis),
            xDir=tuple(local_reference),
            normal=tuple(axis),
        )
        segment = (
            cq.Workplane(plane)
            .polyline([tuple(point) for point in cutter_uv])
            .close()
            .extrude(end_w - start_w)
            .val()
        )
        if segment is None or not segment.isValid() or segment.Volume() <= 0.0:
            raise ValueError(
                f"Port {port.name!r} Boolean segment {index} is invalid"
            )
        boolean_segments.append((left.expected_w, right.expected_w, segment))
    boolean_cutter = boolean_segments[0][2]
    for _, _, segment in boolean_segments[1:]:
        boolean_cutter = boolean_cutter.fuse(segment)
    return PortApertureModel(
        port.name,
        tuple(loops),
        axis,
        local_reference,
        local_normal,
        anchor,
        float(geometric_tolerance),
        inner_solid,
        outer_solid,
        boolean_cutter,
        tuple(boolean_segments),
        liner_solid,
    )
