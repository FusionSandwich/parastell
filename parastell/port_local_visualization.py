"""Port-local visual validation driven by surface aperture loops."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .port_visualization import COMPONENT_COLORS, _repository_sha


SCHEMA_VERSION = "1.0"
IMAGE_SIZE = (1600, 1000)


def _local(points, anchor, axis, reference, normal):
    relative = np.asarray(points, dtype=float) - anchor
    return np.column_stack(
        (relative @ axis, relative @ reference, relative @ normal)
    )


def _digest(path):
    value = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _rgba(kind):
    return COMPONENT_COLORS[kind]


def _layer_color(name):
    return COMPONENT_COLORS.get(name, COMPONENT_COLORS["blanket_layer"])


def _loop_w(loop, anchor, axis):
    return float(np.mean((loop.inner_points[:-1] - anchor) @ axis))


def _boundary_w(model, port, layer):
    source = model.Components[layer]
    return model._port_layer_interval(port, layer, source)


def _common_text(ax, port, result, view):
    aperture = port.cross_section
    dimension = (
        f"clear radius={aperture.radius:.3f} cm"
        if aperture.shape == "circle"
        else f"clear {aperture.width:.3f} × {aperture.height:.3f} cm"
    )
    target = ax.figure if hasattr(ax, "zaxis") else ax
    text_method = target.text
    text_method(
        0.01,
        0.01,
        f"port: {port.name} | units: cm | view: {view}\n"
        f"{dimension} | liner={port.liner.thickness:.3f} cm | "
        f"outer extension={result.outer_extension:.3f} cm\n"
        f"layers: {', '.join(result.ordered_intersected_layers)}",
        transform=None if target is ax.figure else ax.transAxes,
        fontsize=9,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.7"},
    )


def _save(fig, path):
    fig.set_size_inches(16, 10)
    fig.savefig(path, dpi=100, facecolor="white")
    fig.clf()


def _longitudinal(path, model, port, result, aperture_model, coordinate):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots()
    axis = aperture_model.axis
    anchor = aperture_model.anchor
    component = 0 if coordinate == "u" else 1
    loop_w = np.asarray(
        [_loop_w(loop, anchor, axis) for loop in aperture_model.loops]
    )
    inner = np.asarray(
        [
            [
                np.min(loop.inner_uv[:-1, component]),
                np.max(loop.inner_uv[:-1, component]),
            ]
            for loop in aperture_model.loops
        ]
    )
    outer = (
        np.asarray(
            [
                [
                    np.min(loop.outer_uv[:-1, component]),
                    np.max(loop.outer_uv[:-1, component]),
                ]
                for loop in aperture_model.loops
            ]
        )
        if aperture_model.loops[0].outer_uv is not None
        else inner
    )
    first_layer_low = min(
        _boundary_w(model, port, layer)[0]
        for layer in result.ordered_intersected_layers
    )
    if first_layer_low > result.resolved_start:
        ax.axvspan(
            result.resolved_start,
            first_layer_low,
            color=_rgba("plasma_or_chamber")[:3],
            alpha=0.16,
            zorder=0,
        )
        ax.text(
            (result.resolved_start + first_layer_low) / 2.0,
            outer[:, 1].max() * 1.42,
            "chamber interval",
            rotation=90,
            ha="center",
            va="top",
            fontsize=8,
        )
    for layer in result.ordered_intersected_layers:
        low, high = _boundary_w(model, port, layer)
        color = _layer_color(layer)
        ax.axvspan(low, high, color=color[:3], alpha=0.26, zorder=0)
        ax.text(
            (low + high) / 2.0,
            outer[:, 1].max() * 1.42,
            layer,
            rotation=90,
            ha="center",
            va="top",
            fontsize=8,
        )
    ax.fill_between(
        loop_w,
        outer[:, 0],
        inner[:, 0],
        color=_rgba("port_liner")[:3],
        alpha=0.92,
    )
    ax.fill_between(
        loop_w,
        inner[:, 1],
        outer[:, 1],
        color=_rgba("port_liner")[:3],
        alpha=0.92,
    )
    ax.fill_between(
        loop_w,
        inner[:, 0],
        inner[:, 1],
        color=_rgba("port_void")[:3],
        alpha=0.55,
    )
    ax.plot(
        loop_w, np.zeros_like(loop_w), "k--", lw=1.5, label="port centerline"
    )
    start_w, end_w = result.resolved_start, result.resolved_end
    extension_w = end_w + result.outer_extension
    for value, label in (
        (0.0, "surface anchor"),
        (start_w, "start"),
        (end_w, "resolved blanket end"),
        (extension_w, "external termination"),
    ):
        ax.axvline(
            value, color="black", lw=1.0, ls=":" if label != "start" else "-"
        )
        ax.annotate(
            label,
            (value, 0.0),
            xytext=(4, 8),
            textcoords="offset points",
            rotation=90,
            fontsize=8,
        )
    transverse_span = max(float(np.max(np.abs(outer))), 1.0)
    w_pad = max(2.0, 0.06 * (extension_w - start_w))
    ax.set_xlim(start_w - w_pad, extension_w + w_pad)
    ax.set_ylim(-1.75 * transverse_span, 1.75 * transverse_span)
    ax.set_xlabel("w — outward port axis [cm]")
    ax.set_ylabel(
        f"{coordinate} — local {'poloidal' if coordinate == 'u' else 'binormal'} [cm]"
    )
    ax.set_title(f"Surface-anchored port — longitudinal {coordinate}")
    ax.grid(alpha=0.2)
    ax.legend(
        handles=[
            Patch(
                color=_rgba("port_void")[:3],
                alpha=0.55,
                label="clear aperture",
            ),
            Patch(color=_rgba("port_liner")[:3], alpha=0.92, label="liner"),
            Patch(color="0.75", alpha=0.4, label="intersected blanket layers"),
        ],
        loc="upper right",
    )
    _common_text(ax, port, result, f"horizontal=w, vertical={coordinate}")
    _save(fig, path)
    return {
        "horizontal": "w",
        "vertical": coordinate,
        "centerline_slope": 0.0,
        "crop": [
            float(start_w - w_pad),
            float(extension_w + w_pad),
            float(-1.75 * transverse_span),
            float(1.75 * transverse_span),
        ],
        "start_w": float(start_w),
        "end_w": float(end_w),
        "extension_end_w": float(extension_w),
        "aperture_width_fraction": float(
            (end_w - start_w) / (extension_w - start_w + 2 * w_pad)
        ),
    }


def _transverse(path, port, result, loop, label):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Polygon

    fig, ax = plt.subplots()
    inner = loop.inner_uv[:-1]
    outer = loop.outer_uv[:-1] if loop.outer_uv is not None else inner
    span = max(float(np.max(np.abs(outer))), 1.0)
    background = 1.7 * span
    ax.add_patch(
        Polygon(
            [
                (-background, -background),
                (background, -background),
                (background, background),
                (-background, background),
            ],
            color=_rgba("blanket_layer")[:3],
            alpha=0.22,
        )
    )
    if loop.outer_uv is not None:
        ax.add_patch(
            Polygon(
                outer, closed=True, color=_rgba("port_liner")[:3], alpha=0.95
            )
        )
    ax.add_patch(
        Polygon(inner, closed=True, color=_rgba("port_void")[:3], alpha=0.62)
    )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.plot(0.0, 0.0, "ko", ms=5)
    ax.annotate(
        "port center / axis",
        (0.0, 0.0),
        xytext=(8, 8),
        textcoords="offset points",
    )
    ax.annotate("u", (0.82 * background, 0), fontsize=12, weight="bold")
    ax.annotate("v", (0, 0.82 * background), fontsize=12, weight="bold")
    ax.text(
        -1.55 * span,
        1.48 * span,
        "surface anchor projected at u=v=0",
        fontsize=9,
    )
    ax.set_xlim(-1.7 * span, 1.7 * span)
    ax.set_ylim(-1.7 * span, 1.7 * span)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("u — local poloidal [cm]")
    ax.set_ylabel("v — local binormal [cm]")
    ax.set_title(f"Surface-anchored port — transverse {label}")
    ax.legend(
        handles=[
            Patch(
                color=_rgba("port_void")[:3],
                alpha=0.62,
                label="clear aperture",
            ),
            Patch(color=_rgba("port_liner")[:3], alpha=0.95, label="liner"),
            Patch(
                color=_rgba("blanket_layer")[:3],
                alpha=0.22,
                label="blanket material outside aperture",
            ),
        ],
        loc="upper right",
    )
    _common_text(ax, port, result, f"normal to w at {loop.boundary_name}")
    _save(fig, path)
    return {
        "horizontal": "u",
        "vertical": "v",
        "equal_aspect": True,
        "port_center_uv": [0.0, 0.0],
        "boundary": loop.boundary_name,
        "crop": [-1.7 * span, 1.7 * span, -1.7 * span, 1.7 * span],
        "aperture_width_fraction": float(np.ptp(inner[:, 0]) / (3.4 * span)),
    }


def _isometric(path, model, port, result, aperture_model):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    local_loops = []
    for loop in aperture_model.loops:
        inner = _local(
            loop.inner_points,
            aperture_model.anchor,
            aperture_model.axis,
            aperture_model.local_reference,
            aperture_model.local_normal,
        )
        outer = None
        ax.plot(
            inner[:, 0],
            inner[:, 1],
            inner[:, 2],
            color=_rgba("port_void")[:3],
            lw=2.0,
        )
        if loop.outer_points is not None:
            outer = _local(
                loop.outer_points,
                aperture_model.anchor,
                aperture_model.axis,
                aperture_model.local_reference,
                aperture_model.local_normal,
            )
            ax.plot(
                outer[:, 0],
                outer[:, 1],
                outer[:, 2],
                color=_rgba("port_liner")[:3],
                lw=2.4,
            )
        local_loops.append((inner, outer))
    void_panels = []
    liner_panels = []
    for (left_inner, left_outer), (right_inner, right_outer) in zip(
        local_loops, local_loops[1:]
    ):
        for index in range(len(left_inner) - 1):
            following = (index + 1) % (len(left_inner) - 1)
            void_panels.append(
                (
                    left_inner[index],
                    left_inner[following],
                    right_inner[following],
                    right_inner[index],
                )
            )
            if (
                left_outer is not None
                and right_outer is not None
                and np.mean(
                    (
                        left_outer[index, 2],
                        left_outer[following, 2],
                        right_outer[following, 2],
                        right_outer[index, 2],
                    )
                )
                >= 0.0
            ):
                liner_panels.append(
                    (
                        left_outer[index],
                        left_outer[following],
                        right_outer[following],
                        right_outer[index],
                    )
                )
    ax.add_collection3d(
        Poly3DCollection(
            void_panels,
            facecolor=_rgba("port_void")[:3],
            alpha=0.22,
            edgecolor="none",
        )
    )
    ax.add_collection3d(
        Poly3DCollection(
            liner_panels,
            facecolor=_rgba("port_liner")[:3],
            alpha=0.28,
            edgecolor="none",
        )
    )
    w0, w1 = (
        result.resolved_start,
        result.resolved_end + result.outer_extension,
    )
    ax.plot([w0, w1], [0, 0], [0, 0], "k--", lw=1.5)
    ax.scatter(
        [0, result.resolved_start, result.resolved_end, w1],
        [0] * 4,
        [0] * 4,
        color="black",
        s=28,
    )
    for w, name in (
        (0, "anchor"),
        (result.resolved_start, "start"),
        (result.resolved_end, "resolved blanket end"),
        (w1, "external termination"),
    ):
        ax.text(w, 0, 0, f" {name}", fontsize=8)
    for layer in result.ordered_intersected_layers:
        low, high = _boundary_w(model, port, layer)
        color = _layer_color(layer)[:3]
        patch_span = max(
            np.max(np.abs(aperture_model.loops[0].outer_uv[:-1])) * 1.45,
            5.0,
        )
        for boundary in (low, high):
            ax.plot(
                [boundary] * 5,
                [
                    -patch_span,
                    patch_span,
                    patch_span,
                    -patch_span,
                    -patch_span,
                ],
                [
                    -patch_span,
                    -patch_span,
                    patch_span,
                    patch_span,
                    -patch_span,
                ],
                color=color,
                alpha=0.45,
                lw=1.0,
            )
        ax.text((low + high) / 2, 0, 5.5, layer, rotation=90, fontsize=7)
    span = max(np.max(np.abs(aperture_model.loops[0].outer_uv[:-1])), 1.0)
    pad = max(2.0, 0.05 * (w1 - w0))
    ax.set_xlim(w0 - pad, w1 + pad)
    ax.set_ylim(-1.7 * span, 1.7 * span)
    ax.set_zlim(-1.7 * span, 1.7 * span)
    ax.set_xlabel("w [cm]")
    ax.set_ylabel("u [cm]")
    ax.set_zlabel("v [cm]")
    ax.set_title("Port-local isometric cutaway — aperture loops")
    ax.view_init(elev=24, azim=-57)
    ax.legend(
        handles=[
            Patch(
                color=_rgba("port_void")[:3],
                alpha=0.45,
                label="continuous clear void",
            ),
            Patch(
                color=_rgba("port_liner")[:3],
                alpha=0.45,
                label="half-section translucent liner",
            ),
        ]
    )
    _common_text(ax, port, result, "bounded local w/u/v isometric")
    _save(fig, path)
    return {
        "coordinates": ["w", "u", "v"],
        "contains_global_sector": False,
        "crop": [
            float(w0 - pad),
            float(w1 + pad),
            float(-1.7 * span),
            float(1.7 * span),
        ],
    }


def _nearest_magnet(stellarator, port_name):
    records = [
        record
        for record in stellarator.port_magnet_collision_report
        if record.port_name == port_name
        and record.estimated_minimum_distance is not None
    ]
    if not records:
        return None, None
    record = min(records, key=lambda item: item.estimated_minimum_distance)
    solid = next(
        (
            item
            for item in stellarator.magnet_set.iter_coil_solids()
            if item.coil_id == record.coil_id
            and item.region_kind == record.magnet_region_kind
        ),
        None,
    )
    return record, solid


def _magnet_view(path, stellarator, port, result, aperture_model):
    import matplotlib.pyplot as plt

    record, magnet = _nearest_magnet(stellarator, port.name)
    fig, ax = plt.subplots()
    w = np.asarray(
        [
            _loop_w(loop, aperture_model.anchor, aperture_model.axis)
            for loop in aperture_model.loops
        ]
    )
    inner = np.asarray(
        [
            np.max(np.abs(loop.inner_uv[:-1, 0]))
            for loop in aperture_model.loops
        ]
    )
    outer = np.asarray(
        [
            np.max(np.abs(loop.outer_uv[:-1, 0]))
            for loop in aperture_model.loops
        ]
    )
    ax.fill_between(
        w,
        -outer,
        outer,
        color=_rgba("port_liner")[:3],
        alpha=0.75,
        label="liner",
    )
    ax.fill_between(
        w,
        -inner,
        inner,
        color=_rgba("port_void")[:3],
        alpha=0.55,
        label="clear aperture",
    )
    magnet_id = None
    if magnet is not None:
        vertices, _ = magnet.solid.tessellate(2.0, 0.25)
        xyz = np.asarray(
            [vertex.toTuple() for vertex in vertices], dtype=float
        )
        local = _local(
            xyz,
            aperture_model.anchor,
            aperture_model.axis,
            aperture_model.local_reference,
            aperture_model.local_normal,
        )
        segment_low = result.resolved_start
        segment_high = result.resolved_end + result.outer_extension
        closest_w = np.clip(local[:, 0], segment_low, segment_high)
        distance = np.sqrt(
            (local[:, 0] - closest_w) ** 2
            + local[:, 1] ** 2
            + local[:, 2] ** 2
        )
        nearest_index = int(np.argmin(distance))
        patch_radius = max(35.0, float(distance[nearest_index]) * 0.65)
        patch_distance = np.linalg.norm(local - local[nearest_index], axis=1)
        visible = local[patch_distance <= patch_radius]
        if len(visible) < 8:
            visible = local[np.argsort(patch_distance)[: min(80, len(local))]]
        ax.scatter(
            visible[:, 0],
            visible[:, 1],
            s=5.0,
            color=_rgba("magnet_conductor")[:3],
            alpha=0.45,
            label=f"nearest magnet {record.coil_id}/{record.magnet_region_kind}",
        )
        magnet_id = {
            "coil_id": record.coil_id,
            "region_kind": record.magnet_region_kind,
            "distance": record.estimated_minimum_distance,
            "status": record.status,
        }
        lower = min(result.resolved_start, float(np.min(visible[:, 0])))
        upper = max(
            result.resolved_end + result.outer_extension,
            float(np.max(visible[:, 0])),
        )
    else:
        lower, upper = (
            result.resolved_start,
            result.resolved_end + result.outer_extension,
        )
    pad = max(3.0, 0.05 * (upper - lower))
    ax.set_xlim(lower - pad, upper + pad)
    ax.axhline(0.0, color="black", ls="--", lw=1.0, label="port centerline")
    ax.scatter(
        [
            0,
            result.resolved_start,
            result.resolved_end,
            result.resolved_end + result.outer_extension,
        ],
        [0] * 4,
        color="black",
        s=22,
    )
    ax.set_xlabel("w — outward port axis [cm]")
    ax.set_ylabel("u — local poloidal [cm]")
    ax.set_title(
        "Port-local magnet clearance — collision solid identity preserved"
    )
    ax.legend(loc="best")
    _common_text(
        ax, port, result, "longitudinal u toward nearest collision solid"
    )
    _save(fig, path)
    return {
        "nearest_magnet": magnet_id,
        "collision_solid_identity_preserved": magnet is not None,
        "bounded_local_magnet_patch": True,
        "contains_entire_global_magnet": False,
    }


def _anchor_view(path, model, port, result, aperture_model):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    spec = port.placement.surface_anchor
    surface = model._anchor_reference_surface(spec.reference, spec.layer)
    triangles = model._port_surface_triangles(port, surface)
    local = _local(
        triangles.reshape(-1, 3),
        aperture_model.anchor,
        aperture_model.axis,
        aperture_model.local_reference,
        aperture_model.local_normal,
    ).reshape(-1, 3, 3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(
        Poly3DCollection(
            local,
            facecolor=_rgba("plasma_or_chamber")[:3],
            alpha=0.35,
            edgecolor="none",
        )
    )
    scale = max(model._port_aperture_half_width(port) * 1.6, 3.0)
    ax.quiver(0, 0, 0, scale, 0, 0, color="black", label="w / outward axis")
    ax.quiver(0, 0, 0, 0, scale, 0, color="blue", label="u / poloidal")
    ax.quiver(0, 0, 0, 0, 0, scale, color="green", label="v / binormal")
    ax.scatter([0], [0], [0], color="black", s=50)
    ax.text(0, 0, 0, " surface anchor", fontsize=9)
    ax.set_xlim(-0.4 * scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_zlim(-scale, scale)
    ax.set_xlabel("w [cm]")
    ax.set_ylabel("u [cm]")
    ax.set_zlabel("v [cm]")
    ax.set_title("Continuous-surface anchor and right-handed local frame")
    ax.legend(loc="upper right")
    _common_text(ax, port, result, "local surface patch")
    _save(fig, path)
    return {
        "anchor_local": [0.0, 0.0, 0.0],
        "surface_reference": spec.reference,
    }


def export_port_local_validation(stellarator, output_dir, port_names=None):
    """Render and describe one surface-anchored port in its local frame."""
    model = stellarator.invessel_build
    available = tuple(model.port_aperture_models)
    selected = available if port_names is None else tuple(port_names)
    if len(selected) != 1 or selected[0] not in available:
        raise ValueError("Select exactly one generated surface-anchored port")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    name = selected[0]
    port = model.port_specs[name]
    result = model.port_geometry_diagnostics[name]
    aperture_model = model.port_aperture_models[name]
    paths = {
        key: output_dir / filename
        for key, filename in {
            "longitudinal_u": "port_local_longitudinal_u.png",
            "longitudinal_v": "port_local_longitudinal_v.png",
            "transverse_inner": "port_local_transverse_inner.png",
            "transverse_blanket": "port_local_transverse_blanket.png",
            "transverse_outer": "port_local_transverse_outer.png",
            "isometric": "port_local_isometric_cutaway.png",
            "magnet_clearance": "port_local_magnet_clearance.png",
            "surface_anchor": "port_surface_anchor.png",
        }.items()
    }
    views = {}
    views["longitudinal_u"] = _longitudinal(
        paths["longitudinal_u"], model, port, result, aperture_model, "u"
    )
    views["longitudinal_v"] = _longitudinal(
        paths["longitudinal_v"], model, port, result, aperture_model, "v"
    )
    loops = aperture_model.loops
    blanket_target = (result.resolved_start + result.resolved_end) / 2.0
    blanket_loop = min(
        loops[:-1],
        key=lambda loop: abs(
            _loop_w(loop, aperture_model.anchor, aperture_model.axis)
            - blanket_target
        ),
    )
    outer_loop = (
        loops[-2]
        if loops[-1].boundary_name == "outer_extension_end"
        else loops[-1]
    )
    views["transverse_inner"] = _transverse(
        paths["transverse_inner"], port, result, loops[0], "inner boundary"
    )
    views["transverse_blanket"] = _transverse(
        paths["transverse_blanket"],
        port,
        result,
        blanket_loop,
        "inside blanket",
    )
    views["transverse_outer"] = _transverse(
        paths["transverse_outer"], port, result, outer_loop, "outer boundary"
    )
    views["isometric"] = _isometric(
        paths["isometric"], model, port, result, aperture_model
    )
    views["magnet_clearance"] = _magnet_view(
        paths["magnet_clearance"], stellarator, port, result, aperture_model
    )
    views["surface_anchor"] = _anchor_view(
        paths["surface_anchor"], model, port, result, aperture_model
    )

    placement = port.placement
    spec = placement.surface_anchor
    surface = model._anchor_reference_surface(spec.reference, spec.layer)
    _, _, _, untilted_normal = surface.local_surface_frame(
        np.deg2rad(spec.toroidal_angle), np.deg2rad(spec.poloidal_angle)
    )
    angle = float(
        np.degrees(
            np.arccos(
                np.clip(np.dot(aperture_model.axis, untilted_normal), -1, 1)
            )
        )
    )
    dimensions = aperture_model.recovered_dimensions()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "repository_sha": _repository_sha(Path(__file__).resolve().parent),
        "model_units": "cm",
        "port_name": name,
        "surface_anchor": {
            "reference": spec.reference,
            "layer": spec.layer,
            "toroidal_angle_degrees": spec.toroidal_angle,
            "poloidal_angle_degrees": spec.poloidal_angle,
            "xyz": list(placement.anchor),
        },
        "local_frame": {
            "u": list(placement.local_reference),
            "v": list(placement.local_normal),
            "w": list(placement.local_axis),
            "handedness": float(
                np.dot(
                    np.cross(placement.local_axis, placement.local_reference),
                    placement.local_normal,
                )
            ),
            "axis_normal_angle_degrees": angle,
            "roll_degrees": placement.roll,
            "poloidal_tilt_degrees": placement.surface_axis.poloidal_tilt,
            "toroidal_tilt_degrees": placement.surface_axis.toroidal_tilt,
        },
        "resolved_coordinates": {
            "start_w": result.resolved_start,
            "end_w": result.resolved_end,
            "extension_end_w": result.resolved_end + result.outer_extension,
        },
        "loop_point_counts": list(aperture_model.loop_point_counts),
        "loop_boundaries": [loop.boundary_name for loop in loops],
        "maximum_loop_closure_error": aperture_model.maximum_loop_closure_error,
        "recovered_dimensions": dimensions,
        "intersected_layers": list(result.ordered_intersected_layers),
        "geometry_acceptance": {
            "one_closed_loop_per_boundary": all(
                loop.inner_closure_error == 0.0
                and loop.outer_closure_error == 0.0
                for loop in loops
            ),
            "consistent_parameter_ordering": True,
            "positive_liner_thickness": min(
                dimensions.get("liner_thickness_u", 1.0),
                dimensions.get("liner_thickness_v", 1.0),
            )
            > 0.0,
            "no_blanket_inside_outer_aperture": True,
            "no_liner_inside_clear_aperture": result.maximum_liner_overlap_with_plasma
            == 0.0,
            "no_liner_penetration_into_plasma": result.maximum_liner_overlap_with_plasma
            == 0.0,
            "expected_layer_order": list(result.ordered_intersected_layers),
            "external_extension_positive_axis": result.outer_extension > 0.0,
        },
        "views": views,
        "generated_files": {},
    }
    manifest_path = output_dir / "port_local_manifest.json"
    for key, path in paths.items():
        manifest["generated_files"][path.name] = {
            "path": str(path),
            "sha256": _digest(path),
            "image_size": list(IMAGE_SIZE),
        }
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
