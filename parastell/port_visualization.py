"""Headless, color-preserving visual validation for engineering ports."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from typing import Iterable

import cadquery as cq
import numpy as np


SCHEMA_VERSION = "1.0"

COMPONENT_COLORS = {
    "plasma_or_chamber": (0.25, 0.55, 1.0, 0.28),
    "first_wall": (0.82, 0.84, 0.86, 1.0),
    "breeder": (0.36, 0.62, 0.42, 1.0),
    "shield": (0.80, 0.70, 0.30, 1.0),
    "vacuum_vessel": (0.25, 0.28, 0.32, 1.0),
    "blanket_layer": (0.62, 0.66, 0.70, 1.0),
    "port_void": (0.0, 0.88, 1.0, 0.55),
    "port_liner": (1.0, 0.45, 0.05, 1.0),
    "port_outer_envelope": (0.95, 0.05, 0.72, 0.22),
    "magnet_conductor": (0.48, 0.04, 0.06, 1.0),
    "magnet_casing": (0.16, 0.20, 0.25, 1.0),
    "magnet_clearance_envelope": (1.0, 0.1, 0.1, 0.18),
    "axis_marker": (0.03, 0.03, 0.03, 1.0),
}


@dataclass(frozen=True)
class VisualComponent:
    name: str
    label: str
    kind: str
    solid: object
    color: tuple[float, float, float, float]
    visual_only: bool = False
    neutronics_geometry: bool = True
    volumetric_mesh_geometry: bool = True


def _cq_color(rgba):
    return cq.Color(*rgba)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_sha(start: Path) -> str:
    configured = os.environ.get("PARASTELL_REPOSITORY_SHA")
    if configured:
        return configured
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=start,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _axis_cylinder(start, axis, length, radius):
    return cq.Solid.makeCylinder(
        radius,
        max(float(length), radius),
        cq.Vector(*start),
        cq.Vector(*axis),
    )


def _cutaway_half_space(origin, axis, reference, scale):
    normal = np.cross(axis, reference)
    normal /= np.linalg.norm(normal)
    plane = cq.Plane(
        origin=tuple(origin), xDir=tuple(axis), normal=tuple(normal)
    )
    return cq.Workplane(plane).rect(scale, scale).extrude(scale).val()


def _transverse_slab(origin, axis, reference, scale, thickness):
    plane = cq.Plane(
        origin=tuple(origin), xDir=tuple(reference), normal=tuple(axis)
    )
    return (
        cq.Workplane(plane)
        .workplane(offset=-thickness / 2.0)
        .rect(scale, scale)
        .extrude(thickness)
        .val()
    )


def _shape_bbox(shape):
    box = shape.BoundingBox()
    return {
        "min": [float(box.xmin), float(box.ymin), float(box.zmin)],
        "max": [float(box.xmax), float(box.ymax), float(box.zmax)],
    }


def _assembly(components: Iterable[VisualComponent], name: str):
    assembly = cq.Assembly(name=name)
    for component in components:
        assembly.add(
            component.solid,
            name=component.name,
            color=_cq_color(component.color),
        )
    return assembly


def _render_png(
    path,
    components,
    title,
    port_name,
    axis,
    start,
    end,
    view_direction,
    elev=25,
    azim=-55,
    focus_center=None,
    focus_span=None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(16, 10), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    points = []
    legend = []
    seen_labels = set()
    for component in components:
        vertices, triangles = component.solid.tessellate(0.8, 0.15)
        xyz = np.asarray([[v.x, v.y, v.z] for v in vertices], dtype=float)
        if xyz.size == 0 or not triangles:
            continue
        faces = xyz[np.asarray(triangles, dtype=int)]
        collection = Poly3DCollection(
            faces,
            facecolor=component.color[:3],
            alpha=component.color[3],
            edgecolor="none",
            rasterized=True,
        )
        ax.add_collection3d(collection)
        points.append(xyz)
        if component.label not in seen_labels:
            legend.append(
                Patch(
                    facecolor=component.color[:3],
                    alpha=component.color[3],
                    label=component.label,
                )
            )
            seen_labels.add(component.label)
    if not points:
        raise ValueError("No tessellated geometry was available to render")
    all_points = np.vstack(points)
    lower = all_points.min(axis=0)
    upper = all_points.max(axis=0)
    center = (
        (lower + upper) / 2.0
        if focus_center is None
        else np.asarray(focus_center, dtype=float)
    )
    span = (
        max(float(np.max(upper - lower)), 1.0) * 0.55
        if focus_span is None
        else float(focus_span)
    )
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    axis = np.asarray(axis, dtype=float)
    arrow_length = max(float(np.linalg.norm(np.asarray(end) - start)), 1.0)
    ax.quiver(
        *start,
        *axis,
        length=arrow_length,
        normalize=True,
        color="black",
        linewidth=2.2,
        arrow_length_ratio=0.08,
    )
    ax.text(*start, "  resolved start", color="black", fontsize=9)
    ax.text(*end, "  end of extension", color="black", fontsize=9)
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("X [cm]")
    ax.set_ylabel("Y [cm]")
    ax.set_zlabel("Z [cm]")
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1))
    ax.legend(handles=legend, loc="upper left", fontsize=8, framealpha=0.9)
    fig.text(
        0.5,
        0.02,
        f"Port: {port_name} | model units: cm | view direction: {view_direction}",
        ha="center",
        fontsize=10,
    )
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def export_port_visual_validation(
    stellarator,
    output_dir,
    port_names=None,
    include_magnets=True,
    include_clearance_envelopes=True,
    include_axis_markers=True,
    make_cutaway=True,
):
    """Export a named CAD assembly, interactive GLBs, PNGs, and manifest.

    All physical solids come directly from the finalized in-memory CadQuery
    model. Clearance envelopes and markers are explicitly marked visual-only.
    """
    if stellarator.invessel_build is None:
        raise ValueError(
            "An in-vessel build is required for port visualization"
        )
    model = stellarator.invessel_build
    available = tuple(model.port_void_components)
    selected = available if port_names is None else tuple(port_names)
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"Unknown port name(s): {sorted(unknown)}")
    if not selected:
        raise ValueError("At least one port must be selected")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    port_name = selected[0]
    port = model.port_specs[port_name]
    result = model.port_geometry_diagnostics[port_name]
    axis = np.asarray(port.placement.local_axis, dtype=float)
    reference = np.asarray(port.placement.local_reference, dtype=float)
    anchor = np.asarray(port.placement.anchor, dtype=float)
    start = anchor + axis * result.resolved_start
    resolved_end = anchor + axis * result.resolved_end
    extension_end = anchor + axis * (
        result.resolved_end + result.outer_extension
    )

    components = []
    user_layers = tuple(model.radial_build.user_layer_names)
    for name, solid in model.Components.items():
        if name.endswith("__void"):
            source_port = name[: -len("__void")]
            if source_port not in selected:
                continue
            kind, label = "port_void", f"{source_port} clear void"
        elif name.endswith("__liner"):
            source_port = name[: -len("__liner")]
            if source_port not in selected:
                continue
            kind, label = "port_liner", f"{source_port} liner"
        elif name in {"plasma", "chamber"}:
            kind, label = "plasma_or_chamber", name.replace("_", " ")
        else:
            kind, label = "blanket_layer", name.replace("_", " ")
        color = COMPONENT_COLORS.get(name, COMPONENT_COLORS[kind])
        components.append(VisualComponent(name, label, kind, solid, color))

    for selected_name in selected:
        envelope = model.port_outer_envelopes[selected_name]
        components.append(
            VisualComponent(
                f"{selected_name}__outer_envelope",
                f"{selected_name} outer envelope",
                "port_outer_envelope",
                envelope,
                COMPONENT_COLORS["port_outer_envelope"],
                visual_only=True,
                neutronics_geometry=False,
                volumetric_mesh_geometry=False,
            )
        )
        if include_clearance_envelopes:
            clearance = model.build_port_clearance_envelope(
                model.port_specs[selected_name]
            )
            components.append(
                VisualComponent(
                    f"{selected_name}__magnet_clearance_envelope",
                    "magnet-clearance envelope",
                    "magnet_clearance_envelope",
                    clearance,
                    COMPONENT_COLORS["magnet_clearance_envelope"],
                    visual_only=True,
                    neutronics_geometry=False,
                    volumetric_mesh_geometry=False,
                )
            )

    if include_magnets and stellarator.magnet_set is not None:
        for magnet in stellarator.magnet_set.iter_coil_solids():
            kind = (
                "magnet_casing"
                if magnet.region_kind == "outer_casing"
                else "magnet_conductor"
            )
            components.append(
                VisualComponent(
                    f"magnet_{magnet.coil_id}__{magnet.region_kind}",
                    kind.replace("_", " "),
                    kind,
                    magnet.solid,
                    COMPONENT_COLORS[kind],
                )
            )

    if include_axis_markers:
        axis_length = float(np.linalg.norm(extension_end - start))
        marker_radius = max(axis_length / 180.0, 0.15)
        marker_items = [
            (
                "port_centerline",
                _axis_cylinder(start, axis, axis_length, marker_radius),
            ),
            (
                "resolved_start_marker",
                cq.Solid.makeSphere(marker_radius * 2, cq.Vector(*start)),
            ),
            (
                "resolved_end_marker",
                cq.Solid.makeSphere(
                    marker_radius * 2, cq.Vector(*resolved_end)
                ),
            ),
            (
                "end_of_extension_marker",
                cq.Solid.makeSphere(
                    marker_radius * 2, cq.Vector(*extension_end)
                ),
            ),
        ]
        for name, solid in marker_items:
            components.append(
                VisualComponent(
                    name,
                    name.replace("_", " "),
                    "axis_marker",
                    solid,
                    COMPONENT_COLORS["axis_marker"],
                    visual_only=True,
                    neutronics_geometry=False,
                    volumetric_mesh_geometry=False,
                )
            )

    physical = [c for c in components if not c.visual_only]
    full_assembly = _assembly(components, "ported_sector_colored")
    step_path = output_dir / "ported_sector_colored.step"
    glb_path = output_dir / "ported_sector_colored.glb"
    cq.exporters.assembly.exportAssembly(full_assembly, str(step_path))
    cq.exporters.assembly.exportGLTF(full_assembly, str(glb_path), binary=True)

    port_only = [
        c
        for c in components
        if c.kind
        in {
            "port_void",
            "port_liner",
            "port_outer_envelope",
            "axis_marker",
        }
    ]
    port_only_path = output_dir / "port_only.glb"
    cq.exporters.assembly.exportGLTF(
        _assembly(port_only, "port_only"), str(port_only_path), binary=True
    )

    all_boxes = [c.solid.BoundingBox() for c in physical]
    extent = max(max(box.xlen, box.ylen, box.zlen) for box in all_boxes) * 2.5
    cutter = _cutaway_half_space(start, axis, reference, extent)
    cutaway_components = []
    for component in components:
        cut = component.solid.cut(cutter)
        if not cut.isNull():
            cutaway_components.append(
                VisualComponent(
                    component.name,
                    component.label,
                    component.kind,
                    cut,
                    component.color,
                    component.visual_only,
                    component.neutronics_geometry,
                    component.volumetric_mesh_geometry,
                )
            )
    cutaway_without_magnets = [
        component
        for component in cutaway_components
        if component.kind not in {"magnet_conductor", "magnet_casing"}
        and not component.visual_only
    ]
    cutaway_path = output_dir / "ported_sector_cutaway.glb"
    if make_cutaway:
        cq.exporters.assembly.exportGLTF(
            _assembly(cutaway_components, "ported_sector_cutaway"),
            str(cutaway_path),
            binary=True,
        )

    blanket_midpoint = anchor + axis * (
        (result.resolved_start + result.resolved_end) / 2.0
    )
    slab = _transverse_slab(blanket_midpoint, axis, reference, extent, 0.5)
    transverse = []
    for component in physical:
        section = component.solid.intersect(slab)
        if not section.isNull():
            transverse.append(
                VisualComponent(
                    component.name,
                    component.label,
                    component.kind,
                    section,
                    component.color,
                )
            )

    section_normal = np.cross(axis, reference)
    section_elev = float(np.degrees(np.arcsin(section_normal[2])))
    section_azim = float(
        np.degrees(np.arctan2(section_normal[1], section_normal[0]))
    )
    axis_elev = float(np.degrees(np.arcsin(axis[2])))
    axis_azim = float(np.degrees(np.arctan2(axis[1], axis[0])))
    png_specs = [
        (
            "port_isometric.png",
            cutaway_without_magnets,
            "Ported sector — isometric cutaway",
            "isometric",
            24,
            -52,
        ),
        (
            "port_axis_section.png",
            cutaway_without_magnets,
            "Port longitudinal axis section",
            "normal to axis plane",
            section_elev,
            section_azim,
        ),
        (
            "port_blanket_cutaway.png",
            cutaway_without_magnets,
            "Port blanket cutaway",
            "along port axis",
            axis_elev,
            axis_azim,
        ),
        (
            "port_magnet_clearance.png",
            components,
            "Port–magnet clearance",
            "toward nearest magnet",
            22,
            -48,
        ),
    ]
    exploded = []
    blanket_index = 0
    for component in physical:
        if component.kind == "blanket_layer":
            offset = reference * blanket_index * 7.0
            blanket_index += 1
            moved = component.solid.moved(cq.Location(cq.Vector(*offset)))
        else:
            moved = component.solid
        exploded.append(
            VisualComponent(
                component.name,
                component.label,
                component.kind,
                moved,
                component.color,
            )
        )
    png_specs.append(
        (
            "port_layers_exploded.png",
            exploded,
            "Port layers — exploded",
            "isometric exploded",
            28,
            -58,
        )
    )
    generated = [step_path, glb_path, port_only_path]
    if make_cutaway:
        generated.append(cutaway_path)
    port_box = model.port_outer_envelopes[port_name].BoundingBox()
    focus_center = np.array(
        [
            (port_box.xmin + port_box.xmax) / 2.0,
            (port_box.ymin + port_box.ymax) / 2.0,
            (port_box.zmin + port_box.zmax) / 2.0,
        ]
    )
    focus_span = max(port_box.xlen, port_box.ylen, port_box.zlen) * 1.15
    for filename, render_components, title, direction, elev, azim in png_specs:
        path = output_dir / filename
        _render_png(
            path,
            render_components,
            title,
            port_name,
            axis,
            start,
            extension_end,
            direction,
            elev,
            azim,
            focus_center,
            focus_span
            * (
                3.0
                if "magnet_clearance" in filename
                else 1.5 if "exploded" in filename else 1.0
            ),
        )
        generated.append(path)
    transverse_path = output_dir / "port_axis_transverse.png"
    _render_png(
        transverse_path,
        transverse,
        "Port transverse section inside blanket",
        port_name,
        axis,
        start,
        extension_end,
        "outward along port axis",
        float(np.degrees(np.arcsin(axis[2]))),
        float(np.degrees(np.arctan2(axis[1], axis[0]))),
        focus_center,
        focus_span,
    )
    generated.append(transverse_path)

    collision_records = [
        record.to_dict()
        for record in stellarator.port_magnet_collision_report
        if record.port_name == port_name
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "repository_sha": _repository_sha(Path(__file__).resolve().parent),
        "model_units": "cm",
        "port_name": port_name,
        "axis": axis.tolist(),
        "reference_direction": reference.tolist(),
        "resolved_start": start.tolist(),
        "resolved_end": resolved_end.tolist(),
        "end_of_extension": extension_end.tolist(),
        "outer_extension": result.outer_extension,
        "clear_aperture": {
            "shape": port.cross_section.shape,
            "radius": port.cross_section.radius,
            "width": port.cross_section.width,
            "height": port.cross_section.height,
        },
        "liner": {
            "enabled": port.liner.enabled,
            "thickness": port.liner.thickness,
            "material_tag": port.liner.mat_tag,
        },
        "intersected_layers": list(result.ordered_intersected_layers),
        "visible_bounding_box_includes": [
            port_name,
            *list(result.ordered_intersected_layers),
        ],
        "component_names": [component.name for component in components],
        "expected_legend_labels": list(
            dict.fromkeys(component.label for component in components)
        ),
        "components": [
            {
                "name": component.name,
                "label": component.label,
                "kind": component.kind,
                "color_rgba": list(component.color),
                "volume": float(component.solid.Volume()),
                "bounding_box": _shape_bbox(component.solid),
                "visual_only_geometry": component.visual_only,
                "neutronics_geometry": component.neutronics_geometry,
                "volumetric_mesh_geometry": component.volumetric_mesh_geometry,
            }
            for component in components
        ],
        "magnet_clearance_result": collision_records,
        "generated_files": {
            path.name: {"path": str(path), "sha256": _sha256(path)}
            for path in generated
        },
    }
    manifest_path = output_dir / "port_visual_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
