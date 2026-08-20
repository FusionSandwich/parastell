"""Deterministic renderer for actual ParaStell layer-bounded port solids."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import cadquery as cq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import yaml

from .invessel_build import InVesselBuild
from .invessel_build import RadialBuild
from .parastell import Stellarator


VIEW_NAMES = (
    "01_unported_isometric",
    "02_ported_isometric",
    "03_ported_section",
    "04_aperture_closeup",
    "05_selected_layer_cuts",
    "06_vacuum_fill",
    "07_volume_closure",
    "08_one_layer_variant",
    "09_circular_variant",
    "10_port_variants_contact_sheet",
)
COLORS = {
    "first_wall": "#d95f3f",
    "breeder": "#e6ad36",
    "back_wall": "#8a9a5b",
    "shield": "#477998",
    "vacuum_vessel": "#293241",
    "equatorial_diagnostic": "#c9f2ef",
}


class _RenderReferenceSurface:
    """Smooth five-period stellarator fixture used only for CAD rendering."""

    angles_in_degrees = False

    def angles_to_xyz(self, toroidal_angles, poloidal_angles, s, scale):
        phi = float(toroidal_angles)
        theta = np.atleast_1d(np.asarray(poloidal_angles, dtype=float))
        major = 6.0 * scale
        modulation = 1.0 + 0.10 * np.cos(5.0 * phi)
        minor = 1.6 * (1.0 + 0.2 * (s - 1.0)) * scale
        x = (major + minor * modulation * np.cos(theta)) * np.cos(phi)
        y = (major + minor * modulation * np.cos(theta)) * np.sin(phi)
        z = minor * (1.0 - 0.08 * np.cos(5.0 * phi)) * np.sin(theta)
        return np.stack([x, y, z], axis=-1)

    def calculate_tangents(self, toroidal_angle, poloidal_angles, s, scale):
        theta = np.asarray(poloidal_angles, dtype=float)
        delta = 1.0e-5
        backward = self.angles_to_xyz(toroidal_angle, theta - delta, s, scale)
        forward = self.angles_to_xyz(toroidal_angle, theta + delta, s, scale)
        tangent = forward - backward
        return tangent / np.linalg.norm(tangent, axis=1)[:, None]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, Mapping):
        raise ValueError("port rendering configuration must be a mapping")
    return dict(value)


def _surface_placement(
    stellarator: Stellarator, placement: Mapping[str, Any], wall_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref = stellarator._ref_surf
    degrees = bool(getattr(ref, "angles_in_degrees", False))
    toroidal = float(placement["toroidal_deg"])
    poloidal = float(placement["poloidal_deg"])
    delta = 0.01
    if not degrees:
        toroidal = np.deg2rad(toroidal)
        poloidal = np.deg2rad(poloidal)
        delta = np.deg2rad(delta)
    point = np.asarray(
        ref.angles_to_xyz(toroidal, [poloidal], wall_s, 100.0), dtype=float
    )[0]
    pt = np.asarray(
        ref.angles_to_xyz(toroidal, [poloidal + delta], wall_s, 100.0),
        dtype=float,
    )[0]
    pp = np.asarray(
        ref.angles_to_xyz(toroidal + delta, [poloidal], wall_s, 100.0),
        dtype=float,
    )[0]
    poloidal_direction = pt - point
    toroidal_direction = pp - point
    poloidal_direction /= np.linalg.norm(poloidal_direction)
    toroidal_direction /= np.linalg.norm(toroidal_direction)
    normal = np.cross(toroidal_direction, poloidal_direction)
    normal /= np.linalg.norm(normal)
    axis_point = np.asarray(
        ref.angles_to_xyz(toroidal, [poloidal], 0.0, 100.0), dtype=float
    )[0]
    if np.dot(normal, point - axis_point) < 0.0:
        normal *= -1.0
    return point, normal, poloidal_direction


def _port_mapping(
    stellarator: Stellarator,
    config: Mapping[str, Any],
    variant: Mapping[str, Any],
) -> dict[str, Any]:
    placement = config["port"]["placement"]
    point, normal, reference = _surface_placement(
        stellarator, placement, float(config["invessel_build"]["wall_s"])
    )
    cross = dict(config["port"]["cross_section"])
    cross.update(variant.get("cross_section", {}))
    if cross["shape"] == "circle":
        cross.pop("width", None)
        cross.pop("height", None)
    else:
        cross.pop("radius", None)
    span = dict(config["port"]["layer_span"])
    span.update(variant.get("layer_span", {}))
    layer_names = list(config["invessel_build"]["radial_build"])
    start_index = layer_names.index(span["start"])
    step = 1 if span["direction"] == "outward" else -1
    target_layers = [
        layer_names[start_index + step * index]
        for index in range(int(span["count"]))
    ]
    return {
        "name": config["port"]["name"],
        "placement": {
            "mode": "cartesian",
            "anchor": point.tolist(),
            "axis": normal.tolist(),
            "reference_direction": reference.tolist(),
            "max_search_length": 1000.0,
        },
        "cross_section": cross,
        "extent": {
            "start": {
                "reference": "layer",
                "layer": target_layers[0],
                "fraction": 0.10,
            },
            "end": {
                "reference": "layer",
                "layer": target_layers[-1],
                "fraction": 0.50,
            },
            "outer_extension": 0.0,
        },
        "expected_layers": target_layers,
        "fill": dict(config["port"]["fill"]),
        "repetition": {"mode": "single"},
    }


def _build(
    config: Mapping[str, Any],
    config_path: Path,
    variant: Mapping[str, Any] | None,
    ported: bool,
) -> tuple[Stellarator, dict[str, cq.Shape]]:
    ivb = dict(config["invessel_build"])
    reference = _RenderReferenceSurface()
    stellarator = SimpleNamespace(_ref_surf=reference)
    ports = None
    if ported:
        ports = [_port_mapping(stellarator, config, variant or {})]
    radial = RadialBuild(
        ivb["toroidal_angles"],
        ivb["poloidal_angles"],
        ivb["wall_s"],
        ivb["radial_build"],
        split_chamber=bool(ivb.get("split_chamber", False)),
    )
    model = InVesselBuild(
        reference,
        radial,
        num_ribs=int(ivb.get("num_ribs", 31)),
        num_rib_pts=int(ivb.get("num_rib_pts", 49)),
        scale=float(ivb.get("scale", 100.0)),
        ports=ports,
    )
    model.populate_surfaces()
    model.calculate_loci()
    model.generate_components_cadquery()
    stellarator.invessel_build = model
    return stellarator, dict(model.Components)


def _triangles(shape: cq.Shape, tolerance: float = 3.0) -> np.ndarray:
    vertices, triangles = shape.tessellate(tolerance)
    xyz = np.asarray(
        [[point.x, point.y, point.z] for point in vertices], dtype=float
    )
    return xyz[np.asarray(triangles, dtype=int)]


def _camera(
    ax, triangles: Sequence[np.ndarray], closeup: bool, section: bool
) -> dict[str, Any]:
    points = np.concatenate([item.reshape(-1, 3) for item in triangles])
    if section:
        points = points[points[:, 1] >= np.median(points[:, 1])]
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2.0
    radius = max(float(np.max(high - low)) / (5.0 if closeup else 1.8), 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.view_init(elev=24.0, azim=-48.0)
    ax.set_box_aspect((1, 1, 0.8))
    ax.set_axis_off()
    return {
        "elevation_deg": 24.0,
        "azimuth_deg": -48.0,
        "center_cm": center.tolist(),
        "radius_cm": radius,
    }


def _render(
    path: Path,
    components: Mapping[str, cq.Shape],
    *,
    title: str,
    resolution: tuple[int, int],
    section: bool = False,
    transparency: float = 0.45,
    only: Sequence[str] | None = None,
    closeup: bool = False,
) -> dict[str, Any]:
    names = [name for name in components if only is None or name in only]
    mesh = {name: _triangles(components[name]) for name in names}
    fig = plt.figure(
        figsize=(resolution[0] / 120, resolution[1] / 120),
        dpi=120,
        facecolor="#f4efe5",
    )
    ax = fig.add_subplot(111, projection="3d", facecolor="#f4efe5")
    plotted = []
    for index, (name, triangles) in enumerate(mesh.items()):
        visible = triangles
        if section:
            visible = triangles[
                np.mean(triangles[:, :, 1], axis=1)
                >= np.median(triangles[:, :, 1])
            ]
        alpha = 0.95 if name == "equatorial_diagnostic" else transparency
        poly = Poly3DCollection(
            visible,
            facecolor=COLORS.get(name, plt.cm.Set2(index / max(len(mesh), 1))),
            edgecolor="#202b33",
            linewidth=0.05,
            alpha=alpha,
        )
        ax.add_collection3d(poly)
        plotted.append(visible)
    focus = [
        triangles
        for component_name, triangles in mesh.items()
        if "__void" in component_name
    ]
    camera = _camera(
        ax,
        focus if closeup and focus else plotted,
        False if closeup and focus else closeup,
        False if closeup and focus else section,
    )
    ax.set_title(
        title, fontfamily="DejaVu Serif", fontsize=16, color="#202b33", pad=8
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return camera


def render_port_views(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    name: str | None = None,
    resolution: tuple[int, int] = (1600, 1000),
    section: bool = False,
    transparency: float = 0.45,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = _load(config_path)
    output = Path(output_dir).resolve()
    variants = config["variants"]
    _, unported = _build(config, config_path, None, False)
    models: dict[str, dict[str, cq.Shape]] = {}
    diagnostics: dict[str, Any] = {}
    for variant_name, definition in variants.items():
        stellarator, components = _build(config, config_path, definition, True)
        models[variant_name] = components
        port_name = config["port"]["name"]
        fill_name = (
            f"{port_name}__void"
            if f"{port_name}__void" in components
            else port_name
        )
        fill = components[fill_name]
        port_spec = next(
            port
            for port in stellarator.invessel_build.ports
            if port.name == port_name
        )
        selected = (
            port_spec.resolution.layers
            if port_spec.resolution is not None
            else port_spec.expected_layers
        )
        result = stellarator.invessel_build.port_geometry_diagnostics[
            port_name
        ]
        baseline = float(result.original_blanket_volume)
        remaining = float(result.remaining_blanket_volume)
        fill_volume = float(result.void_volume_inside_blanket)
        diagnostics[variant_name] = {
            "selected_layers": list(selected),
            "baseline_selected_volume_cm3": baseline,
            "remaining_selected_volume_cm3": remaining,
            "vacuum_fill_volume_cm3": fill_volume,
            "total_void_solid_volume_cm3": float(fill.Volume()),
            "closure_relative_error": float(result.closure_error) / baseline,
            "all_solids_valid": all(
                shape.isValid() for shape in components.values()
            ),
            "fill_overlap_cm3": sum(
                float(fill.intersect(components[layer]).Volume())
                for layer in selected
            ),
        }
        if (
            not diagnostics[variant_name]["all_solids_valid"]
            or diagnostics[variant_name]["fill_overlap_cm3"] > 1e-6
            or diagnostics[variant_name]["closure_relative_error"] > 1e-5
        ):
            raise RuntimeError(
                f"port CAD validation failed for {variant_name}: "
                f"{diagnostics[variant_name]}"
            )
    primary = models[name or "three_layer_rectangle"]
    raw_port_name = config["port"]["name"]
    port_name = (
        f"{raw_port_name}__void"
        if f"{raw_port_name}__void" in primary
        else raw_port_name
    )
    selected = diagnostics[name or "three_layer_rectangle"]["selected_layers"]
    specs = [
        (
            VIEW_NAMES[0],
            unported,
            "Unported 90 degree ParaStell sector",
            False,
            None,
            False,
        ),
        (
            VIEW_NAMES[1],
            primary,
            "Layer-bounded diagnostic port",
            False,
            None,
            False,
        ),
        (
            VIEW_NAMES[2],
            primary,
            "Section through port and radial build",
            True,
            None,
            False,
        ),
        (
            VIEW_NAMES[3],
            primary,
            "Clear aperture at the magnet-facing stack",
            False,
            [*selected, port_name],
            True,
        ),
        (
            VIEW_NAMES[4],
            primary,
            "Material removed only from selected layers",
            True,
            selected,
            True,
        ),
        (
            VIEW_NAMES[5],
            primary,
            "Vacuum fill volume",
            False,
            [port_name],
            True,
        ),
        (
            VIEW_NAMES[6],
            primary,
            "Volume-closure geometry",
            True,
            [*selected, port_name],
            True,
        ),
        (
            VIEW_NAMES[7],
            models["one_layer_rectangle"],
            "One-layer rectangular variant",
            True,
            ["first_wall", port_name],
            True,
        ),
        (
            VIEW_NAMES[8],
            models["two_layer_circle"],
            "Two-layer circular variant",
            True,
            ["breeder", "back_wall", port_name],
            True,
        ),
    ]
    manifest_images = []
    for view, components, title, cut, only, closeup in specs:
        path = output / f"{view}.png"
        camera = _render(
            path,
            components,
            title=title,
            resolution=resolution,
            section=cut or section,
            transparency=transparency,
            only=only,
            closeup=closeup,
        )
        manifest_images.append(
            {
                "name": view,
                "path": path.name,
                "sha256": _sha256(path),
                "camera": camera,
                "section": bool(cut or section),
                "components": list(components if only is None else only),
            }
        )
    contact = output / f"{VIEW_NAMES[9]}.png"
    figures = [
        plt.imread(output / f"{item}.png")
        for item in (VIEW_NAMES[3], VIEW_NAMES[7], VIEW_NAMES[8])
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#f4efe5")
    for ax, image, label in zip(
        axes,
        figures,
        ("3 layers rectangle", "1 layer rectangle", "2 layers circle"),
    ):
        ax.imshow(image)
        ax.set_title(label, fontfamily="DejaVu Serif")
        ax.axis("off")
    fig.savefig(contact, dpi=120, bbox_inches="tight")
    plt.close(fig)
    manifest_images.append(
        {
            "name": VIEW_NAMES[9],
            "path": contact.name,
            "sha256": _sha256(contact),
            "camera": "contact_sheet",
            "section": False,
            "components": [],
        }
    )
    manifest = {
        "schema": "parastell.port_visualization/v1",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "renderer": "parastell.port_visualization",
        "geometry_basis": "actual CadQuery solids produced by RadialBuild and InVesselBuild",
        "reference_surface": "documented analytic five-period stellarator rendering fixture; VMEC is reserved for the transport workflow",
        "units": "cm",
        "resolution_px": list(resolution),
        "transparency": transparency,
        "variants": diagnostics,
        "images": manifest_images,
    }
    manifest_path = output / "port_visualization_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--name")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--resolution",
        nargs=2,
        type=int,
        default=(1600, 1000),
        metavar=("WIDTH", "HEIGHT"),
    )
    parser.add_argument("--section", action="store_true")
    parser.add_argument("--transparency", type=float, default=0.45)
    args = parser.parse_args(argv)
    manifest = render_port_views(
        args.config,
        args.output,
        name=args.name,
        resolution=tuple(args.resolution),
        section=args.section,
        transparency=args.transparency,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
