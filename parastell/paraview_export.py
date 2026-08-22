"""Generic named ParaView exports for DAGMC and MOAB models.

The writer intentionally has no dependency on ParaStell port classes.  It
converts named DAGMC volumes and named MOAB tetrahedron sets into ordinary VTK
XML leaf files referenced by a multiblock ``.vtm`` file.  ParaView therefore
does not need a MOAB reader plugin to inspect the exported bundle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Iterable, Mapping
import xml.etree.ElementTree as ET

import numpy as np


COMPONENT_CLASS_IDS = {
    "plasma": 1,
    "chamber": 2,
    "first_wall": 3,
    "breeder": 4,
    "shield": 5,
    "vacuum_vessel": 6,
    "port_void": 7,
    "port_liner": 8,
    "magnet_conductor": 9,
    "magnet_casing": 10,
    "graveyard": 11,
    "blanket_layer": 12,
}

COMPONENT_COLORS = {
    "plasma": (0.20, 0.55, 1.00),
    "chamber": (0.20, 0.55, 1.00),
    "first_wall": (0.82, 0.82, 0.82),
    "breeder": (0.36, 0.65, 0.38),
    "shield": (0.90, 0.78, 0.24),
    "vacuum_vessel": (0.25, 0.28, 0.32),
    "port_void": (0.00, 0.90, 1.00),
    "port_liner": (1.00, 0.48, 0.08),
    "magnet_conductor": (0.78, 0.08, 0.08),
    "magnet_casing": (0.28, 0.13, 0.13),
    "graveyard": (0.70, 0.70, 0.70),
    "blanket_layer": (0.60, 0.60, 0.60),
}

REQUIRED_IMAGE_NAMES = (
    "full_reactor_isometric.png",
    "full_reactor_top.png",
    "full_reactor_side.png",
    "full_reactor_field_periods.png",
    "full_reactor_port_location.png",
    "full_reactor_port_and_all_magnets.png",
    "full_reactor_port_nearest_magnets.png",
    "full_reactor_clip_through_port_axis.png",
    "full_reactor_transverse_port_slice.png",
    "full_reactor_blanket_cutaway.png",
    "full_reactor_streaming_path.png",
    "full_reactor_volume_mesh_regions.png",
    "full_reactor_volume_mesh_quality.png",
)

PROHIBITED_PUBLIC_NAME_RE = re.compile(r"agent|codex|chatgpt", re.IGNORECASE)


@dataclass(frozen=True)
class ParaViewExecutables:
    """Resolved ParaView command-line programs from one installation."""

    paraview: str | None
    pvpython: str | None
    pvbatch: str | None
    version: str | None


@dataclass(frozen=True)
class ParaViewBlock:
    """Metadata retained on one VTK multiblock leaf."""

    component_name: str
    component_kind: str
    component_class: str
    material_tag: str
    dagmc_volume_id: int | None = None
    field_period_id: int | None = None
    instance_id: str | None = None
    port_name: str | None = None
    visual_only: bool = False
    transport_geometry: bool = True
    volumetric_geometry: bool = False
    hidden_by_default: bool = False
    file: str | None = None
    cell_count: int = 0
    bounds: tuple[float, float, float, float, float, float] | None = None


def _run_version(executable: str | None) -> str | None:
    if executable is None:
        return None
    completed = subprocess.run(
        [executable, "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    match = re.search(r"paraview version\s+([^\s]+)", completed.stdout, re.I)
    return match.group(1) if match else None


def discover_paraview_executables(
    search_roots: Iterable[str | os.PathLike] | None = None,
) -> ParaViewExecutables:
    """Find ParaView GUI and CLI executables without starting the GUI."""

    names = ("paraview", "pvpython", "pvbatch")
    found = {name: shutil.which(name) for name in names}
    roots = [Path(path) for path in (search_roots or ())]
    if os.name == "nt":
        program_files = Path(
            os.environ.get("ProgramFiles", r"C:\Program Files")
        )
        roots.extend(sorted(program_files.glob("ParaView*"), reverse=True))
    for root in roots:
        if not root.exists():
            continue
        candidates = (root, root / "bin")
        for directory in candidates:
            for name in names:
                if found[name] is not None:
                    continue
                suffix = ".exe" if os.name == "nt" else ""
                candidate = directory / f"{name}{suffix}"
                if candidate.is_file():
                    found[name] = str(candidate.resolve())
    version = _run_version(found["pvbatch"] or found["paraview"])
    return ParaViewExecutables(
        found["paraview"], found["pvpython"], found["pvbatch"], version
    )


def audit_public_names(values: Iterable[str]) -> tuple[str, ...]:
    """Return public names containing prohibited automation-oriented terms."""

    return tuple(
        value for value in values if PROHIBITED_PUBLIC_NAME_RE.search(value)
    )


def _component_class(name: str, kind: str | None = None) -> str:
    lowered = name.lower()
    if kind in COMPONENT_CLASS_IDS:
        return str(kind)
    if lowered == "plasma":
        return "plasma"
    if lowered in {"chamber", "sol"}:
        return "chamber"
    if "first_wall" in lowered:
        return "first_wall"
    if "breeder" in lowered:
        return "breeder"
    if "shield" in lowered or "back_wall" in lowered:
        return "shield"
    if "vacuum_vessel" in lowered or "vac_vessel" in lowered:
        return "vacuum_vessel"
    if lowered.endswith("__void"):
        return "port_void"
    if lowered.endswith("__liner"):
        return "port_liner"
    if "conductor" in lowered:
        return "magnet_conductor"
    if "casing" in lowered:
        return "magnet_casing"
    if lowered == "graveyard":
        return "graveyard"
    return "blanket_layer"


def _component_kind(name: str) -> str:
    cls = _component_class(name)
    return {
        "plasma": "plasma_or_chamber",
        "chamber": "plasma_or_chamber",
        "port_void": "port_void",
        "port_liner": "port_liner",
        "magnet_conductor": "magnet_conductor",
        "magnet_casing": "magnet_casing",
        "graveyard": "graveyard",
    }.get(cls, "blanket_layer")


def _safe_leaf_name(name: str) -> str:
    leaf = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    if not leaf:
        raise ValueError(
            "ParaView block name has no filesystem-safe characters"
        )
    return leaf


def _bounds(points: np.ndarray):
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    return (
        float(lower[0]),
        float(upper[0]),
        float(lower[1]),
        float(upper[1]),
        float(lower[2]),
        float(upper[2]),
    )


def _data_array(parent, name, values, vtk_type="Float64", components=None):
    attributes = {"type": vtk_type, "Name": name, "format": "ascii"}
    if components is not None:
        attributes["NumberOfComponents"] = str(components)
    array = ET.SubElement(parent, "DataArray", attributes)
    if vtk_type == "String":
        array.text = json.dumps(str(values))
    else:
        flat = np.asarray(values).reshape(-1)
        array.text = " ".join(str(value) for value in flat)
    return array


def _field_data(piece, block: ParaViewBlock):
    field = ET.SubElement(piece, "FieldData")
    for name, value in (
        ("component_name", block.component_name),
        ("component_kind", block.component_kind),
        ("component_class", block.component_class),
        ("material_tag", block.material_tag),
        ("instance_id", block.instance_id or ""),
        ("port_name", block.port_name or ""),
    ):
        _data_array(field, name, value, "String")
    for name, value in (
        ("dagmc_volume_id", block.dagmc_volume_id or -1),
        ("field_period_id", block.field_period_id or 0),
        ("visual_only", int(block.visual_only)),
        ("transport_geometry", int(block.transport_geometry)),
        ("volumetric_geometry", int(block.volumetric_geometry)),
    ):
        _data_array(field, name, [value], "Int32")


def write_polydata(path, triangles, block: ParaViewBlock) -> ParaViewBlock:
    """Write triangle coordinates to a metadata-bearing ASCII VTP leaf."""

    path = Path(path)
    triangles = np.asarray(triangles, dtype=float).reshape((-1, 3, 3))
    points = triangles.reshape((-1, 3))
    root = ET.Element(
        "VTKFile",
        {"type": "PolyData", "version": "1.0", "byte_order": "LittleEndian"},
    )
    poly = ET.SubElement(root, "PolyData")
    piece = ET.SubElement(
        poly,
        "Piece",
        {
            "NumberOfPoints": str(len(points)),
            "NumberOfVerts": "0",
            "NumberOfLines": "0",
            "NumberOfStrips": "0",
            "NumberOfPolys": str(len(triangles)),
        },
    )
    _field_data(piece, block)
    ET.SubElement(piece, "PointData")
    cell_data = ET.SubElement(piece, "CellData")
    class_id = COMPONENT_CLASS_IDS[block.component_class]
    _data_array(
        cell_data, "component_class_id", [class_id] * len(triangles), "Int32"
    )
    _data_array(
        cell_data,
        "dagmc_volume_id",
        [block.dagmc_volume_id or -1] * len(triangles),
        "Int32",
    )
    points_node = ET.SubElement(piece, "Points")
    _data_array(points_node, "Points", points, components=3)
    polys = ET.SubElement(piece, "Polys")
    _data_array(polys, "connectivity", np.arange(len(points)), "Int64")
    _data_array(
        polys, "offsets", np.arange(1, len(triangles) + 1) * 3, "Int64"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return ParaViewBlock(
        **{
            **asdict(block),
            "file": str(path),
            "cell_count": len(triangles),
            "bounds": _bounds(points),
        }
    )


def write_unstructured_grid(
    path, points, connectivity, block: ParaViewBlock, quality=None
) -> ParaViewBlock:
    """Write tetrahedra to a metadata-bearing ASCII VTU leaf."""

    path = Path(path)
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    connectivity = np.asarray(connectivity, dtype=np.int64).reshape((-1, 4))
    root = ET.Element(
        "VTKFile",
        {
            "type": "UnstructuredGrid",
            "version": "1.0",
            "byte_order": "LittleEndian",
        },
    )
    grid = ET.SubElement(root, "UnstructuredGrid")
    piece = ET.SubElement(
        grid,
        "Piece",
        {
            "NumberOfPoints": str(len(points)),
            "NumberOfCells": str(len(connectivity)),
        },
    )
    _field_data(piece, block)
    ET.SubElement(piece, "PointData")
    cell_data = ET.SubElement(piece, "CellData")
    class_id = COMPONENT_CLASS_IDS[block.component_class]
    _data_array(
        cell_data,
        "component_class_id",
        [class_id] * len(connectivity),
        "Int32",
    )
    for name, values in (quality or {}).items():
        values = np.asarray(values, dtype=float)
        if len(values) != len(connectivity):
            raise ValueError(
                f"Quality array {name!r} has the wrong cell count"
            )
        _data_array(cell_data, name, values)
    points_node = ET.SubElement(piece, "Points")
    _data_array(points_node, "Points", points, components=3)
    cells = ET.SubElement(piece, "Cells")
    _data_array(cells, "connectivity", connectivity, "Int64")
    _data_array(
        cells, "offsets", np.arange(1, len(connectivity) + 1) * 4, "Int64"
    )
    _data_array(cells, "types", [10] * len(connectivity), "UInt8")
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return ParaViewBlock(
        **{
            **asdict(block),
            "file": str(path),
            "cell_count": len(connectivity),
            "bounds": _bounds(points),
        }
    )


def _dagmc_blocks(h5m, block_dir, ledger=None):
    import pydagmc

    from .dagmc_assembly import dagmc_component_names

    model = pydagmc.Model(str(h5m))
    ledger_by_id = {
        int(record["dagmc_volume_id"]): record for record in (ledger or ())
    }
    try:
        names = dagmc_component_names(model)
    except RuntimeError:
        names = {}
    names.update(
        {
            volume_id: str(record["component_name"])
            for volume_id, record in ledger_by_id.items()
            if record.get("component_name")
        }
    )
    blocks = []
    for volume in sorted(model.volumes, key=lambda item: int(item.id)):
        volume_id = int(volume.id)
        name = names.get(volume_id, f"volume_{volume_id}")
        record = ledger_by_id.get(volume_id, {})
        kind = record.get("component_kind", _component_kind(name))
        cls = _component_class(name, record.get("component_class"))
        port_name = record.get("port_name")
        if port_name is None and name.endswith(("__void", "__liner")):
            port_name = name.rsplit("__", 1)[0]
        block = ParaViewBlock(
            component_name=name,
            component_kind=kind,
            component_class=cls,
            material_tag=str(
                record.get("material_tag", volume.material or "")
            ),
            dagmc_volume_id=volume_id,
            field_period_id=record.get("field_period_id", 0),
            instance_id=record.get("instance_id", f"dagmc-volume-{volume_id}"),
            port_name=port_name,
            hidden_by_default=cls == "graveyard",
        )
        triangles = np.asarray(volume.triangle_coords).reshape((-1, 3, 3))
        blocks.append(
            write_polydata(
                Path(block_dir)
                / f"dagmc_{volume_id:04d}_{_safe_leaf_name(name)}.vtp",
                triangles,
                block,
            )
        )
    return blocks


def _tag_value(mb, tag, entity, default=None):
    try:
        value = mb.tag_get_data(tag, [entity], flat=True)[0]
    except Exception:
        return default
    if isinstance(value, bytes):
        return value.decode(errors="replace").rstrip("\x00")
    return value.item() if hasattr(value, "item") else value


def _volume_mesh_blocks(h5m, block_dir):
    from pymoab import core, types

    mb = core.Core()
    mb.load_file(str(h5m))
    tags = {}
    for name in ("NAME", "MATERIAL", "GLOBAL_ID", "CATEGORY"):
        try:
            tags[name] = mb.tag_get_handle(name)
        except Exception:
            tags[name] = None
    blocks = []
    sets = mb.get_entities_by_type(0, types.MBENTITYSET)
    for meshset in sets:
        tetrahedra = list(mb.get_entities_by_type(meshset, types.MBTET))
        if not tetrahedra:
            continue
        name = str(
            _tag_value(mb, tags["NAME"], meshset, f"region_{len(blocks)+1}")
        )
        material = str(_tag_value(mb, tags["MATERIAL"], meshset, ""))
        region_id = int(
            _tag_value(mb, tags["GLOBAL_ID"], meshset, len(blocks) + 1)
        )
        handles = np.asarray(
            sorted(
                {
                    int(vertex)
                    for tet in tetrahedra
                    for vertex in mb.get_connectivity(tet)
                }
            ),
            dtype=np.uint64,
        )
        index = {int(handle): offset for offset, handle in enumerate(handles)}
        points = np.asarray(mb.get_coords(handles)).reshape((-1, 3))
        connectivity = np.asarray(
            [
                [index[int(vertex)] for vertex in mb.get_connectivity(tet)]
                for tet in tetrahedra
            ],
            dtype=np.int64,
        )
        cls = _component_class(name)
        block = ParaViewBlock(
            component_name=name,
            component_kind=_component_kind(name),
            component_class=cls,
            material_tag=material,
            dagmc_volume_id=region_id,
            field_period_id=0,
            instance_id=f"volume-mesh-region-{region_id}",
            port_name=(
                name.rsplit("__", 1)[0]
                if name.endswith(("__void", "__liner"))
                else None
            ),
            transport_geometry=False,
            volumetric_geometry=True,
        )
        blocks.append(
            write_unstructured_grid(
                Path(block_dir)
                / f"mesh_{region_id:04d}_{_safe_leaf_name(name)}.vtu",
                points,
                connectivity,
                block,
            )
        )
    return sorted(blocks, key=lambda block: block.dagmc_volume_id or 0)


def write_vtm(path, blocks: Iterable[ParaViewBlock], root_name="geometry"):
    """Write one named VTM referencing prewritten VTP/VTU leaves."""

    path = Path(path)
    blocks = tuple(blocks)
    root = ET.Element(
        "VTKFile",
        {
            "type": "vtkMultiBlockDataSet",
            "version": "1.0",
            "byte_order": "LittleEndian",
        },
    )
    dataset = ET.SubElement(root, "vtkMultiBlockDataSet")
    group = ET.SubElement(dataset, "Block", {"index": "0", "name": root_name})
    for index, block in enumerate(blocks):
        if block.file is None:
            raise ValueError(
                f"ParaView block {block.component_name!r} has no file"
            )
        relative = os.path.relpath(block.file, path.parent).replace(
            os.sep, "/"
        )
        ET.SubElement(
            group,
            "DataSet",
            {
                "index": str(index),
                "name": block.component_name,
                "file": relative,
            },
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def _file_record(path, root):
    path = Path(path)
    return {
        "path": os.path.relpath(path, root).replace(os.sep, "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256(path.read_bytes()).hexdigest(),
    }


def _render_script_text():
    colors = json.dumps(COMPONENT_COLORS, sort_keys=True)
    images = json.dumps(REQUIRED_IMAGE_NAMES)
    return f"""from pathlib import Path
import json
from paraview.simple import *

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "full_reactor_manifest.json").read_text())
colors = {colors}
image_names = {images}
sources = []
for block in manifest["dagmc_blocks"] + manifest.get("volume_mesh_blocks", []):
    path = ROOT / block["file"]
    reader = XMLPolyDataReader(FileName=[str(path)]) if path.suffix == ".vtp" else XMLUnstructuredGridReader(FileName=[str(path)])
    reader.UpdatePipeline()
    sources.append((reader, block))

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [1920, 1080]
view.OrientationAxesVisibility = 1
view.Background = [1.0, 1.0, 1.0]
visible = []
for source, block in sources:
    display = Show(source, view)
    display.Representation = "Surface"
    display.DiffuseColor = colors[block["component_class"]]
    display.Opacity = 0.22 if block["component_class"] in ("plasma", "chamber", "port_liner") else 1.0
    if block["hidden_by_default"] or block["volumetric_geometry"]:
        Hide(source, view)
    else:
        visible.append((source, block))

label = Text()
label.Text = manifest["annotation"]
label_display = Show(label, view)
label_display.WindowLocation = "Upper Left Corner"
label_display.Color = [0.0, 0.0, 0.0]
legend = Text()
legend.Text = "Legend: plasma/chamber blue | first wall gray | breeder green | shield yellow | vessel slate | void cyan | liner orange | conductor red | casing dark red\\nUnits: cm"
legend_display = Show(legend, view)
legend_display.WindowLocation = "Lower Left Corner"
legend_display.Color = [0.0, 0.0, 0.0]

Render(view)
ResetCamera(view)
camera = GetActiveCamera()
for image_name in image_names:
    lowered = image_name.lower()
    ResetCamera(view)
    if "top" in lowered:
        camera.SetPosition(0, 0, 1); camera.SetViewUp(0, 1, 0)
    elif "side" in lowered:
        camera.SetPosition(0, -1, 0); camera.SetViewUp(0, 0, 1)
    else:
        camera.Azimuth(35); camera.Elevation(22)
    Render(view)
    SaveScreenshot(str(ROOT / image_name), view, ImageResolution=[1920, 1080])

SaveState(str(ROOT / "full_reactor.pvsm"))
bounds = []
for source, block in sources:
    current = source.GetDataInformation().GetBounds()
    bounds.append({{"component_name": block["component_name"], "bounds": list(current)}})
validation = {{
    "paraview_version": ".".join(str(value) for value in GetParaViewVersion().version),
    "expected_components": [block["component_name"] for _, block in sources],
    "graveyard_hidden": all(block["component_class"] != "graveyard" or block["hidden_by_default"] for _, block in sources),
    "camera_bounds": bounds,
    "images": image_names,
}}
validation["empty_physical_bounds"] = [
    record["component_name"]
    for record in bounds
    if record["bounds"][0] > record["bounds"][1]
]
if validation["empty_physical_bounds"]:
    raise RuntimeError(
        "ParaView loaded empty physical blocks: "
        + ", ".join(validation["empty_physical_bounds"])
    )
(ROOT / "full_reactor_paraview_validation.json").write_text(json.dumps(validation, indent=2) + "\\n")
"""


def export_paraview_bundle(
    output_dir,
    *,
    dagmc_h5m,
    volume_mesh_h5m=None,
    ledger=None,
    repository_sha=None,
    scope="sector_transport_model",
    reactor_metadata: Mapping | None = None,
    port_metadata: Mapping | None = None,
):
    """Export generic DAGMC/MOAB multiblock files and a batch render script."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    block_dir = output_dir / "paraview_blocks"
    dagmc_blocks = _dagmc_blocks(dagmc_h5m, block_dir, ledger=ledger)
    dagmc_vtm = write_vtm(
        output_dir / "full_reactor_dagmc.vtm", dagmc_blocks, "dagmc"
    )
    volume_blocks = []
    volume_vtm = output_dir / "full_reactor_volume_mesh.vtm"
    if volume_mesh_h5m is not None:
        volume_blocks = _volume_mesh_blocks(volume_mesh_h5m, block_dir)
        write_vtm(volume_vtm, volume_blocks, "volume_mesh")
    else:
        write_vtm(volume_vtm, (), "volume_mesh")
    manifest = {
        "schema_version": "1.0",
        "repository_sha": repository_sha,
        "scope": scope,
        "reactor": dict(reactor_metadata or {}),
        "port": dict(port_metadata or {}),
        "annotation": (
            f"ParaStell | {scope} | port: "
            f"{dict(port_metadata or {}).get('name', 'none')} | "
            f"field periods: {dict(reactor_metadata or {}).get('field_periods', 'unknown')} | "
            f"SHA: {repository_sha or 'unknown'}"
        ),
        "dagmc_blocks": [asdict(block) for block in dagmc_blocks],
        "volume_mesh_blocks": [asdict(block) for block in volume_blocks],
        "graveyard_hidden_by_default": all(
            block.component_class != "graveyard" or block.hidden_by_default
            for block in dagmc_blocks
        ),
        "required_images": list(REQUIRED_IMAGE_NAMES),
    }
    for collection in (
        manifest["dagmc_blocks"],
        manifest["volume_mesh_blocks"],
    ):
        for block in collection:
            block["file"] = os.path.relpath(block["file"], output_dir).replace(
                os.sep, "/"
            )
    manifest_path = output_dir / "full_reactor_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    render_script = output_dir / "full_reactor_render.py"
    render_script.write_text(_render_script_text())
    files = [dagmc_vtm, volume_vtm, manifest_path, render_script]
    manifest["generated_files"] = [
        _file_record(path, output_dir) for path in files
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run_paraview_batch(output_dir, executable=None):
    """Run the generated renderer using ParaView's off-screen batch CLI."""

    output_dir = Path(output_dir)
    resolved = discover_paraview_executables()
    pvbatch = str(executable or resolved.pvbatch or "")
    if not pvbatch:
        raise FileNotFoundError("pvbatch was not found")
    command = [
        pvbatch,
        "--force-offscreen-rendering",
        str(output_dir / "full_reactor_render.py"),
    ]
    completed = subprocess.run(
        command,
        cwd=output_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (output_dir / "paraview_batch_output.txt").write_text(completed.stdout)
    if completed.returncode:
        raise RuntimeError(
            f"ParaView batch rendering failed with {completed.returncode}:\n"
            f"{completed.stdout}"
        )
    return command
