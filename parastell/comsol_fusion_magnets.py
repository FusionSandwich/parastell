"""Open-source, COMSOL-ready geometry for fusion magnet studies.

The generators in this module do not require a COMSOL installation.  They
create neutral CAD files, machine-readable model manifests, and small COMSOL
Java import programs.  A COMSOL installation is still required to create and
solve the final ``.mph`` models.

The geometries are intentionally generic.  They are suitable for workflow and
method development, but they are not validated reactor designs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class FusionMagnetDesign(str, Enum):
    """Supported generic fusion-magnet model families."""

    TOKAMAK_TF_D_SHAPE = "tokamak_tf_d_shape"
    SPHERICAL_TOKAMAK_TF = "spherical_tokamak_tf"
    CENTRAL_SOLENOID_PF = "central_solenoid_pf"
    DEMOUNTABLE_TF_JOINT = "demountable_tf_joint"


@dataclass(frozen=True)
class DShapedCoilParameters:
    """Dimensions for a D-shaped toroidal-field coil pack."""

    inboard_radius_m: float
    outboard_radius_m: float
    half_height_m: float
    radial_build_m: float
    poloidal_build_m: float
    toroidal_depth_m: float
    curve_points: int = 96

    def __post_init__(self) -> None:
        values = (
            self.inboard_radius_m,
            self.outboard_radius_m,
            self.half_height_m,
            self.radial_build_m,
            self.poloidal_build_m,
            self.toroidal_depth_m,
        )
        if any(value <= 0.0 for value in values):
            raise ValueError("all D-shaped coil dimensions must be positive")
        if self.inboard_radius_m >= self.outboard_radius_m:
            raise ValueError("inboard radius must be below outboard radius")
        if self.radial_build_m * 2.0 >= (
            self.outboard_radius_m - self.inboard_radius_m
        ):
            raise ValueError("radial build leaves no D-shaped aperture")
        if self.poloidal_build_m >= self.half_height_m:
            raise ValueError("poloidal build leaves no D-shaped aperture")
        if self.curve_points < 24:
            raise ValueError("curve_points must be at least 24")


@dataclass(frozen=True)
class CentralSolenoidPFParameters:
    """Dimensions for a modular central-solenoid and PF-coil surrogate."""

    cs_inner_radius_m: float = 0.45
    cs_radial_build_m: float = 0.28
    cs_module_height_m: float = 0.55
    cs_module_gap_m: float = 0.06
    cs_modules: int = 6
    pf_inner_radius_m: float = 2.1
    pf_radial_build_m: float = 0.35
    pf_height_m: float = 0.35
    pf_axial_offset_m: float = 2.35

    def __post_init__(self) -> None:
        values = (
            self.cs_inner_radius_m,
            self.cs_radial_build_m,
            self.cs_module_height_m,
            self.cs_module_gap_m,
            self.pf_inner_radius_m,
            self.pf_radial_build_m,
            self.pf_height_m,
            self.pf_axial_offset_m,
        )
        if any(value <= 0.0 for value in values):
            raise ValueError("all CS/PF dimensions must be positive")
        if self.cs_modules < 1:
            raise ValueError("cs_modules must be positive")


@dataclass(frozen=True)
class DemountableJointParameters:
    """Joint dimensions added to a generic D-shaped TF coil."""

    coil: DShapedCoilParameters
    joint_radial_span_m: float = 0.65
    joint_poloidal_span_m: float = 0.50
    contact_thickness_m: float = 0.012

    def __post_init__(self) -> None:
        if self.joint_radial_span_m <= 0.0:
            raise ValueError("joint radial span must be positive")
        if self.joint_poloidal_span_m <= 0.0:
            raise ValueError("joint poloidal span must be positive")
        if not 0.0 < self.contact_thickness_m < self.joint_poloidal_span_m:
            raise ValueError("contact thickness must fit inside the joint")


@dataclass
class FusionMagnetCadModel:
    """Neutral-CAD representation and COMSOL setup metadata."""

    design: FusionMagnetDesign
    parts: dict[str, Any]
    parameters: dict[str, Any]
    description: str
    recommended_physics: list[dict[str, str]]
    limitations: list[str]
    references: list[dict[str, str]]


def _cadquery() -> Any:
    try:
        import cadquery as cq
    except ImportError as exc:  # pragma: no cover - dependency is in ParaStell
        raise RuntimeError(
            "cadquery is required to generate COMSOL-ready CAD files"
        ) from exc
    return cq


def _d_outline_points(
    inboard_radius_m: float,
    outboard_radius_m: float,
    half_height_m: float,
    point_count: int,
) -> list[tuple[float, float]]:
    points = [
        (inboard_radius_m, -half_height_m),
        (inboard_radius_m, half_height_m),
    ]
    for index in range(1, point_count + 1):
        angle = math.pi * index / point_count
        radius = inboard_radius_m + (
            outboard_radius_m - inboard_radius_m
        ) * math.sin(angle)
        height = half_height_m * math.cos(angle)
        points.append((radius, height))
    return points


def _build_d_shaped_coil(
    parameters: DShapedCoilParameters,
) -> Any:
    cq = _cadquery()
    outer_points = _d_outline_points(
        parameters.inboard_radius_m,
        parameters.outboard_radius_m,
        parameters.half_height_m,
        parameters.curve_points,
    )
    inner_points = _d_outline_points(
        parameters.inboard_radius_m + parameters.radial_build_m,
        parameters.outboard_radius_m - parameters.radial_build_m,
        parameters.half_height_m - parameters.poloidal_build_m,
        parameters.curve_points,
    )
    outer = (
        cq.Workplane("XY")
        .polyline(outer_points)
        .close()
        .extrude(parameters.toroidal_depth_m / 2.0, both=True)
    )
    cutter = (
        cq.Workplane("XY")
        .polyline(inner_points)
        .close()
        .extrude(parameters.toroidal_depth_m, both=True)
    )
    return outer.cut(cutter)


def _build_tokamak_tf_model() -> FusionMagnetCadModel:
    parameters = DShapedCoilParameters(
        inboard_radius_m=0.95,
        outboard_radius_m=4.00,
        half_height_m=3.10,
        radial_build_m=0.38,
        poloidal_build_m=0.38,
        toroidal_depth_m=0.55,
    )
    coil = _build_d_shaped_coil(parameters)
    return FusionMagnetCadModel(
        design=FusionMagnetDesign.TOKAMAK_TF_D_SHAPE,
        parts={"tf_coil_pack": coil},
        parameters=asdict(parameters),
        description=(
            "Generic D-shaped tokamak toroidal-field coil pack for "
            "magnetostatic and Lorentz-load studies."
        ),
        recommended_physics=[
            {
                "interface": "Magnetic Fields (mf)",
                "study": "Stationary",
                "purpose": "Field and Lorentz-force calculation",
            },
            {
                "interface": "Solid Mechanics",
                "study": "Stationary",
                "purpose": "Coil-pack and case stress response",
            },
            {
                "interface": "Heat Transfer in Solids",
                "study": "Stationary or Time Dependent",
                "purpose": "Nuclear-heating and cryogenic margin",
            },
        ],
        limitations=[
            "The D profile is a smooth polygonal approximation.",
            "The winding pack is homogenized and contains no turn detail.",
            "No reactor-specific dimensions or proprietary geometry are used.",
        ],
        references=[
            {
                "title": (
                    "Analysis of D-Shaped Toroidal Superconductive Coils "
                    "for Medium Size Fusion Experiment Facility"
                ),
                "url": "https://www.comsol.com/paper/63511",
                "use": "model-family and multiphysics-method precedent",
            },
            {
                "title": (
                    "Magneto-structural Analysis of Fusion grade "
                    "Superconducting Toroidal Field Coils"
                ),
                "url": "https://www.comsol.com/paper/7362",
                "use": "magnetostatic-to-structural coupling precedent",
            },
        ],
    )


def _build_spherical_tokamak_tf_model() -> FusionMagnetCadModel:
    parameters = DShapedCoilParameters(
        inboard_radius_m=0.22,
        outboard_radius_m=2.35,
        half_height_m=2.55,
        radial_build_m=0.24,
        poloidal_build_m=0.30,
        toroidal_depth_m=0.42,
    )
    coil = _build_d_shaped_coil(parameters)
    cq = _cadquery()
    split_radius = parameters.inboard_radius_m + 2.2 * (
        parameters.radial_build_m
    )
    center_column_box = cq.Workplane("XY").box(
        split_radius,
        2.5 * parameters.half_height_m,
        2.0 * parameters.toroidal_depth_m,
        centered=(False, True, True),
    )
    center_column = coil.intersect(center_column_box)
    return_limb = coil.cut(center_column_box)
    model_parameters = asdict(parameters)
    model_parameters["center_column_split_radius_m"] = split_radius
    return FusionMagnetCadModel(
        design=FusionMagnetDesign.SPHERICAL_TOKAMAK_TF,
        parts={
            "tf_center_column": center_column,
            "tf_outer_return": return_limb,
        },
        parameters=model_parameters,
        description=(
            "Low-aspect-ratio TF-coil surrogate with a separately named "
            "center column for quasi-2.5D HTS-loss and irradiation studies."
        ),
        recommended_physics=[
            {
                "interface": "Magnetic Field Formulation (mfh)",
                "study": "Time Dependent",
                "purpose": "HTS current redistribution and hysteretic loss",
            },
            {
                "interface": "Magnetic Fields, No Currents (mfnc)",
                "study": "Time Dependent",
                "purpose": "Nonconducting background-field domain",
            },
            {
                "interface": "Heat Transfer in Solids",
                "study": "Time Dependent",
                "purpose": "AC-loss and nuclear-heating temperature rise",
            },
        ],
        limitations=[
            "The center column is a homogenized geometric partition.",
            "The model does not encode a specific VST or CORC cable layout.",
            "PF and CS background fields must be supplied separately.",
        ],
        references=[
            {
                "title": (
                    "Modelling superconductor AC losses in the STEP TF "
                    "magnet during plasma initiation"
                ),
                "url": "https://www.comsol.com/paper/146111",
                "use": "quasi-2.5D and H-H0-phi method precedent",
            },
            {
                "title": (
                    "A STEP in the Right Direction: Modeling AC Losses "
                    "in a Tokamak Design"
                ),
                "url": (
                    "https://www.comsol.com/story/"
                    "a-step-in-the-right-direction-modeling-ac-losses-"
                    "in-a-tokamak-design-152141"
                ),
                "use": "spherical-tokamak HTS workflow precedent",
            },
        ],
    )


def _annular_coil(
    inner_radius_m: float,
    radial_build_m: float,
    height_m: float,
    center_z_m: float,
) -> Any:
    cq = _cadquery()
    coil = (
        cq.Workplane("XY")
        .circle(inner_radius_m + radial_build_m)
        .circle(inner_radius_m)
        .extrude(height_m)
    )
    return coil.translate((0.0, 0.0, center_z_m - height_m / 2.0))


def _build_central_solenoid_pf_model() -> FusionMagnetCadModel:
    parameters = CentralSolenoidPFParameters()
    parts: dict[str, Any] = {}
    module_pitch = parameters.cs_module_height_m + parameters.cs_module_gap_m
    for index in range(parameters.cs_modules):
        center_z = (index - (parameters.cs_modules - 1) / 2.0) * module_pitch
        parts[f"cs_module_{index + 1:02d}"] = _annular_coil(
            parameters.cs_inner_radius_m,
            parameters.cs_radial_build_m,
            parameters.cs_module_height_m,
            center_z,
        )
    parts["pf_upper"] = _annular_coil(
        parameters.pf_inner_radius_m,
        parameters.pf_radial_build_m,
        parameters.pf_height_m,
        parameters.pf_axial_offset_m,
    )
    parts["pf_lower"] = _annular_coil(
        parameters.pf_inner_radius_m,
        parameters.pf_radial_build_m,
        parameters.pf_height_m,
        -parameters.pf_axial_offset_m,
    )
    return FusionMagnetCadModel(
        design=FusionMagnetDesign.CENTRAL_SOLENOID_PF,
        parts=parts,
        parameters=asdict(parameters),
        description=(
            "Axisymmetric modular central-solenoid and poloidal-field "
            "coil surrogate for ramp-field and feeder calculations."
        ),
        recommended_physics=[
            {
                "interface": "Magnetic Fields (mf), 2D Axisymmetric",
                "study": "Stationary or Time Dependent",
                "purpose": "CS/PF background-field calculation",
            },
            {
                "interface": "Electrical Circuit",
                "study": "Time Dependent",
                "purpose": "Current-ramp and dump-circuit coupling",
            },
            {
                "interface": "Heat Transfer in Solids",
                "study": "Time Dependent",
                "purpose": "Pulsed loss and nuclear-heating response",
            },
        ],
        limitations=[
            "Each module is a homogenized annular winding pack.",
            "Plasma-current and ferromagnetic-structure effects are omitted.",
            "The default dimensions are generic workflow-test values.",
        ],
        references=[
            {
                "title": (
                    "Electromagnetic Analysis of the Superconducting "
                    "Magnet System of the Divertor Tokamak Test Facility"
                ),
                "url": "https://www.comsol.com/paper/94651",
                "use": "3D TF plus 2D-axisymmetric PF/CS decomposition",
            }
        ],
    )


def _build_demountable_joint_model() -> FusionMagnetCadModel:
    coil_parameters = DShapedCoilParameters(
        inboard_radius_m=0.80,
        outboard_radius_m=3.35,
        half_height_m=2.75,
        radial_build_m=0.34,
        poloidal_build_m=0.34,
        toroidal_depth_m=0.50,
    )
    parameters = DemountableJointParameters(coil=coil_parameters)
    coil = _build_d_shaped_coil(parameters.coil)
    cq = _cadquery()
    joint_center_x = parameters.coil.outboard_radius_m - (
        parameters.coil.radial_build_m / 2.0
    )
    joint_zone_box = (
        cq.Workplane("XY")
        .box(
            parameters.joint_radial_span_m,
            parameters.joint_poloidal_span_m,
            2.0 * parameters.coil.toroidal_depth_m,
        )
        .translate((joint_center_x, 0.0, 0.0))
    )
    joint_zone = coil.intersect(joint_zone_box)
    contact_box = (
        cq.Workplane("XY")
        .box(
            1.5 * parameters.joint_radial_span_m,
            parameters.contact_thickness_m,
            2.0 * parameters.coil.toroidal_depth_m,
        )
        .translate((joint_center_x, 0.0, 0.0))
    )
    contact_layer = joint_zone.intersect(contact_box)
    joint_blocks = joint_zone.cut(contact_box)
    coil_body = coil.cut(joint_zone_box)
    model_parameters = asdict(parameters)
    model_parameters["joint_center_x_m"] = joint_center_x
    return FusionMagnetCadModel(
        design=FusionMagnetDesign.DEMOUNTABLE_TF_JOINT,
        parts={
            "tf_coil_body": coil_body,
            "joint_blocks": joint_blocks,
            "joint_contact_layer": contact_layer,
        },
        parameters=model_parameters,
        description=(
            "Generic demountable outboard TF joint with an explicit thin "
            "contact layer for electro-thermal sensitivity studies."
        ),
        recommended_physics=[
            {
                "interface": "Electric Currents",
                "study": "Stationary or Time Dependent",
                "purpose": "Joint-current and contact-resistance losses",
            },
            {
                "interface": "Heat Transfer in Solids",
                "study": "Stationary or Time Dependent",
                "purpose": "Joint hot-spot and cryogenic margin",
            },
            {
                "interface": "Solid Mechanics",
                "study": "Stationary",
                "purpose": "Contact-pressure and Lorentz-load response",
            },
        ],
        limitations=[
            "The joint is a generic outboard butt-joint surrogate.",
            "Contact pressure and resistance laws must be supplied by users.",
            "No proprietary SPARC, ARC, or commercial joint geometry is used.",
        ],
        references=[
            {
                "title": (
                    "Evaluation of Shear Strength in Soldered and "
                    "Mechanical Lap Joints of High-Temperature "
                    "Superconducting Tapes Intended for a Remountable "
                    "Magnet"
                ),
                "url": (
                    "https://www.jstage.jst.go.jp/article/pfr/11/0/"
                    "11_2405065/_article"
                ),
                "use": "demountable REBCO lap-joint precedent",
            },
            {
                "title": (
                    "Soldered joints -- an essential component of "
                    "demountable high temperature superconducting "
                    "fusion magnets"
                ),
                "url": (
                    "https://scientific-publications.ukaea.uk/papers/"
                    "soldered-joints-an-essential-component-of-"
                    "demountable-high-temperature-superconducting-"
                    "fusion-magnets/"
                ),
                "use": "electro-thermal joint-model precedent",
            },
        ],
    )


def build_fusion_magnet_model(
    design: FusionMagnetDesign | str,
) -> FusionMagnetCadModel:
    """Build one generic fusion-magnet CAD model."""

    selected = FusionMagnetDesign(design)
    builders = {
        FusionMagnetDesign.TOKAMAK_TF_D_SHAPE: _build_tokamak_tf_model,
        FusionMagnetDesign.SPHERICAL_TOKAMAK_TF: (
            _build_spherical_tokamak_tf_model
        ),
        FusionMagnetDesign.CENTRAL_SOLENOID_PF: (
            _build_central_solenoid_pf_model
        ),
        FusionMagnetDesign.DEMOUNTABLE_TF_JOINT: (
            _build_demountable_joint_model
        ),
    }
    return builders[selected]()


def _class_name(design: FusionMagnetDesign) -> str:
    words = re.split(r"[^A-Za-z0-9]+", design.value)
    return "".join(word.capitalize() for word in words) + "Import"


def _java_import_program(
    design: FusionMagnetDesign,
    step_filename: str,
) -> str:
    class_name = _class_name(design)
    mph_filename = f"{design.value}.mph"
    return f"""import com.comsol.model.*;
import com.comsol.model.util.*;

public class {class_name} {{
  public static void main(String[] args) {{
    String geometryFile = args.length > 0 ? args[0] : "{step_filename}";
    String outputFile = args.length > 1 ? args[1] : "{mph_filename}";
    run(geometryFile, outputFile);
  }}

  public static Model run(String geometryFile, String outputFile) {{
    Model model = ModelUtil.create("Model");
    model.label("{mph_filename}");
    model.component().create("comp1");
    model.component("comp1").geom().create("geom1", 3);
    model.component("comp1").geom("geom1").lengthUnit("m");
    model.component("comp1").geom("geom1").create("imp1", "Import");
    model.component("comp1").geom("geom1").feature("imp1")
        .set("filename", geometryFile);
    model.component("comp1").geom("geom1").feature("imp1")
        .set("selresult", "on");
    model.component("comp1").geom("geom1").run();
    model.save(outputFile);
    return model;
  }}
}}
"""


def _shape_statistics(shape: Any) -> dict[str, float | int]:
    solids = shape.solids().vals()
    if not solids:
        raise ValueError("generated part contains no solid geometry")
    cq = _cadquery()
    combined = cq.Compound.makeCompound(solids)
    bounding_box = combined.BoundingBox()
    return {
        "solid_count": len(solids),
        "volume_m3": sum(solid.Volume() for solid in solids),
        "x_min_m": bounding_box.xmin,
        "x_max_m": bounding_box.xmax,
        "y_min_m": bounding_box.ymin,
        "y_max_m": bounding_box.ymax,
        "z_min_m": bounding_box.zmin,
        "z_max_m": bounding_box.zmax,
    }


def _manifest(
    model: FusionMagnetCadModel,
    exported_files: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "design": model.design.value,
        "units": "m",
        "description": model.description,
        "parameters": model.parameters,
        "files": exported_files,
        "parts": {
            name: _shape_statistics(shape)
            for name, shape in model.parts.items()
        },
        "recommended_comsol_physics": model.recommended_physics,
        "parastell_radiation_handoff": {
            "supported_inputs": [
                "energy-group neutron or photon flux",
                "signed boundary current by energy group",
                "nuclear heating",
                "species-resolved PKA source terms",
            ],
            "suggested_import": (
                "Interpolation functions, tables, or General Extrusion "
                "operators keyed by the ParaStell handoff coordinates"
            ),
        },
        "limitations": model.limitations,
        "method_references": model.references,
    }


def export_fusion_magnet_model(
    model: FusionMagnetCadModel,
    output_directory: str | Path,
    formats: Iterable[str] = ("step",),
) -> dict[str, Path]:
    """Export neutral CAD, a manifest, and a COMSOL Java importer."""

    cq = _cadquery()
    output_path = Path(output_directory).resolve() / model.design.value
    output_path.mkdir(parents=True, exist_ok=True)
    selected_formats = {item.lower() for item in formats}
    if not selected_formats:
        raise ValueError("at least one export format is required")
    unsupported = selected_formats - {"step", "stl"}
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"unsupported export formats: {names}")

    written: dict[str, Path] = {}
    exported_files: dict[str, Any] = {"parts": {}}
    if "step" in selected_formats:
        assembly = cq.Assembly(name=model.design.value)
        for name, shape in model.parts.items():
            assembly.add(shape, name=name)
        combined_step = output_path / f"{model.design.value}.step"
        cq.exporters.assembly.exportAssembly(
            assembly, str(combined_step), unit="M"
        )
        written["combined_step"] = combined_step
        exported_files["combined_step"] = combined_step.name

    for name, shape in model.parts.items():
        part_files: dict[str, str] = {}
        if "step" in selected_formats:
            part_step = output_path / f"{name}.step"
            part_assembly = cq.Assembly()
            part_assembly.add(shape, name=name)
            cq.exporters.assembly.exportAssembly(
                part_assembly, str(part_step), unit="M"
            )
            written[f"{name}_step"] = part_step
            part_files["step"] = part_step.name
        if "stl" in selected_formats:
            part_stl = output_path / f"{name}.stl"
            cq.exporters.export(shape, str(part_stl))
            written[f"{name}_stl"] = part_stl
            part_files["stl"] = part_stl.name
        exported_files["parts"][name] = part_files

    if "step" in selected_formats:
        step_filename = exported_files["combined_step"]
        java_path = output_path / f"{_class_name(model.design)}.java"
        java_path.write_text(
            _java_import_program(model.design, step_filename),
            encoding="utf-8",
        )
        written["comsol_java"] = java_path
        exported_files["comsol_java"] = java_path.name

    manifest_path = output_path / "model_manifest.json"
    manifest_path.write_text(
        json.dumps(_manifest(model, exported_files), indent=2) + "\n",
        encoding="utf-8",
    )
    written["manifest"] = manifest_path
    return written


def export_all_fusion_magnet_models(
    output_directory: str | Path,
    formats: Iterable[str] = ("step",),
) -> dict[FusionMagnetDesign, dict[str, Path]]:
    """Build and export all supported generic model families."""

    return {
        design: export_fusion_magnet_model(
            build_fusion_magnet_model(design),
            output_directory,
            formats,
        )
        for design in FusionMagnetDesign
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate open-source, COMSOL-ready fusion magnet CAD models"
        )
    )
    parser.add_argument(
        "--design",
        choices=["all", *(design.value for design in FusionMagnetDesign)],
        default="all",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("comsol_fusion_magnet_models"),
    )
    parser.add_argument(
        "--format",
        action="append",
        choices=("step", "stl"),
        dest="formats",
        help="Repeat to request more than one neutral CAD format.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""

    arguments = _parser().parse_args(argv)
    formats = tuple(arguments.formats or ("step",))
    if arguments.design == "all":
        outputs = export_all_fusion_magnet_models(
            arguments.output_dir,
            formats,
        )
        summary = {
            design.value: {name: str(path) for name, path in files.items()}
            for design, files in outputs.items()
        }
    else:
        model = build_fusion_magnet_model(arguments.design)
        files = export_fusion_magnet_model(
            model,
            arguments.output_dir,
            formats,
        )
        summary = {name: str(path) for name, path in files.items()}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
