"""Portable blanket-material recipes for parametric ParaStell transport.

The recipes are distilled from the read-only BlanketNeutronics and
radial_build_tools references, but execution depends only on a hash-bound
fusion-material-db JSON file.  Global magnets remain homogenized; this module
does not create tape, cable, casing, or other local engineering geometry.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping
import xml.etree.ElementTree as ET


SCHEMA = "parastell.blanket_material_bundle/v1.0.0"
COMPONENT_ROLES = (
    "first_wall",
    "breeder",
    "back_wall",
    "high_temperature_shield",
    "vacuum_vessel",
    "low_temperature_shield",
    "homogenized_magnet",
)

REFERENCE_SOURCES = (
    {
        "repository": "https://github.com/StellaratorOptimization/BlanketNeutronics.git",
        "revision": "bc4ab3d0f27369d4eda908a3fc187a10b6c7fedb",
        "path": "neutronics_wistelld_DCLL_1/1_makeMaterials.py",
        "sha256": "de9ef612601d31fe52916ac19881635b6706f83d317957ecac46a866acdd0ef2",
        "role": "WISTELL-D_DCLL_material_reference",
    },
    {
        "repository": "https://github.com/FusionSandwich/radial_build_tools.git",
        "revision": "d195d5a9f777c3ac42a9033646937558e07748e0",
        "path": "examples/dcll_hcpb_examples/dcll_radial_build_example/dcll_materials.py",
        "sha256": "a7fdb8ac1b78151fa54e6cd248344b7f5a6560da2b8f0cd4631a0a0f8d4d4589",
        "role": "portable_DCLL_recipe_reference",
    },
    {
        "repository": "https://github.com/FusionSandwich/radial_build_tools.git",
        "revision": "d195d5a9f777c3ac42a9033646937558e07748e0",
        "path": "examples/dcll_hcpb_examples/hcpb_radial_build_example/HCPB_Mix_Materials.py",
        "sha256": "050a9fa5bc3c875dfd61f59f22bf7fdc312a90014b8c4cfd3112cba1c5d06131",
        "role": "portable_HCPB_recipe_reference",
    },
)


RECIPES: dict[str, dict[str, Any]] = {
    "dcll_rafs_helium_first_wall": {
        "volume_fractions": {"MF82H": 0.34, "HeNIST": 0.66},
        "citation": "DavisFusEngDes_2018",
    },
    "hcpb_tungsten_eurofer_first_wall": {
        "volume_fractions": {"W": 0.1, "EUROFER97": 0.9},
        "citation": "ZhouEUDEMOHCPB_2023",
    },
    "dcll_pb157li90_breeder": {
        "volume_fractions": {
            "Pb157Li90": 0.77,
            "MF82H": 0.06,
            "HeT410P80": 0.135,
            "SiC": 0.035,
        },
        "citation": "EliasUWFMD1424_2015 and MadaniUWFDM1423_2015",
    },
    "hcpb_beryllide_ceramic_breeder": {
        "volume_fractions": {
            "Be12Ti": 0.8,
            "Li4SiO4Li60.0": 0.14837,
            "Li2TiO3Li60.0": 0.05163,
        },
        "citation": "ZhouEUDEMOHCPB_2023",
    },
    "rafs_helium_back_wall": {
        "volume_fractions": {"MF82H": 0.8, "HeT410P80": 0.2},
        "citation": "DavisFusEngDes_2018",
    },
    "dcll_wc_rafs_helium_shield": {
        "volume_fractions": {
            "MF82H": 0.28,
            "WC": 0.52,
            "HeT410P80": 0.2,
        },
        "citation": "ElGuebalyFusSciTec_2017 and BlanketNeutronics DCLL reference",
    },
    "hcpb_borated_rafs_helium_shield": {
        "volume_fractions": {
            "MF82H": 0.2,
            "HeNIST": 0.28,
            "BMF82H": 0.52,
        },
        "citation": "DavisFusEngDes_2018",
    },
    "dcll_homogenized_vacuum_vessel": {
        "volume_fractions": {
            "SS316L": 0.2,
            "SS316LN": 0.2,
            "Cr3FS": 0.36,
            "HeNIST": 0.24,
        },
        "citation": "BlanketNeutronics 2/6/2 cm vessel plate-fill-plate homogenization",
    },
    "hcpb_homogenized_vacuum_vessel": {
        "volume_fractions": {"Cr3FS": 0.6, "HeNIST": 0.4},
        "citation": "DavisFusEngDes_2018",
    },
    "water_borated_rafs_low_temperature_shield": {
        "volume_fractions": {"Cr3FS": 0.39, "BMF82H": 0.29, "Water": 0.32},
        "citation": "DavisFusEngDes_2018",
    },
    "blanket_legacy_homogenized_hts_coil": {
        "volume_fractions": {
            "SS316L": 0.7746,
            "Cu": 0.1618,
            "Pb": 0.01392,
            "Sn": 0.02088,
            "HeT410P80": 0.0288,
        },
        "citation": "BlanketNeutronics WISTELL-D homogenized coil recipe",
        "note": "global homogenized transport proxy; no explicit REBCO tape layer",
    },
    "hcpb_homogenized_nb3sn_coil": {
        "volume_fractions": {
            "JK2LBSteel": 0.3,
            "Cu": 0.25,
            "TernaryNb3Sn": 0.25,
            "Eins": 0.1,
            "HeNIST": 0.1,
        },
        "citation": "DavisFusEngDes_2018",
        "note": "global homogenized transport proxy; not a REBCO local model",
    },
}


PRESETS = {
    "dcll": {
        "first_wall": "dcll_rafs_helium_first_wall",
        "breeder": "dcll_pb157li90_breeder",
        "back_wall": "rafs_helium_back_wall",
        "high_temperature_shield": "dcll_wc_rafs_helium_shield",
        "vacuum_vessel": "dcll_homogenized_vacuum_vessel",
        "low_temperature_shield": "water_borated_rafs_low_temperature_shield",
        "homogenized_magnet": "blanket_legacy_homogenized_hts_coil",
    },
    "hcpb": {
        "first_wall": "hcpb_tungsten_eurofer_first_wall",
        "breeder": "hcpb_beryllide_ceramic_breeder",
        "back_wall": "rafs_helium_back_wall",
        "high_temperature_shield": "hcpb_borated_rafs_helium_shield",
        "vacuum_vessel": "hcpb_homogenized_vacuum_vessel",
        "low_temperature_shield": "water_borated_rafs_low_temperature_shield",
        "homogenized_magnet": "hcpb_homogenized_nb3sn_coil",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _normalized_composition(
    row: Mapping[str, Any], name: str
) -> dict[str, float]:
    density = float(row.get("density", math.nan))
    composition = row.get("comp")
    if (
        not math.isfinite(density)
        or density <= 0.0
        or not isinstance(composition, Mapping)
    ):
        raise ValueError(f"pure material {name} is malformed")
    result = {str(key): float(value) for key, value in composition.items()}
    if any(
        not math.isfinite(value) or value < 0.0 for value in result.values()
    ):
        raise ValueError(f"pure material {name} composition is invalid")
    total = sum(result.values())
    if not math.isclose(total, 1.0, rel_tol=1.0e-8, abs_tol=1.0e-10):
        result = {key: value / total for key, value in result.items()}
    return result


def _resolve_recipe(
    pure: Mapping[str, Any],
    recipe_id: str,
    material_id: str,
    temperature_K: float,
) -> dict[str, Any]:
    recipe = RECIPES[recipe_id]
    fractions = recipe["volume_fractions"]
    total = sum(float(value) for value in fractions.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError(
            f"recipe {recipe_id} volume fractions do not sum to one"
        )
    density = 0.0
    isotope_mass: dict[str, float] = {}
    constituent_citations: dict[str, str | None] = {}
    for constituent, volume_fraction in fractions.items():
        if constituent not in pure:
            raise ValueError(
                f"recipe {recipe_id} requires missing material {constituent}"
            )
        row = pure[constituent]
        rho = float(row["density"])
        contribution = float(volume_fraction) * rho
        density += contribution
        for nuclide, mass_fraction in _normalized_composition(
            row, constituent
        ).items():
            isotope_mass[nuclide] = (
                isotope_mass.get(nuclide, 0.0) + contribution * mass_fraction
            )
        metadata = row.get("metadata", {})
        constituent_citations[constituent] = (
            str(metadata.get("citation"))
            if isinstance(metadata, Mapping)
            else None
        )
    isotopes = {
        key: value / density for key, value in sorted(isotope_mass.items())
    }
    material = {
        "material_id": material_id,
        "name": material_id,
        "recipe_id": recipe_id,
        "density_g_cm3": density,
        "temperature_K": float(temperature_K),
        "composition_basis": "mass_fraction",
        "isotopes": isotopes,
        "volume_fractions": dict(fractions),
        "mixture_citation": recipe["citation"],
        "constituent_citations": constituent_citations,
        "global_magnet_homogenized": material_id == "homogenized_magnet",
        "local_tape_geometry_created": False,
    }
    if "note" in recipe:
        material["note"] = recipe["note"]
    material["composition_sha256"] = _canonical_sha(
        {
            "density_g_cm3": density,
            "temperature_K": float(temperature_K),
            "isotopes": isotopes,
        }
    )
    return material


def build_blanket_material_bundle(
    pure_materials_path: str | Path,
    *,
    preset: str = "dcll",
    role_recipe_overrides: Mapping[str, str] | None = None,
    temperature_K: float = 293.6,
) -> dict[str, Any]:
    """Resolve a DCLL/HCPB or controlled role-substitution material bundle."""
    source = Path(pure_materials_path).resolve(strict=True)
    pure = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(pure, Mapping) or preset not in PRESETS:
        raise ValueError("pure material database or blanket preset is invalid")
    temperature = float(temperature_K)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("material temperature must be positive")
    selection = dict(PRESETS[preset])
    overrides = dict(role_recipe_overrides or {})
    if set(overrides) - set(COMPONENT_ROLES):
        raise ValueError(
            "material recipe override uses an unknown component role"
        )
    if any(recipe not in RECIPES for recipe in overrides.values()):
        raise ValueError("material recipe override is unknown")
    selection.update(overrides)
    materials = [
        _resolve_recipe(pure, selection[role], role, temperature)
        for role in COMPONENT_ROLES
    ]
    bundle = {
        "schema": SCHEMA,
        "status": "MATERIAL_BUNDLE_RESOLVED_EXECUTION_NOT_STARTED",
        "preset": preset,
        "role_recipe_overrides": overrides,
        "component_recipe_bindings": selection,
        "pure_material_source": {
            "path": str(source),
            "sha256": _sha256(source),
            "expected_reference_repository": "https://github.com/FusionSandwich/fusion-material-db.git",
            "expected_reference_revision": "9d8f84ad7b3ac2c587edfbe8ec1ed74891484498",
        },
        "reference_sources": [dict(row) for row in REFERENCE_SOURCES],
        "materials": materials,
        "physical_geometry_changed": False,
        "global_magnet_representation": "homogenized",
        "local_magnet_stack_owner": "downstream_local_model",
        "transport_or_production_authorized": False,
    }
    bundle["bundle_sha256"] = _canonical_sha(bundle)
    validate_blanket_material_bundle(bundle, verify_source=True)
    return bundle


def validate_blanket_material_bundle(
    bundle: Mapping[str, Any], *, verify_source: bool = False
) -> None:
    if bundle.get("schema") != SCHEMA or bundle.get("preset") not in PRESETS:
        raise ValueError("unsupported blanket material bundle")
    if bundle.get("physical_geometry_changed") is not False:
        raise ValueError("material selection cannot change physical geometry")
    if bundle.get("global_magnet_representation") != "homogenized":
        raise ValueError("global magnet must remain homogenized")
    materials = bundle.get("materials")
    if not isinstance(materials, list) or [
        row.get("material_id") for row in materials
    ] != list(COMPONENT_ROLES):
        raise ValueError("blanket material component inventory is invalid")
    for row in materials:
        isotopes = row.get("isotopes", {})
        if (
            row.get("composition_basis") != "mass_fraction"
            or not isinstance(isotopes, Mapping)
            or not math.isclose(
                sum(float(v) for v in isotopes.values()), 1.0, rel_tol=1.0e-8
            )
            or float(row.get("density_g_cm3", 0.0)) <= 0.0
            or row.get("local_tape_geometry_created") is not False
        ):
            raise ValueError("blanket material record is invalid")
    source = bundle.get("pure_material_source", {})
    if verify_source:
        path = Path(str(source.get("path", ""))).resolve(strict=True)
        if _sha256(path) != source.get("sha256"):
            raise ValueError("pure material source hash mismatch")
    expected = _canonical_sha(
        {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    )
    if bundle.get("bundle_sha256") != expected:
        raise ValueError("blanket material bundle hash is invalid")


def write_blanket_material_bundle(
    path: str | Path, bundle: Mapping[str, Any]
) -> Path:
    validate_blanket_material_bundle(bundle, verify_source=True)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(bundle, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return destination


def write_openmc_materials_xml(
    path: str | Path, bundle: Mapping[str, Any]
) -> Path:
    """Write deterministic, element-expanded OpenMC material XML."""
    validate_blanket_material_bundle(bundle, verify_source=True)
    root = ET.Element("materials")
    for material_index, row in enumerate(bundle["materials"], start=1):
        material = ET.SubElement(
            root,
            "material",
            {
                "id": str(material_index),
                "name": str(row["material_id"]),
                "temperature": format(float(row["temperature_K"]), ".16g"),
            },
        )
        ET.SubElement(
            material,
            "density",
            {
                "units": "g/cm3",
                "value": format(float(row["density_g_cm3"]), ".16g"),
            },
        )
        for nuclide, fraction in row["isotopes"].items():
            ET.SubElement(
                material,
                "nuclide",
                {"name": str(nuclide), "wo": format(float(fraction), ".17g")},
            )
    ET.indent(root, space="  ")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        ET.ElementTree(root).write(
            stream, encoding="utf-8", xml_declaration=True
        )
        stream.write(b"\n")
    return destination
