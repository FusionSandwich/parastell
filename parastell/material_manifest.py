"""Pinned, dependency-free adapter for generated fusion-material-db artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np


SCHEMA = "parastell.magnet_material_manifest/v1.0.0"
NUCLEAR_DATA_SCHEMA = "parastell.nuclear_data_audit/v1.0.0"
NUCLIDE_PATTERN = re.compile(r"^[A-Z][a-z]?\d+(?:_m\d+)?$")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_material(name: str, value: Mapping[str, Any]) -> dict[str, Any]:
    density = float(value["density"])
    if not np.isfinite(density) or density <= 0.0:
        raise ValueError(f"material {name!r} has invalid density")
    composition = {
        str(nuclide): float(fraction)
        for nuclide, fraction in value["comp"].items()
    }
    if not composition or any(
        not NUCLIDE_PATTERN.fullmatch(nuclide)
        or int(re.search(r"\d+", nuclide).group()) == 0
        for nuclide in composition
    ):
        raise ValueError(
            f"material {name!r} must use explicit transport nuclides; pseudo-nuclides are forbidden"
        )
    fractions = np.asarray(list(composition.values()), dtype=float)
    if np.any(fractions < 0.0) or not np.isclose(
        fractions.sum(), 1.0, atol=1.0e-8
    ):
        raise ValueError(
            f"material {name!r} isotope fractions must sum to one"
        )
    return {
        "name": name,
        "density_g_cm3": density,
        "temperature_K": float(value.get("temperature_K", 293.6)),
        "composition_basis": str(
            value.get("composition_basis", "mass_fraction")
        ),
        "nuclides": composition,
        "metadata": dict(value.get("metadata", {})),
        "atoms_per_molecule": float(value.get("atoms_per_molecule", -1.0)),
        "molecular_mass_g_mol": float(value.get("mass", 1.0)),
        "natural_elements_expanded": True,
    }


def _load_source(source: Mapping[str, Any], base_directory: Path):
    expanded_path = os.path.expandvars(str(source["path"]))
    if "$" in expanded_path:
        raise ValueError(
            f"material artifact path contains an unresolved environment variable: {expanded_path}"
        )
    path = Path(expanded_path)
    if not path.is_absolute():
        path = (base_directory / path).resolve()
    expected = str(source["sha256"])
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"material artifact checksum mismatch: {path}")
    if path.suffix.lower() != ".json":
        raise ValueError(
            "pinned material adapter currently consumes generated JSON"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("generated material JSON root must be a mapping")
    return path, data, actual


def _resolve_mixture(
    name: str,
    definition: Mapping[str, Any],
    materials: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    basis = str(definition.get("fraction_basis", "volume"))
    if basis != "volume":
        raise ValueError(
            "homogenized winding-pack inputs must use volume fractions"
        )
    fractions = {
        str(key): float(value)
        for key, value in definition["constituent_volume_fractions"].items()
    }
    values = np.asarray(list(fractions.values()), dtype=float)
    if np.any(values < 0.0) or not np.isclose(values.sum(), 1.0, atol=1.0e-10):
        raise ValueError(f"mixture {name!r} volume fractions must sum to one")
    missing = set(fractions) - set(materials)
    if missing:
        raise ValueError(
            f"mixture {name!r} has unknown constituents {sorted(missing)}"
        )
    densities = {
        key: float(materials[key]["density_g_cm3"]) for key in fractions
    }
    density = sum(fractions[key] * densities[key] for key in fractions)
    mass_fractions = {
        key: fractions[key] * densities[key] / density for key in fractions
    }
    nuclides: dict[str, float] = {}
    for material_name, material_fraction in mass_fractions.items():
        for nuclide, fraction in materials[material_name]["nuclides"].items():
            nuclides[nuclide] = nuclides.get(
                nuclide, 0.0
            ) + material_fraction * float(fraction)
    total = sum(nuclides.values())
    nuclides = {key: value / total for key, value in sorted(nuclides.items())}
    return {
        "name": name,
        "density_g_cm3": density,
        "temperature_K": float(definition.get("temperature_K", 293.6)),
        "composition_basis": "mass_fraction",
        "nuclides": nuclides,
        "natural_elements_expanded": True,
        "homogenization": {
            "input_basis": "volume_fraction",
            "constituent_volume_fractions": fractions,
            "constituent_density_g_cm3": densities,
            "derived_constituent_mass_fractions": mass_fractions,
            "density_mixing_rule": "ideal additive constituent volumes",
        },
        "metadata": {
            "source": str(
                definition.get("source", "explicit_project_assumption")
            ),
            "citation": str(
                definition.get("citation", "not_manufacturer_authoritative")
            ),
            "uncertainty_variant": str(
                definition.get("uncertainty_variant", name)
            ),
            "manufacturer_authoritative": False,
        },
    }


def resolve_material_manifest(
    config_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve selected generated records and explicit project mixtures."""
    config_source = Path(config_path).resolve()
    config_bytes = config_source.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    repository = dict(config["fusion_material_db"])
    sources = []
    raw_records: dict[str, Mapping[str, Any]] = {}
    for source in config["generated_artifacts"]:
        path, data, digest = _load_source(source, config_source.parent)
        selected = tuple(source["selected_materials"])
        missing = set(selected) - set(data)
        if missing:
            raise ValueError(
                f"generated artifact lacks materials {sorted(missing)}"
            )
        for name in selected:
            if name in raw_records:
                raise ValueError(
                    f"material {name!r} selected from multiple artifacts"
                )
            raw_records[name] = data[name]
        sources.append(
            {
                "path": str(path),
                "sha256": digest,
                "size_bytes": path.stat().st_size,
                "selected_materials": list(selected),
            }
        )
    for name, value in config.get("explicit_materials", {}).items():
        if name in raw_records:
            raise ValueError(
                f"explicit material {name!r} shadows generated record"
            )
        raw_records[name] = value
    materials = {
        name: _validate_material(name, value)
        for name, value in raw_records.items()
    }
    variants = {
        name: _resolve_mixture(name, definition, materials)
        for name, definition in config["winding_pack_variants"].items()
    }
    resolved = {**materials, **variants}
    tags = {
        str(key): str(value) for key, value in config["material_tags"].items()
    }
    missing_tags = set(tags.values()) - set(resolved)
    if missing_tags:
        raise ValueError(
            f"material tags reference unknown records {sorted(missing_tags)}"
        )
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "path": str(config_source),
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
        },
        "fusion_material_db": {
            "repository": repository["repository"],
            "commit": repository["commit"],
            "runtime_dependency": False,
        },
        "generated_artifacts": sources,
        "material_tags": tags,
        "materials": resolved,
        "assumption_policy": (
            "generated records are pinned; homogenized variants are explicit project assumptions, "
            "not manufacturer-authoritative conductor recipes"
        ),
        "pseudo_nuclides_present": False,
    }
    hash_binding = {
        "schema": manifest["schema"],
        "configuration_sha256": manifest["configuration"]["sha256"],
        "fusion_material_db": manifest["fusion_material_db"],
        "generated_artifacts": [
            {
                "sha256": item["sha256"],
                "size_bytes": item["size_bytes"],
                "selected_materials": item["selected_materials"],
            }
            for item in manifest["generated_artifacts"]
        ],
        "material_tags": manifest["material_tags"],
        "materials": manifest["materials"],
        "assumption_policy": manifest["assumption_policy"],
    }
    canonical = json.dumps(
        hash_binding, sort_keys=True, separators=(",", ":")
    ).encode()
    manifest["resolved_manifest_sha256"] = hashlib.sha256(
        canonical
    ).hexdigest()
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifest


def openmc_materials_from_manifest(
    manifest: Mapping[str, Any], material_tags: Sequence[str] | None = None
):
    """Create OpenMC materials lazily from immutable resolved records."""
    try:
        import openmc
    except ImportError as exc:
        raise RuntimeError(
            "OpenMC is required to construct transport materials"
        ) from exc
    selected_tags = tuple(material_tags or manifest["material_tags"])
    output = openmc.Materials()
    for tag in selected_tags:
        name = manifest["material_tags"][tag]
        record = manifest["materials"][name]
        material = openmc.Material(name=tag)
        material.set_density("g/cm3", float(record["density_g_cm3"]))
        material.temperature = float(record["temperature_K"])
        for nuclide, fraction in record["nuclides"].items():
            material.add_nuclide(nuclide, float(fraction), percent_type="wo")
        output.append(material)
    return output


def audit_nuclear_data(
    manifest: Mapping[str, Any],
    *,
    cross_sections_path: str | Path,
    approved_library: str,
    evaluation_release: str,
    photon_evaluation_release: str | None = None,
    approved_mixed_case: bool = False,
    temperature_method: str | None = None,
    temperature_tolerance_K: float | None = None,
) -> dict[str, Any]:
    """Audit every configured nuclide without silently mixing libraries."""
    source = Path(cross_sections_path).resolve()
    root = ET.parse(source).getroot()
    libraries = root.findall(".//library")
    available_neutron = {
        name
        for library in libraries
        if library.attrib.get("type") == "neutron"
        for name in str(library.attrib.get("materials", "")).split()
    }
    available_photon = {
        name
        for library in libraries
        if library.attrib.get("type") == "photon"
        for name in str(library.attrib.get("materials", "")).split()
    }
    neutron_library_paths = {
        name: Path(library.attrib["path"])
        for library in libraries
        if library.attrib.get("type") == "neutron"
        for name in str(library.attrib.get("materials", "")).split()
    }
    neutron_library_paths = {
        name: (path if path.is_absolute() else source.parent / path).resolve()
        for name, path in neutron_library_paths.items()
    }
    requested_temperatures = {}
    for material in manifest["materials"].values():
        temperature = float(material["temperature_K"])
        for nuclide in material["nuclides"]:
            requested_temperatures.setdefault(nuclide, set()).add(temperature)

    def available_temperatures_K(nuclide: str) -> list[float]:
        path = neutron_library_paths.get(nuclide)
        if path is None or not path.is_file():
            return []
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                "h5py is required for the nuclear-data temperature audit"
            ) from exc
        with h5py.File(path, "r") as data:
            group = data[nuclide]["kTs"]
            values = []
            for label in group:
                match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)K", label)
                if match:
                    values.append(float(match.group(1)))
            return sorted(values)

    if temperature_method not in {None, "nearest", "interpolation"}:
        raise ValueError("temperature_method must be nearest or interpolation")
    if temperature_method is not None and (
        temperature_tolerance_K is None or temperature_tolerance_K < 0.0
    ):
        raise ValueError(
            "a nonnegative temperature_tolerance_K is required with a "
            "temperature method"
        )
    nuclides = sorted(
        {
            nuclide
            for material in manifest["materials"].values()
            for nuclide in material["nuclides"]
        }
    )
    rows = []
    for nuclide in nuclides:
        present = nuclide in available_neutron
        element = re.match(r"[A-Z][a-z]?", nuclide).group()
        photon_present = element in available_photon
        requested = sorted(requested_temperatures.get(nuclide, ()))
        available_temperatures = (
            available_temperatures_K(nuclide) if present else []
        )
        temperature_fallback = False
        temperature_supported = True
        if temperature_method is not None:
            for temperature in requested:
                if not available_temperatures:
                    temperature_supported = False
                    continue
                exact = any(
                    np.isclose(temperature, candidate, atol=1.0e-8)
                    for candidate in available_temperatures
                )
                if exact:
                    continue
                temperature_fallback = True
                if temperature_method == "nearest":
                    temperature_supported &= min(
                        abs(temperature - candidate)
                        for candidate in available_temperatures
                    ) <= float(temperature_tolerance_K)
                else:
                    temperature_supported &= (
                        min(available_temperatures)
                        <= temperature
                        <= max(available_temperatures)
                    )
        rows.append(
            {
                "nuclide": nuclide,
                "neutron_data_available": present,
                "photon_data_available": photon_present,
                "photon_element": element,
                "evaluation_release": evaluation_release,
                "temperature_support": {
                    "requested_K": requested,
                    "available_K": available_temperatures,
                    "method": temperature_method or "audit_at_openmc_load",
                    "tolerance_K": temperature_tolerance_K,
                    "supported": temperature_supported,
                },
                "missing_reaction_data": (
                    [] if present else ["all_neutron_reactions"]
                ),
                "fallback_used": temperature_fallback,
            }
        )
    missing = [
        row["nuclide"] for row in rows if not row["neutron_data_available"]
    ]
    missing_photon = sorted(
        {
            row["photon_element"]
            for row in rows
            if not row["photon_data_available"]
        }
    )
    unsupported_temperatures = [
        row["nuclide"]
        for row in rows
        if not row["temperature_support"]["supported"]
    ]
    mixed = bool(
        photon_evaluation_release
        and photon_evaluation_release != evaluation_release
    )
    passes_library_policy = not mixed or approved_mixed_case
    result = {
        "schema": NUCLEAR_DATA_SCHEMA,
        "cross_sections_xml": {
            "path": str(source),
            "sha256": sha256(source),
        },
        "approved_library": approved_library,
        "evaluation_release": evaluation_release,
        "photon_evaluation_release": (
            photon_evaluation_release or evaluation_release
        ),
        "mixed_library": mixed,
        "mixed_library_explicitly_approved": bool(
            approved_mixed_case if mixed else False
        ),
        "temperature_policy": {
            "method": temperature_method,
            "tolerance_K": temperature_tolerance_K,
            "explicit": temperature_method is not None,
        },
        "fallback_used": any(row["fallback_used"] for row in rows),
        "nuclides": rows,
        "missing_nuclides": missing,
        "missing_photon_elements": missing_photon,
        "unsupported_temperature_nuclides": unsupported_temperatures,
        "status": (
            "PASS"
            if not missing
            and not missing_photon
            and not unsupported_temperatures
            and passes_library_policy
            else (
                "FAIL_UNAPPROVED_MIXED_LIBRARY"
                if not passes_library_policy
                else "FAIL_MISSING_NUCLEAR_DATA"
            )
        ),
    }
    return result


def deterministic_component_colors(
    component_names: Sequence[str],
    explicit_colors: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Assign stable RGB hex colors without random XKCD selection."""
    explicit = dict(explicit_colors or {})
    colors = {}
    for name in sorted(set(str(value) for value in component_names)):
        if name in explicit:
            color = explicit[name]
            if not re.fullmatch(r"#[0-9A-Fa-f]{6}", color):
                raise ValueError(
                    f"component color for {name!r} is not #RRGGBB"
                )
            colors[name] = color.lower()
            continue
        digest = hashlib.sha256(name.encode("utf-8")).digest()
        # Restrict channels to a readable mid-range and remain deterministic.
        rgb = tuple(48 + byte % 160 for byte in digest[:3])
        colors[name] = "#" + "".join(f"{value:02x}" for value in rgb)
    return colors
