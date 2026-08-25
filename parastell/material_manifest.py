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


def _validate_material(
    name: str,
    value: Mapping[str, Any],
    *,
    source_provenance: Mapping[str, Any],
) -> dict[str, Any]:
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
        "source_provenance": dict(source_provenance),
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
    designated_insulation = definition.get("designated_insulation_material")
    if designated_insulation is not None:
        designated_insulation = str(designated_insulation)
        if designated_insulation not in fractions:
            raise ValueError(
                f"mixture {name!r} designates insulation "
                f"{designated_insulation!r} but does not contain it"
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
            "designated_insulation_material": designated_insulation,
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
            "production_selected": bool(
                definition.get("production_selected", False)
            ),
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
    record_provenance: dict[str, dict[str, Any]] = {}
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
            record_provenance[name] = {
                "kind": "generated_artifact",
                "repository": repository["repository"],
                "repository_commit": repository["commit"],
                "artifact_path": str(source["path"]),
                "artifact_sha256": digest,
                "record_name": name,
            }
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
        record_provenance[name] = {
            "kind": "inline_configuration",
            "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "record_name": name,
        }
    materials = {
        name: _validate_material(
            name,
            value,
            source_provenance=record_provenance[name],
        )
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
    production_selection = dict(
        config.get("production_material_selection", {})
    )
    if production_selection:
        required = {"winding_pack", "insulation"}
        missing_selection = required - set(production_selection)
        if missing_selection:
            raise ValueError(
                "production material selection lacks "
                f"{sorted(missing_selection)}"
            )
        selected_pack = str(production_selection["winding_pack"])
        selected_insulation = str(production_selection["insulation"])
        if tags.get("winding_pack") != selected_pack:
            raise ValueError(
                "production winding-pack selection does not match its "
                "transport material tag"
            )
        if selected_pack not in variants:
            raise ValueError(
                "production winding pack is not a resolved mixture"
            )
        pack_homogenization = variants[selected_pack]["homogenization"]
        if (
            pack_homogenization["designated_insulation_material"]
            != selected_insulation
        ):
            raise ValueError(
                "production insulation does not match the winding-pack "
                "designated insulation"
            )
        insulation = materials.get(selected_insulation)
        if insulation is None:
            raise ValueError(
                "production insulation is not a resolved material"
            )
        declared_density = production_selection.get("insulation_density_g_cm3")
        if declared_density is not None and not np.isclose(
            float(declared_density),
            float(insulation["density_g_cm3"]),
            atol=1.0e-12,
        ):
            raise ValueError(
                "production insulation density disagrees with the pinned record"
            )
        production_selection["winding_pack"] = selected_pack
        production_selection["insulation"] = selected_insulation
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
        "production_material_selection": production_selection,
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
        "production_material_selection": manifest[
            "production_material_selection"
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
    requested_source_max_energy_eV: float | None = None,
    temperature_method: str | None = None,
    temperature_tolerance_K: float | None = None,
) -> dict[str, Any]:
    """Audit configured nuclides against their actual OpenMC HDF5 tables.

    Catalog presence alone is not treated as data coverage.  Each neutron and
    photoatomic table is opened, hash-bound, and inspected.  The requested
    source-energy domain is explicit so a lower-energy table cannot silently
    pass a fusion-spectrum audit.
    """
    source = Path(cross_sections_path).resolve()
    root = ET.parse(source).getroot()
    libraries = root.findall(".//library")

    def library_paths(data_type: str) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for library in libraries:
            if library.attrib.get("type") != data_type:
                continue
            path = Path(library.attrib["path"])
            if not path.is_absolute():
                path = source.parent / path
            path = path.resolve()
            for name in str(library.attrib.get("materials", "")).split():
                if name in paths and paths[name] != path:
                    raise ValueError(
                        f"cross_sections.xml maps {data_type} material "
                        f"{name!r} to multiple tables"
                    )
                paths[name] = path
        return paths

    neutron_library_paths = library_paths("neutron")
    photon_library_paths = library_paths("photon")
    requested_temperatures = {}
    for material in manifest["materials"].values():
        temperature = float(material["temperature_K"])
        for nuclide in material["nuclides"]:
            requested_temperatures.setdefault(nuclide, set()).add(temperature)

    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "h5py is required for the nuclear-data table audit"
        ) from exc

    digest_cache: dict[Path, str] = {}

    def table_file(path: Path | None) -> dict[str, Any]:
        if path is None:
            return {
                "path": None,
                "sha256": None,
                "size_bytes": None,
                "catalog_entry": False,
                "file_exists": False,
            }
        exists = path.is_file()
        if exists and path not in digest_cache:
            digest_cache[path] = sha256(path)
        return {
            "path": str(path),
            "sha256": digest_cache.get(path),
            "size_bytes": path.stat().st_size if exists else None,
            "catalog_entry": True,
            "file_exists": exists,
        }

    def temperature_from_label(label: str) -> float | None:
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)K", label)
        return float(match.group(1)) if match else None

    def decode_label(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if hasattr(value, "tobytes") and not isinstance(value, str):
            decoded = value.tobytes().rstrip(b"\x00")
            try:
                return decoded.decode("utf-8")
            except UnicodeDecodeError:
                pass
        return str(value)

    neutron_cache: dict[str, dict[str, Any]] = {}

    def inspect_neutron_table(nuclide: str) -> dict[str, Any]:
        if nuclide in neutron_cache:
            return neutron_cache[nuclide]
        path = neutron_library_paths.get(nuclide)
        result: dict[str, Any] = {
            "file": table_file(path),
            "readable": False,
            "temperatures_K": [],
            "energy_minimum_eV": None,
            "energy_maximum_eV": None,
            "reaction_mt_labels": [],
            "error": None,
        }
        if path is None or not path.is_file():
            neutron_cache[nuclide] = result
            return result
        try:
            with h5py.File(path, "r") as data:
                if nuclide not in data:
                    raise ValueError(
                        f"table lacks expected nuclide group {nuclide!r}"
                    )
                group = data[nuclide]
                temperatures: set[float] = set()
                if "kTs" in group:
                    for label in group["kTs"]:
                        temperature = temperature_from_label(str(label))
                        if temperature is not None:
                            temperatures.add(temperature)
                energy_minima = []
                energy_maxima = []
                if "energy" not in group:
                    raise ValueError("table lacks an incident-energy grid")
                energy_group = group["energy"]
                energy_items = (
                    energy_group.items()
                    if isinstance(energy_group, h5py.Group)
                    else (("unspecified", energy_group),)
                )
                for label, dataset in energy_items:
                    temperature = temperature_from_label(str(label))
                    if temperature is not None:
                        temperatures.add(temperature)
                    values = np.asarray(dataset[...], dtype=float)
                    values = values[np.isfinite(values)]
                    if values.size:
                        energy_minima.append(float(values.min()))
                        energy_maxima.append(float(values.max()))
                if not energy_maxima:
                    raise ValueError("table incident-energy grid is empty")
                reaction_parent = group.get("reactions", group)
                reactions = []
                for key in reaction_parent:
                    if not str(key).startswith("reaction_"):
                        continue
                    reaction = reaction_parent[key]
                    mt_value = reaction.attrs.get("mt")
                    if mt_value is None:
                        mt_value = str(key).split("_", maxsplit=1)[1]
                    mt = int(mt_value)
                    label = decode_label(
                        reaction.attrs.get("label", f"MT={mt}")
                    )
                    reactions.append({"mt": mt, "label": label})
                result.update(
                    {
                        "readable": True,
                        "temperatures_K": sorted(temperatures),
                        "energy_minimum_eV": min(energy_minima),
                        "energy_maximum_eV": max(energy_maxima),
                        "reaction_mt_labels": sorted(
                            reactions, key=lambda item: item["mt"]
                        ),
                    }
                )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        neutron_cache[nuclide] = result
        return result

    photon_cache: dict[str, dict[str, Any]] = {}

    def inspect_photon_table(element: str) -> dict[str, Any]:
        if element in photon_cache:
            return photon_cache[element]
        path = photon_library_paths.get(element)
        result: dict[str, Any] = {
            "file": table_file(path),
            "readable": False,
            "energy_minimum_eV": None,
            "energy_maximum_eV": None,
            "error": None,
        }
        if path is None or not path.is_file():
            photon_cache[element] = result
            return result
        try:
            with h5py.File(path, "r") as data:
                if element not in data or "energy" not in data[element]:
                    raise ValueError(
                        f"table lacks expected photoatomic group {element!r}"
                    )
                values = np.asarray(data[element]["energy"][...], dtype=float)
                values = values[np.isfinite(values)]
                if not values.size:
                    raise ValueError("photoatomic energy grid is empty")
                result.update(
                    {
                        "readable": True,
                        "energy_minimum_eV": float(values.min()),
                        "energy_maximum_eV": float(values.max()),
                    }
                )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        photon_cache[element] = result
        return result

    if temperature_method not in {None, "nearest", "interpolation"}:
        raise ValueError("temperature_method must be nearest or interpolation")
    if temperature_method is not None and (
        temperature_tolerance_K is None or temperature_tolerance_K < 0.0
    ):
        raise ValueError(
            "a nonnegative temperature_tolerance_K is required with a "
            "temperature method"
        )
    if requested_source_max_energy_eV is not None:
        requested_source_max_energy_eV = float(requested_source_max_energy_eV)
        if (
            not np.isfinite(requested_source_max_energy_eV)
            or requested_source_max_energy_eV <= 0.0
        ):
            raise ValueError(
                "requested_source_max_energy_eV must be finite and positive"
            )
    nuclides = sorted(
        {
            nuclide
            for material in manifest["materials"].values()
            for nuclide in material["nuclides"]
        }
    )
    photon_release = photon_evaluation_release or evaluation_release
    mixed = photon_release != evaluation_release
    if mixed:
        evaluation_classification = (
            "EXPLICIT_APPROVED_MIXED_EVALUATION"
            if approved_mixed_case
            else "UNAPPROVED_MIXED_EVALUATION"
        )
    else:
        evaluation_classification = "SINGLE_EVALUATION"
    rows = []
    for nuclide in nuclides:
        element = re.match(r"[A-Z][a-z]?", nuclide).group()
        neutron = inspect_neutron_table(nuclide)
        photon = inspect_photon_table(element)
        present = bool(neutron["readable"])
        photon_present = bool(photon["readable"])
        requested = sorted(requested_temperatures.get(nuclide, ()))
        available_temperatures = neutron["temperatures_K"]
        temperature_fallback = False
        temperature_supported = present and temperature_method is not None
        if temperature_supported:
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
        table_maximum = neutron["energy_maximum_eV"]
        source_energy_supported = bool(
            requested_source_max_energy_eV is not None
            and table_maximum is not None
            and table_maximum >= requested_source_max_energy_eV
        )
        reactions = neutron["reaction_mt_labels"]
        missing_reaction_data = []
        if not present:
            missing_reaction_data.append("all_neutron_reactions")
        elif not reactions:
            missing_reaction_data.append("no_reaction_channels_found")
        rows.append(
            {
                "nuclide": nuclide,
                "neutron_data_available": present,
                "photon_data_available": photon_present,
                "photon_element": element,
                "evaluation_release": evaluation_release,
                "evaluation": {
                    "neutron_release": evaluation_release,
                    "photon_release": photon_release,
                    "classification": evaluation_classification,
                    "basis": (
                        "explicit audit declarations hash-bound to the table "
                        "files; OpenMC HDF5 does not encode a portable "
                        "evaluation-release identifier"
                    ),
                },
                "neutron_table": neutron["file"],
                "photon_table": photon["file"],
                "table_read_errors": {
                    "neutron": neutron["error"],
                    "photon": photon["error"],
                },
                "incident_energy_support": {
                    "actual_minimum_eV": neutron["energy_minimum_eV"],
                    "actual_maximum_eV": table_maximum,
                    "requested_source_maximum_eV": (
                        requested_source_max_energy_eV
                    ),
                    "requested_source_range_supported": (
                        source_energy_supported
                    ),
                },
                "temperature_support": {
                    "requested_K": requested,
                    "available_K": available_temperatures,
                    "method": temperature_method or "not_declared",
                    "tolerance_K": temperature_tolerance_K,
                    "supported": temperature_supported,
                },
                "reaction_mt_labels": reactions,
                "reaction_channel_count": len(reactions),
                "missing_reaction_data": missing_reaction_data,
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
    unsupported_source_energies = [
        row["nuclide"]
        for row in rows
        if not row["incident_energy_support"][
            "requested_source_range_supported"
        ]
    ]
    missing_reactions = [
        row["nuclide"] for row in rows if row["missing_reaction_data"]
    ]
    passes_library_policy = not mixed or approved_mixed_case
    if not passes_library_policy:
        status = "FAIL_UNAPPROVED_MIXED_LIBRARY"
    elif requested_source_max_energy_eV is None:
        status = "FAIL_SOURCE_ENERGY_UNDECLARED"
    elif missing or missing_photon:
        status = "FAIL_MISSING_NUCLEAR_DATA"
    elif unsupported_source_energies:
        status = "FAIL_SOURCE_ENERGY_RANGE"
    elif unsupported_temperatures:
        status = "FAIL_TEMPERATURE_SUPPORT"
    elif missing_reactions:
        status = "FAIL_REACTION_SUPPORT"
    else:
        status = "PASS"

    rows_by_nuclide = {row["nuclide"]: row for row in rows}
    material_coverage = []
    for name, material in sorted(manifest["materials"].items()):
        material_rows = [
            rows_by_nuclide[nuclide] for nuclide in material["nuclides"]
        ]
        failed_nuclides = sorted(
            row["nuclide"]
            for row in material_rows
            if not row["neutron_data_available"]
            or not row["photon_data_available"]
            or not row["temperature_support"]["supported"]
            or not row["incident_energy_support"][
                "requested_source_range_supported"
            ]
            or row["missing_reaction_data"]
        )
        material_coverage.append(
            {
                "material": name,
                "temperature_K": float(material["temperature_K"]),
                "nuclides": sorted(material["nuclides"]),
                "failed_nuclides": failed_nuclides,
                "status": "PASS" if not failed_nuclides else "FAIL",
            }
        )
    result = {
        "schema": NUCLEAR_DATA_SCHEMA,
        "cross_sections_xml": {
            "path": str(source),
            "sha256": sha256(source),
        },
        "approved_library": approved_library,
        "evaluation_release": evaluation_release,
        "photon_evaluation_release": photon_release,
        "mixed_library": mixed,
        "mixed_library_explicitly_approved": bool(
            approved_mixed_case if mixed else False
        ),
        "evaluation_classification": evaluation_classification,
        "evaluation_policy": {
            "neutron_release": evaluation_release,
            "photon_release": photon_release,
            "classification": evaluation_classification,
            "internally_consistent": not mixed,
            "mixed_data_sensitivity_required": mixed,
        },
        "requested_source_energy": {
            "maximum_eV": requested_source_max_energy_eV,
            "declared": requested_source_max_energy_eV is not None,
            "all_neutron_tables_support_range": not (
                unsupported_source_energies
            ),
        },
        "temperature_policy": {
            "method": temperature_method,
            "tolerance_K": temperature_tolerance_K,
            "explicit": temperature_method is not None,
        },
        "fallback_used": any(row["fallback_used"] for row in rows),
        "nuclides": rows,
        "material_coverage": material_coverage,
        "missing_nuclides": missing,
        "missing_photon_elements": missing_photon,
        "missing_reaction_nuclides": missing_reactions,
        "unsupported_source_energy_nuclides": unsupported_source_energies,
        "unsupported_temperature_nuclides": unsupported_temperatures,
        "status": status,
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
