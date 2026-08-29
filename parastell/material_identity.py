"""Portable, provenance-preserving material identity adapters.

The adapters deliberately avoid a runtime dependency on PyNE or on private
BlanketNeutronics code.  They accept the public fusion-material-db JSON shape
and OpenMC ``materials.xml`` while retaining the original composition basis.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping
import xml.etree.ElementTree as ET


SCHEMA = "parastell.material_identity/v1.0.0"
SOURCE_KINDS = {"fusion-material-db-json", "openmc-materials-xml"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _validate_composition(
    composition: Mapping[str, Any], basis: str
) -> dict[str, float]:
    if basis not in {"atom_fraction", "mass_fraction"}:
        raise ValueError("material composition basis must be explicit")
    if not isinstance(composition, Mapping) or not composition:
        raise ValueError("material isotopic composition is empty")
    values: dict[str, float] = {}
    for nuclide, fraction in composition.items():
        name = str(nuclide).strip()
        value = float(fraction)
        if not name or not math.isfinite(value) or value < 0.0:
            raise ValueError("material composition contains an invalid entry")
        values[name] = value
    total = sum(values.values())
    if total <= 0.0 or not math.isclose(
        total, 1.0, rel_tol=1.0e-5, abs_tol=1.0e-8
    ):
        raise ValueError("material fractions must sum to one")
    return values


def _material_record(
    *,
    material_id: str,
    name: str,
    density_g_cm3: float,
    temperature_K: float | None,
    composition: Mapping[str, Any],
    composition_basis: str,
    source: Mapping[str, Any],
    citation: str | None = None,
) -> dict[str, Any]:
    identifier = str(material_id).strip()
    density = float(density_g_cm3)
    if not identifier or not math.isfinite(density) or density <= 0.0:
        raise ValueError("material ID and positive density are required")
    temperature = None if temperature_K is None else float(temperature_K)
    if temperature is not None and (
        not math.isfinite(temperature) or temperature <= 0.0
    ):
        raise ValueError("material temperature must be positive")
    isotopes = _validate_composition(composition, composition_basis)
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "material_id": identifier,
        "name": str(name).strip() or identifier,
        "density_g_cm3": density,
        "temperature_K": temperature,
        "composition_basis": composition_basis,
        "isotopes": isotopes,
        "citation": None if citation is None else str(citation),
        "source": dict(source),
    }
    record["composition_sha256"] = _canonical_sha256(
        {
            "density_g_cm3": density,
            "temperature_K": temperature,
            "composition_basis": composition_basis,
            "isotopes": isotopes,
        }
    )
    record["record_sha256"] = _canonical_sha256(record)
    return record


def validate_material_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one portable material-identity record."""
    expected_keys = {
        "schema",
        "material_id",
        "name",
        "density_g_cm3",
        "temperature_K",
        "composition_basis",
        "isotopes",
        "citation",
        "source",
        "composition_sha256",
        "record_sha256",
    }
    if set(record) != expected_keys:
        raise ValueError("material-identity record keys differ")
    if record.get("schema") != SCHEMA:
        raise ValueError("unsupported material-identity schema")
    source = record.get("source")
    if not isinstance(source, Mapping) or set(source) != {
        "kind",
        "path",
        "sha256",
    }:
        raise ValueError("material source provenance is required")
    source_path = Path(str(source.get("path", ""))).resolve(strict=True)
    source_hash = str(source.get("sha256", "")).lower()
    if (
        source.get("kind") not in SOURCE_KINDS
        or len(source_hash) != 64
        or any(char not in "0123456789abcdef" for char in source_hash)
        or _sha256(source_path) != source_hash
    ):
        raise ValueError("material source provenance hash mismatch")
    verified_source = {
        "kind": source["kind"],
        "path": str(source_path),
        "sha256": source_hash,
    }
    normalized = _material_record(
        material_id=record.get("material_id", ""),
        name=record.get("name", ""),
        density_g_cm3=record.get("density_g_cm3"),
        temperature_K=record.get("temperature_K"),
        composition=record.get("isotopes", {}),
        composition_basis=str(record.get("composition_basis", "")),
        source=verified_source,
        citation=record.get("citation"),
    )
    if record.get("composition_sha256") != normalized["composition_sha256"]:
        raise ValueError("material composition hash mismatch")
    if record.get("record_sha256") != normalized["record_sha256"]:
        raise ValueError("material record hash mismatch")
    return normalized


def read_fusion_material_db(
    path: str | Path,
    *,
    temperature_K: float | None = None,
) -> list[dict[str, Any]]:
    """Read the public fusion-material-db JSON without requiring PyNE."""
    source = Path(path).resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("fusion-material-db JSON is empty")
    source_binding = {
        "kind": "fusion-material-db-json",
        "path": str(source),
        "sha256": _sha256(source),
    }
    outputs = []
    for key, value in sorted(payload.items()):
        if not isinstance(value, Mapping):
            raise ValueError(f"material {key!r} is malformed")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"material {key!r} metadata is malformed")
        outputs.append(
            _material_record(
                material_id=str(key),
                name=str(metadata.get("name", key)),
                density_g_cm3=value.get("density"),
                temperature_K=temperature_K,
                composition=value.get("comp", {}),
                composition_basis="mass_fraction",
                citation=metadata.get("citation"),
                source=source_binding,
            )
        )
    return outputs


def read_openmc_materials_xml(path: str | Path) -> list[dict[str, Any]]:
    """Read OpenMC material identities while preserving ``ao`` versus ``wo``."""
    source = Path(path).resolve(strict=True)
    root = ET.parse(source).getroot()
    source_binding = {
        "kind": "openmc-materials-xml",
        "path": str(source),
        "sha256": _sha256(source),
    }
    outputs = []
    for element in root.findall("material"):
        density_node = element.find("density")
        if density_node is None or density_node.get("units") != "g/cm3":
            raise ValueError("OpenMC material density must use g/cm3")
        nuclides = element.findall("nuclide")
        bases = {
            "ao" if item.get("ao") is not None else "wo" for item in nuclides
        }
        if len(bases) != 1:
            raise ValueError("OpenMC material mixes atom and mass fractions")
        attribute = next(iter(bases), "")
        basis = "atom_fraction" if attribute == "ao" else "mass_fraction"
        composition = {
            str(item.get("name")): float(item.get(attribute, "nan"))
            for item in nuclides
        }
        outputs.append(
            _material_record(
                material_id=str(element.get("id", "")),
                name=str(element.get("name", element.get("id", ""))),
                density_g_cm3=float(density_node.get("value", "nan")),
                temperature_K=(
                    None
                    if element.get("temperature") is None
                    else float(element.get("temperature", "nan"))
                ),
                composition=composition,
                composition_basis=basis,
                source=source_binding,
            )
        )
    if not outputs:
        raise ValueError("OpenMC materials XML has no materials")
    return outputs


def bind_material_domains(
    materials: list[Mapping[str, Any]],
    domain_bindings: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join material identities to audited DAGMC/OpenMC volume identities."""
    by_id = {str(item.get("material_id")): item for item in materials}
    outputs = []
    for domain_id, binding in sorted(domain_bindings.items()):
        material_id = str(binding.get("material_id", ""))
        if material_id not in by_id:
            raise ValueError(f"domain {domain_id!r} uses an unknown material")
        volume = float(binding.get("volume_cm3", 0.0))
        if not math.isfinite(volume) or volume <= 0.0:
            raise ValueError(f"domain {domain_id!r} volume is invalid")
        row = dict(by_id[material_id])
        row.update(
            {
                "domain_id": str(domain_id),
                "volume_cm3": volume,
                "openmc_cell_id": int(binding.get("openmc_cell_id", 0)),
                "dagmc_volume_id": int(binding.get("dagmc_volume_id", 0)),
                "volume_receipt_sha256": str(
                    binding.get("volume_receipt_sha256", "")
                ),
                "raw_h5m_sha256": str(binding.get("raw_h5m_sha256", "")),
                "canonical_geometry_fingerprint": str(
                    binding.get("canonical_geometry_fingerprint", "")
                ),
            }
        )
        if min(row["openmc_cell_id"], row["dagmc_volume_id"]) <= 0:
            raise ValueError(f"domain {domain_id!r} IDs are invalid")
        outputs.append(row)
    return outputs
