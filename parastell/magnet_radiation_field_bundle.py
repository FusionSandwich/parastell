"""Consumer-neutral magnet radiation-field producer bundle."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence


SCHEMA = "parastell.magnet_radiation_field_bundle/v1.0.0"
PRODUCT_KINDS = {"boundary_phase_space", "volume_scalar_flux", "heating"}
REQUIRED_GEOMETRY = {"raw_h5m_sha256", "canonical_geometry_fingerprint"}
REQUIRED_SOURCE = {"physical_source_rate_per_s", "source_definition_sha256"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_fields(
    value: Mapping[str, Any], required: set[str], label: str
) -> None:
    missing = required - set(value)
    if missing:
        raise ValueError(f"{label} is missing {sorted(missing)}")


def _validate_product(product: Mapping[str, Any]) -> None:
    _require_fields(
        product,
        {"kind", "magnet_id", "path", "units", "normalization"},
        "radiation product",
    )
    if product["kind"] not in PRODUCT_KINDS:
        raise ValueError(f"unknown radiation product kind {product['kind']!r}")
    if product["kind"] == "volume_scalar_flux":
        if product["units"] not in {"1/cm2/source", "particles/cm2/s"}:
            raise ValueError("volume scalar flux has an unknown unit")
        if product.get("quantity") != "volume_scalar_flux":
            raise ValueError(
                "surface current cannot be supplied as volume scalar flux"
            )
    if product["kind"] == "heating":
        if product["units"] not in {"W", "W/cm3"}:
            raise ValueError("heating has an unknown unit")
        if product.get("quantity") != "heating":
            raise ValueError(
                "a non-heating quantity cannot be supplied as heating"
            )


def write_radiation_field_bundle(
    output_directory: str | Path,
    *,
    provenance: Mapping[str, Any],
    geometry: Mapping[str, Any],
    source: Mapping[str, Any],
    nuclear_data: Mapping[str, Any],
    materials: Mapping[str, Any],
    magnet_inventory: Sequence[Mapping[str, Any]],
    products: Sequence[Mapping[str, Any]],
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy neutral products into a hash-bound, dependency-free bundle."""
    _require_fields(geometry, REQUIRED_GEOMETRY, "geometry manifest")
    _require_fields(source, REQUIRED_SOURCE, "source manifest")
    rate = float(source["physical_source_rate_per_s"])
    if rate <= 0.0:
        raise ValueError("physical_source_rate_per_s must be positive")
    if not magnet_inventory:
        raise ValueError("magnet inventory cannot be empty")
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    product_entries = []
    for item in products:
        product = dict(item)
        _validate_product(product)
        source_path = Path(product.pop("path")).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        role_directory = {
            "boundary_phase_space": "boundary",
            "volume_scalar_flux": "volume_flux",
            "heating": "heating",
        }[product["kind"]]
        relative = (
            Path(role_directory) / str(product["magnet_id"]) / source_path.name
        )
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
        product.update(
            {
                "path": relative.as_posix(),
                "sha256": _sha256(target),
                "size_bytes": target.stat().st_size,
            }
        )
        product_entries.append(product)
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": dict(provenance),
        "geometry": dict(geometry),
        "source": dict(source),
        "nuclear_data": dict(nuclear_data),
        "materials": dict(materials),
        "magnet_inventory": [dict(item) for item in magnet_inventory],
        "products": product_entries,
        "verification": dict(verification),
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**manifest, "bundle_manifest_sha256": _sha256(manifest_path)}


def read_radiation_field_bundle(path: str | Path) -> dict[str, Any]:
    """Validate a bundle without importing OpenMC, DAGMC, MOAB, or CadQuery."""
    root = Path(path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("incompatible magnet radiation-field bundle schema")
    _require_fields(
        manifest["geometry"], REQUIRED_GEOMETRY, "geometry manifest"
    )
    _require_fields(manifest["source"], REQUIRED_SOURCE, "source manifest")
    for product in manifest.get("products", []):
        _validate_product(product)
        product_path = root / product["path"]
        if (
            not product_path.is_file()
            or _sha256(product_path) != product["sha256"]
        ):
            raise ValueError(
                f"radiation product hash mismatch: {product['path']}"
            )
    return manifest
