"""Packaged JSON Schemas for neutral magnet-radiation products."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


SCHEMA_FILES = (
    "magnet_boundary_source.schema.json",
    "magnet_damage_gas.schema.json",
    "magnet_radiation_field_bundle.schema.json",
    "magnet_scalar_flux_fields.schema.json",
    "magnet_weight_window_qualification.schema.json",
)


def schema_path(name: str) -> Path:
    """Return one installed schema path and reject unknown names."""
    if name not in SCHEMA_FILES:
        raise KeyError(name)
    return Path(str(files(__package__).joinpath(name)))
