"""Deterministic records shared by ParaStell geometry exporters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ComponentRecord:
    """One named physical solid in an export-ready ParaStell model."""

    name: str
    kind: str
    material_tag: str
    solid: Any
    expected_volume: float
    source_port_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable portion of the record."""
        data = asdict(self)
        data.pop("solid")
        return data


COMPONENT_KIND_ORDER = {
    "plasma_or_chamber": 0,
    "blanket_layer": 1,
    "port_void": 2,
    "port_liner": 3,
    "magnet_casing": 4,
    "magnet_conductor": 5,
    "graveyard": 6,
}


def component_sort_key(record: ComponentRecord) -> tuple[int, str]:
    """Sort component records independently of object construction order."""
    return (COMPONENT_KIND_ORDER.get(record.kind, 99), record.name)
