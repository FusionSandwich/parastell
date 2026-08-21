"""Load immutable, provenance-bearing energy structures.

Continuous particle energy in the magnet boundary bank remains authoritative.
Entries in this registry are purpose-specific projections of that bank.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import resources
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable


_UNIT_SCALE = {"ev": 1.0, "kev": 1.0e3, "mev": 1.0e6, "gev": 1.0e9}


def _canonical_edge_hash(edges: Iterable[float]) -> str:
    payload = json.dumps(
        [float(value) for value in edges], separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def validate_edges(edges: Iterable[float]) -> tuple[float, ...]:
    """Return finite, strictly ascending boundaries or raise ``ValueError``."""

    values = tuple(float(value) for value in edges)
    if len(values) < 2:
        raise ValueError(
            "an energy structure requires at least two boundaries"
        )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("energy boundaries must be finite")
    if any(upper <= lower for lower, upper in zip(values, values[1:])):
        raise ValueError("energy boundaries must be strictly ascending")
    return values


@dataclass(frozen=True)
class EnergyGroupStructure:
    """Validated energy-grid definition and immutable provenance."""

    name: str
    particle: str
    intended_purposes: tuple[str, ...]
    edges_eV: tuple[float, ...] | None
    status: str
    authoritative_source: dict[str, Any]
    aliases: tuple[str, ...] = ()
    edge_sha256: str | None = None
    local_file_sha256: str | None = None

    @property
    def group_count(self) -> int | None:
        return None if self.edges_eV is None else len(self.edges_eV) - 1

    @property
    def is_continuous(self) -> bool:
        return self.edges_eV is None

    def as_dict(self, *, descending: bool = False) -> dict[str, Any]:
        edges = self.edges_eV
        if descending and edges is not None:
            edges = tuple(reversed(edges))
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "particle": self.particle,
            "intended_purposes": list(self.intended_purposes),
            "group_count": self.group_count,
            "edge_units": "eV",
            "minimum_energy": None if self.is_continuous else self.edges_eV[0],
            "maximum_energy": (
                None if self.is_continuous else self.edges_eV[-1]
            ),
            "edge_ordering": "descending" if descending else "ascending",
            "representation": (
                "continuous" if self.is_continuous else "grouped"
            ),
            "status": self.status,
            "edge_sha256": self.edge_sha256,
            "local_file_sha256": self.local_file_sha256,
            "authoritative_source": self.authoritative_source,
            "edges": None if edges is None else list(edges),
        }


def _data_files():
    directory = resources.files(__package__).joinpath("data")
    return sorted(
        (item for item in directory.iterdir() if item.name.endswith(".json")),
        key=lambda item: item.name,
    )


def _load_file(item) -> EnergyGroupStructure:
    raw = item.read_bytes()
    data = json.loads(raw.decode("ascii"))
    representation = data.get("representation", "grouped")
    edges = None
    edge_hash = None
    if representation == "grouped":
        edges = validate_edges(data["edges"])
        if data.get("edge_ordering") != "ascending":
            raise ValueError(f"{item.name}: stored edges must be ascending")
        edge_hash = _canonical_edge_hash(edges)
        if edge_hash != data.get("edge_sha256"):
            raise ValueError(f"{item.name}: edge checksum mismatch")
        if data.get("group_count") != len(edges) - 1:
            raise ValueError(f"{item.name}: group count does not match edges")
        if data.get("minimum_energy") != edges[0]:
            raise ValueError(
                f"{item.name}: minimum energy does not match edges"
            )
        if data.get("maximum_energy") != edges[-1]:
            raise ValueError(
                f"{item.name}: maximum energy does not match edges"
            )
    elif representation != "continuous":
        raise ValueError(f"{item.name}: unsupported representation")
    particle = str(data["particle"]).lower()
    if particle not in {"neutron", "photon"}:
        raise ValueError(f"{item.name}: unsupported particle {particle!r}")
    return EnergyGroupStructure(
        name=str(data["name"]),
        aliases=tuple(str(value) for value in data.get("aliases", ())),
        particle=particle,
        intended_purposes=tuple(data["intended_purposes"]),
        edges_eV=edges,
        status=str(data["status"]),
        edge_sha256=edge_hash,
        local_file_sha256=hashlib.sha256(raw).hexdigest(),
        authoritative_source=dict(data["authoritative_source"]),
    )


def list_structures() -> tuple[EnergyGroupStructure, ...]:
    """Return all built-in structures after full checksum validation."""

    return tuple(_load_file(item) for item in _data_files())


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def get_structure(
    name: str, *, particle: str | None = None
) -> EnergyGroupStructure:
    """Resolve a built-in name or alias, optionally enforcing particle type."""

    requested = _key(name)
    matches = [
        structure
        for structure in list_structures()
        if requested in {_key(structure.name), *map(_key, structure.aliases)}
    ]
    if len(matches) != 1:
        raise KeyError(f"unknown energy structure {name!r}")
    structure = matches[0]
    if particle is not None and structure.particle != particle.lower():
        raise ValueError(
            f"{structure.name} is for {structure.particle}, not {particle}"
        )
    return structure


def load_custom_structure(
    path: str | Path,
    *,
    particle: str,
    units: str = "eV",
    name: str | None = None,
) -> EnergyGroupStructure:
    """Load JSON, CSV, or whitespace boundaries without registering globally."""

    source = Path(path)
    raw = source.read_bytes()
    if source.suffix.lower() == ".json":
        data = json.loads(raw.decode("utf-8"))
        values = data.get("edges", data.get("edges_eV"))
        if values is None:
            raise ValueError("custom JSON requires 'edges' or 'edges_eV'")
        units = data.get("edge_units", units)
        name = name or data.get("name")
    else:
        values = [
            float(value)
            for value in re.findall(
                r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?",
                raw.decode("utf-8"),
            )
        ]
    try:
        scale = _UNIT_SCALE[units.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported energy unit {units!r}") from error
    edges = tuple(float(value) * scale for value in values)
    if edges and edges[0] > edges[-1]:
        edges = tuple(reversed(edges))
    edges = validate_edges(edges)
    particle = particle.lower()
    if particle not in {"neutron", "photon"}:
        raise ValueError("particle must be neutron or photon")
    return EnergyGroupStructure(
        name=name or source.stem,
        particle=particle,
        intended_purposes=("user-defined",),
        edges_eV=edges,
        status="custom-unregistered",
        edge_sha256=_canonical_edge_hash(edges),
        local_file_sha256=hashlib.sha256(raw).hexdigest(),
        authoritative_source={
            "path": str(source.resolve()),
            "license_or_redistribution_note": "User-supplied; not redistributed.",
        },
    )


def compare_structures(first: str, second: str) -> dict[str, Any]:
    """Compare ranges and exact shared boundaries without implying equivalence."""

    left = get_structure(first)
    right = get_structure(second)
    if left.is_continuous or right.is_continuous:
        raise ValueError(
            "continuous representations have no finite boundaries"
        )
    shared = set(left.edges_eV).intersection(right.edges_eV)
    return {
        "first": left.name,
        "second": right.name,
        "particles_match": left.particle == right.particle,
        "first_groups": left.group_count,
        "second_groups": right.group_count,
        "shared_exact_boundaries": len(shared),
        "common_minimum_eV": max(left.edges_eV[0], right.edges_eV[0]),
        "common_maximum_eV": min(left.edges_eV[-1], right.edges_eV[-1]),
        "response_equivalence_assessed": False,
    }
