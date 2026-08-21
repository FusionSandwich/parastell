"""Audit transport libraries against an OpenMC depletion chain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable
import xml.etree.ElementTree as ET

from .model import base_manifest, sha256_file, write_json


def _canonical_nuclide(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", name).lower()


@dataclass(frozen=True)
class ActivationChainAudit:
    """Machine-readable chain and transport compatibility result."""

    chain_path: str
    chain_sha256: str
    cross_sections_path: str
    cross_sections_sha256: str
    requested_nuclides: tuple[str, ...]
    transport_nuclides: tuple[str, ...]
    chain_nuclides: tuple[str, ...]
    missing_transport: tuple[str, ...]
    missing_chain: tuple[str, ...]
    reachable_chain_nuclides: tuple[str, ...]
    missing_chain_targets: tuple[str, ...]
    missing_reactions: tuple[str, ...]
    missing_decay_data: tuple[str, ...]
    release_mismatches: tuple[str, ...]
    photon_source_nuclides: tuple[str, ...]

    @property
    def passes(self) -> bool:
        return not any(
            (
                self.missing_transport,
                self.missing_chain,
                self.missing_chain_targets,
                self.missing_reactions,
                self.missing_decay_data,
                self.release_mismatches,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            **base_manifest(),
            "kind": "activation_chain_audit",
            **self.__dict__,
            "passes": self.passes,
        }

    def write(self, path: str | Path) -> Path:
        return write_json(path, self.as_dict())


def _transport_materials(root: ET.Element) -> set[str]:
    values: set[str] = set()
    for library in root.iter("library"):
        for material in library.attrib.get("materials", "").split():
            values.add(_canonical_nuclide(material))
    return values


def audit_activation_chain(
    chain_path: str | Path,
    cross_sections_path: str | Path,
    requested_nuclides: Iterable[str],
    *,
    chain_release: str | None = None,
    transport_release: str | None = None,
    allow_release_mismatch: bool = False,
) -> ActivationChainAudit:
    """Audit required nuclides and reject unapproved release mixing."""

    chain_file = Path(chain_path).resolve()
    xs_file = Path(cross_sections_path).resolve()
    if not chain_file.is_file():
        raise FileNotFoundError(f"depletion chain not found: {chain_file}")
    if not xs_file.is_file():
        raise FileNotFoundError(f"cross sections index not found: {xs_file}")

    chain_root = ET.parse(chain_file).getroot()
    xs_root = ET.parse(xs_file).getroot()
    chain_entries = {
        _canonical_nuclide(node.attrib["name"]): node
        for node in chain_root.iter("nuclide")
        if "name" in node.attrib
    }
    transport = _transport_materials(xs_root)
    requested = tuple(
        dict.fromkeys(str(value) for value in requested_nuclides)
    )

    missing_transport: list[str] = []
    missing_chain: list[str] = []
    missing_reactions: list[str] = []
    missing_decay: list[str] = []
    for name in requested:
        key = _canonical_nuclide(name)
        if key not in transport:
            missing_transport.append(name)
        node = chain_entries.get(key)
        if node is None:
            missing_chain.append(name)
            continue
        if not list(node.iter("reaction")):
            missing_reactions.append(name)
        half_life = node.attrib.get("half_life")
        stable = half_life is None and not list(node.iter("decay"))
        if not stable and not list(node.iter("decay")):
            missing_decay.append(name)

    reachable: dict[str, str] = {}
    missing_targets: list[str] = []
    queue = list(requested)
    while queue:
        name = queue.pop(0)
        key = _canonical_nuclide(name)
        if key in reachable:
            continue
        node = chain_entries.get(key)
        if node is None:
            continue
        canonical_name = node.attrib["name"]
        reachable[key] = canonical_name
        for child in node:
            if child.tag not in {"reaction", "decay"}:
                continue
            target = child.attrib.get("target")
            if not target or target.lower() in {"nothing", "fission"}:
                continue
            target_key = _canonical_nuclide(target)
            if target_key not in chain_entries:
                missing_targets.append(f"{canonical_name}->{target}")
            elif target_key not in reachable:
                queue.append(chain_entries[target_key].attrib["name"])
    photon_sources = [
        name
        for key, name in reachable.items()
        if any(
            source.attrib.get("particle", "").lower() == "photon"
            for source in chain_entries[key].iter("source")
        )
    ]

    mismatches: list[str] = []
    if (
        chain_release
        and transport_release
        and chain_release.strip().lower() != transport_release.strip().lower()
        and not allow_release_mismatch
    ):
        mismatches.append(
            f"chain={chain_release}; transport={transport_release}"
        )

    return ActivationChainAudit(
        chain_path=str(chain_file),
        chain_sha256=sha256_file(chain_file),
        cross_sections_path=str(xs_file),
        cross_sections_sha256=sha256_file(xs_file),
        requested_nuclides=requested,
        transport_nuclides=tuple(sorted(transport)),
        chain_nuclides=tuple(sorted(chain_entries)),
        missing_transport=tuple(missing_transport),
        missing_chain=tuple(missing_chain),
        reachable_chain_nuclides=tuple(sorted(reachable.values())),
        missing_chain_targets=tuple(dict.fromkeys(missing_targets)),
        missing_reactions=tuple(missing_reactions),
        missing_decay_data=tuple(missing_decay),
        release_mismatches=tuple(mismatches),
        photon_source_nuclides=tuple(photon_sources),
    )
