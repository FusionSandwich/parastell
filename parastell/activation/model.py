"""Shared, solver-neutral activation problem definitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


ACTIVATION_SCHEMA = "parastell.activation/v1.0.0"


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 checksum of a file without loading it at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write deterministic JSON, creating only the requested parent path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return target


@dataclass(frozen=True)
class ActivationStep:
    """One irradiation or cooling interval.

    ``source_rate_n_s`` is the physical neutron source rate. A zero value is a
    cooling interval; no separate, ambiguous cooling convention is used.
    """

    duration_s: float
    source_rate_n_s: float
    label: str

    def __post_init__(self):
        if not math.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("activation step duration must be positive")
        if (
            not math.isfinite(self.source_rate_n_s)
            or self.source_rate_n_s < 0.0
        ):
            raise ValueError("activation source rate must be nonnegative")
        if not self.label.strip():
            raise ValueError("activation step label must not be empty")

    @property
    def kind(self) -> str:
        return "cooling" if self.source_rate_n_s == 0.0 else "irradiation"

    def as_dict(self) -> dict[str, Any]:
        return {
            "duration_s": self.duration_s,
            "source_rate_n_s": self.source_rate_n_s,
            "label": self.label,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class ActivationSchedule:
    """Ordered physical source history shared by all activation backends."""

    name: str
    steps: tuple[ActivationStep, ...]

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("activation schedule name must not be empty")
        if not self.steps:
            raise ValueError("activation schedule requires at least one step")

    @classmethod
    def from_iterable(
        cls, name: str, steps: Iterable[ActivationStep | dict[str, Any]]
    ) -> "ActivationSchedule":
        normalized = tuple(
            (
                step
                if isinstance(step, ActivationStep)
                else ActivationStep(**step)
            )
            for step in steps
        )
        return cls(name=name, steps=normalized)

    @property
    def timesteps_s(self) -> tuple[float, ...]:
        return tuple(step.duration_s for step in self.steps)

    @property
    def source_rates_n_s(self) -> tuple[float, ...]:
        return tuple(step.source_rate_n_s for step in self.steps)

    @property
    def cooling_result_indices(self) -> tuple[int, ...]:
        return tuple(
            index + 1
            for index, step in enumerate(self.steps)
            if step.kind == "cooling"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "steps": [step.as_dict() for step in self.steps],
            "total_duration_s": sum(self.timesteps_s),
            "integrated_source_neutrons": sum(
                step.duration_s * step.source_rate_n_s for step in self.steps
            ),
        }


@dataclass(frozen=True)
class ActivationRegion:
    """Stable mapping from transport space to one activation inventory."""

    region_id: str
    domain_kind: str
    domain_id: int
    material_id: int
    material_name: str
    volume_cm3: float
    density_g_cm3: float
    temperature_K: float
    nuclide_atom_fractions: tuple[tuple[str, float], ...]
    geometry_sha256: str

    def __post_init__(self):
        if self.domain_kind not in {"cell", "mesh_element"}:
            raise ValueError("domain_kind must be cell or mesh_element")
        for name, value in (
            ("volume_cm3", self.volume_cm3),
            ("density_g_cm3", self.density_g_cm3),
            ("temperature_K", self.temperature_K),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if len(self.geometry_sha256) != 64:
            raise ValueError("geometry_sha256 must be a SHA-256 digest")
        if not self.nuclide_atom_fractions:
            raise ValueError("activation region requires nuclides")
        if any(value <= 0.0 for _, value in self.nuclide_atom_fractions):
            raise ValueError("nuclide atom fractions must be positive")
        total = sum(value for _, value in self.nuclide_atom_fractions)
        if not math.isclose(total, 1.0, rel_tol=1.0e-10, abs_tol=1.0e-12):
            raise ValueError("nuclide atom fractions must sum to one")

    def as_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "domain_kind": self.domain_kind,
            "domain_id": self.domain_id,
            "material_id": self.material_id,
            "material_name": self.material_name,
            "volume_cm3": self.volume_cm3,
            "density_g_cm3": self.density_g_cm3,
            "temperature_K": self.temperature_K,
            "nuclide_atom_fractions": dict(self.nuclide_atom_fractions),
            "geometry_sha256": self.geometry_sha256,
        }


def base_manifest() -> dict[str, Any]:
    """Return common activation provenance fields."""

    return {
        "schema": ACTIVATION_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
