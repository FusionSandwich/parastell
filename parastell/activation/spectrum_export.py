"""Conservative activation-spectrum interchange and backend projections."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from parastell.energy_groups import validate_edges

from .model import base_manifest, write_json


@dataclass(frozen=True)
class ActivationSpectrum:
    """Group-integrated scalar flux in ascending-energy order."""

    name: str
    particle: str
    edges_eV: tuple[float, ...]
    group_flux_cm2_s: tuple[float, ...]
    region_id: str
    reference_source_rate_n_s: float
    source: dict[str, Any]

    def __post_init__(self):
        edges = validate_edges(self.edges_eV)
        object.__setattr__(self, "edges_eV", edges)
        if self.particle not in {"neutron", "photon"}:
            raise ValueError("particle must be neutron or photon")
        if len(self.group_flux_cm2_s) != len(edges) - 1:
            raise ValueError("one integrated flux is required per group")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in self.group_flux_cm2_s
        ):
            raise ValueError(
                "group flux values must be finite and nonnegative"
            )
        if (
            not math.isfinite(self.reference_source_rate_n_s)
            or self.reference_source_rate_n_s <= 0.0
        ):
            raise ValueError("reference source rate must be positive")

    @classmethod
    def create(
        cls,
        name: str,
        particle: str,
        edges_eV: Iterable[float],
        group_flux_cm2_s: Iterable[float],
        region_id: str,
        reference_source_rate_n_s: float,
        source: dict[str, Any] | None = None,
    ) -> "ActivationSpectrum":
        return cls(
            name=name,
            particle=particle.lower(),
            edges_eV=tuple(float(value) for value in edges_eV),
            group_flux_cm2_s=tuple(float(value) for value in group_flux_cm2_s),
            region_id=region_id,
            reference_source_rate_n_s=float(reference_source_rate_n_s),
            source=dict(source or {}),
        )

    @property
    def total_flux_cm2_s(self) -> float:
        return math.fsum(self.group_flux_cm2_s)

    @property
    def content_sha256(self) -> str:
        payload = json.dumps(
            {
                "edges_eV": self.edges_eV,
                "group_flux_cm2_s": self.group_flux_cm2_s,
            },
            separators=(",", ":"),
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            **base_manifest(),
            "kind": "activation_spectrum",
            "name": self.name,
            "particle": self.particle,
            "region_id": self.region_id,
            "reference_source_rate_n_s": self.reference_source_rate_n_s,
            "energy_units": "eV",
            "energy_ordering": "ascending",
            "quantity": "group-integrated scalar flux",
            "flux_units": "particles/cm^2/s",
            "edges_eV": list(self.edges_eV),
            "group_flux_cm2_s": list(self.group_flux_cm2_s),
            "total_flux_cm2_s": self.total_flux_cm2_s,
            "content_sha256": self.content_sha256,
            "source": self.source,
        }

    def write_json(self, path: str | Path) -> Path:
        return write_json(path, self.as_dict())

    def write_alara_flux(
        self, path: str | Path, *, ordering: str = "descending"
    ) -> Path:
        """Write one ALARA interval spectrum in explicit library ordering."""

        if self.particle != "neutron":
            raise ValueError("ALARA activation flux must be neutron")
        if ordering not in {"ascending", "descending"}:
            raise ValueError("ordering must be ascending or descending")
        values = self.group_flux_cm2_s
        if ordering == "descending":
            values = tuple(reversed(values))
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "\n".join(f"{value:.17e}" for value in values) + "\n",
            encoding="ascii",
        )
        return target

    def write_fispact_arb_flux(
        self, path: str | Path, *, wall_loading_MW_m2: float = 1.0
    ) -> Path:
        """Write the documented FISPACT-II GRPCONVERT ``arb_flux`` format."""

        if self.particle != "neutron":
            raise ValueError("FISPACT-II activation flux must be neutron")
        if len(self.group_flux_cm2_s) < 3:
            raise ValueError(
                "FISPACT-II GRPCONVERT requires more than 2 groups"
            )
        if wall_loading_MW_m2 <= 0.0:
            raise ValueError("wall loading must be positive")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            " ".join(f"{value:.17e}" for value in reversed(self.edges_eV)),
            " ".join(
                f"{value:.17e}" for value in reversed(self.group_flux_cm2_s)
            ),
            f"{wall_loading_MW_m2:.17e}",
            self.name[:100],
        ]
        target.write_text("\n".join(lines) + "\n", encoding="ascii")
        return target


def load_activation_spectrum(path: str | Path) -> ActivationSpectrum:
    data = json.loads(Path(path).read_text(encoding="ascii"))
    if data.get("kind") != "activation_spectrum":
        raise ValueError("not a ParaStell activation spectrum")
    spectrum = ActivationSpectrum.create(
        data["name"],
        data["particle"],
        data["edges_eV"],
        data["group_flux_cm2_s"],
        data["region_id"],
        data["reference_source_rate_n_s"],
        data.get("source"),
    )
    if spectrum.content_sha256 != data.get("content_sha256"):
        raise ValueError("activation spectrum checksum mismatch")
    if not math.isclose(
        spectrum.total_flux_cm2_s,
        float(data["total_flux_cm2_s"]),
        rel_tol=1.0e-13,
        abs_tol=0.0,
    ):
        raise ValueError("activation spectrum normalization mismatch")
    return spectrum
