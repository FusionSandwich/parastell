"""Evaluated-data multigroup coefficients for the HTS reference solver."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Callable, Mapping, Sequence

import h5py
import numpy as np


SCHEMA = "parastell.evaluated_multigroup_layers/v1.0.0"
NEUTRON_TOTAL_MT = 1
NEUTRON_ABSORPTION_MT = 101
PHOTON_SCATTER_MTS = (502, 504)
PHOTON_REMOVAL_MTS = (515, 517, 522)


@dataclass(frozen=True)
class MaterialSpec:
    name: str
    density_g_cm3: float
    element_weight_fractions: Mapping[str, float]
    provenance: str

    def __post_init__(self) -> None:
        fractions = np.asarray(list(self.element_weight_fractions.values()))
        if self.density_g_cm3 <= 0.0 or not np.isfinite(self.density_g_cm3):
            raise ValueError("material density must be finite and positive")
        if (
            not self.element_weight_fractions
            or np.any(~np.isfinite(fractions))
            or np.any(fractions <= 0.0)
            or not np.isclose(fractions.sum(), 1.0, rtol=0.0, atol=1.0e-12)
        ):
            raise ValueError(
                "element weight fractions must be positive and sum to one"
            )


def collapse_lethargy_weighted(
    edges_eV: Sequence[float],
    evaluator: Callable[[np.ndarray], np.ndarray],
    *,
    minimum_eV: float,
    maximum_eV: float,
    points_per_group: int = 48,
) -> np.ndarray:
    """Collapse a continuous cross section with a flat-lethargy spectrum."""

    edges = np.asarray(edges_eV, dtype=float)
    if (
        edges.ndim != 1
        or len(edges) < 2
        or np.any(~np.isfinite(edges))
        or np.any(np.diff(edges) <= 0.0)
        or minimum_eV <= 0.0
        or maximum_eV <= minimum_eV
        or points_per_group < 4
    ):
        raise ValueError("invalid multigroup collapse controls")
    result = np.zeros(len(edges) - 1)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        lower = max(float(lower), minimum_eV)
        upper = min(float(upper), maximum_eV)
        if upper <= lower:
            continue
        energy = np.geomspace(lower, upper, points_per_group)
        values = np.asarray(evaluator(energy), dtype=float)
        if values.shape != energy.shape or np.any(~np.isfinite(values)):
            raise ValueError("cross-section evaluator returned invalid values")
        integrator = getattr(np, "trapezoid", None)
        if integrator is None:
            integrator = np.trapz
        result[index] = integrator(values / energy, energy) / np.log(
            upper / lower
        )
    return result


def _formula_weight_fractions(
    stoichiometry: Mapping[str, float],
) -> dict[str, float]:
    import openmc.data

    masses = {
        element: float(count) * _element_atomic_mass(element)
        for element, count in stoichiometry.items()
    }
    total = sum(masses.values())
    return {element: mass / total for element, mass in masses.items()}


def _element_atomic_mass(element: str) -> float:
    import openmc.data

    isotopes = {
        isotope: abundance
        for isotope, abundance in openmc.data.NATURAL_ABUNDANCE.items()
        if re.match(r"[A-Z][a-z]?", isotope).group() == element
    }
    if not isotopes:
        raise ValueError(f"natural abundance is unavailable for {element}")
    return sum(
        openmc.data.atomic_mass(isotope) * abundance
        for isotope, abundance in isotopes.items()
    ) / sum(isotopes.values())


def representative_hts_materials() -> tuple[MaterialSpec, ...]:
    """Return documented reference compositions for the explicit tape stack."""

    return (
        MaterialSpec("Cu", 8.96, {"Cu": 1.0}, "elemental copper"),
        MaterialSpec("Ag", 10.49, {"Ag": 1.0}, "elemental silver"),
        MaterialSpec(
            "REBCO",
            6.30,
            _formula_weight_fractions({"Y": 1, "Ba": 2, "Cu": 3, "O": 7}),
            "YBa2Cu3O7 reference stoichiometry",
        ),
        MaterialSpec(
            "buffer",
            6.00,
            _formula_weight_fractions({"Zr": 0.92, "Y": 0.16, "O": 2.08}),
            "8 mol% Y2O3-stabilized ZrO2 representative buffer",
        ),
        MaterialSpec(
            "Hastelloy",
            8.89,
            {
                "Ni": 0.57,
                "Cr": 0.16,
                "Mo": 0.16,
                "Fe": 0.05,
                "W": 0.04,
                "Co": 0.02,
            },
            "representative Hastelloy C-276 major-element mass fractions",
        ),
        MaterialSpec(
            "solder",
            7.40,
            {"Sn": 0.965, "Ag": 0.03, "Cu": 0.005},
            "SAC305 representative solder",
        ),
        MaterialSpec(
            "polyimide",
            1.42,
            _formula_weight_fractions({"C": 22, "H": 10, "N": 2, "O": 5}),
            "C22H10N2O5 representative polyimide repeat unit",
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _records(library, type_: str) -> dict[str, Path]:
    result = {}
    for entry in library.libraries:
        if entry["type"] == type_:
            for material in entry["materials"]:
                result[str(material)] = Path(entry["path"])
    return result


def _element_targets(
    element: str, available: Mapping[str, Path]
) -> dict[str, float]:
    import openmc.data

    natural = f"{element}0"
    if natural in available:
        return {natural: 1.0}
    abundances = {
        isotope: abundance
        for isotope, abundance in openmc.data.NATURAL_ABUNDANCE.items()
        if re.match(r"[A-Z][a-z]?", isotope).group() == element
        and isotope in available
    }
    total = sum(abundances.values())
    if total <= 0.0:
        raise ValueError(
            f"neutron library has no natural target for {element}"
        )
    return {
        isotope: abundance / total for isotope, abundance in abundances.items()
    }


def _atom_densities(
    material: MaterialSpec, neutron_records: Mapping[str, Path]
) -> tuple[dict[str, float], dict[str, float]]:
    import openmc.data

    neutron = {}
    elements = {}
    avogadro = 6.02214076e23
    for element, mass_fraction in material.element_weight_fractions.items():
        atom_density = (
            material.density_g_cm3
            * mass_fraction
            * avogadro
            / _element_atomic_mass(element)
            * 1.0e-24
        )
        elements[element] = atom_density
        for target, fraction in _element_targets(
            element, neutron_records
        ).items():
            neutron[target] = atom_density * fraction
    return neutron, elements


def _sum_neutron_reactions(
    data, mts: Sequence[int], energy: np.ndarray
) -> np.ndarray:
    values = np.zeros_like(energy)
    for mt in mts:
        if mt not in data.reactions:
            continue
        function = data.reactions[mt].xs[data.temperatures[0]]
        mask = (energy >= function.x[0]) & (energy <= function.x[-1])
        values[mask] += function(energy[mask])
    return values


def _sum_photon_reactions(
    data, mts: Sequence[int], energy: np.ndarray
) -> np.ndarray:
    values = np.zeros_like(energy)
    for mt in mts:
        if mt not in data.reactions:
            continue
        function = data.reactions[mt].xs
        mask = (energy >= function.x[0]) & (energy <= function.x[-1])
        values[mask] += function(energy[mask])
    return values


def build_evaluated_multigroup_file(
    cross_sections_xml: str | Path,
    output_path: str | Path,
    *,
    neutron_edges_eV: Sequence[float],
    photon_edges_eV: Sequence[float],
    materials: Sequence[MaterialSpec] | None = None,
) -> Path:
    """Build evaluated diagonal-scatter coefficients for reference replay."""

    import openmc.data

    xml = Path(cross_sections_xml).resolve()
    library = openmc.data.DataLibrary.from_xml(xml)
    neutron_records = _records(library, "neutron")
    photon_records = _records(library, "photon")
    neutron_edges = np.asarray(neutron_edges_eV, dtype=float)
    photon_edges = np.asarray(photon_edges_eV, dtype=float)
    material_specs = tuple(materials or representative_hts_materials())
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as stream:
        stream.attrs["schema"] = SCHEMA
        manifest = {
            "schema": SCHEMA,
            "cross_sections_xml": str(xml),
            "cross_sections_xml_sha256": _sha256(xml),
            "neutron_collapse": "flat lethargy within each deterministic group",
            "photon_collapse": "flat lethargy within each deterministic group",
            "scattering_contract": "evaluated non-absorption represented as within-group diagonal scattering",
            "missing_physics": [
                "energy-angle scattering transfer matrices",
                "neutron-to-photon production matrices",
                "charged-particle transport",
            ],
        }
        stream.create_dataset("manifest_json", data=json.dumps(manifest))
        stream.create_dataset("neutron_energy_edges_eV", data=neutron_edges)
        stream.create_dataset("photon_energy_edges_eV", data=photon_edges)
        root = stream.create_group("materials")
        for spec in material_specs:
            neutron_atoms, element_atoms = _atom_densities(
                spec, neutron_records
            )
            neutron_total = np.zeros(len(neutron_edges) - 1)
            neutron_absorption = np.zeros_like(neutron_total)
            for target, atom_density in neutron_atoms.items():
                data = openmc.data.IncidentNeutron.from_hdf5(
                    neutron_records[target]
                )
                total_mts = data.get_reaction_components(NEUTRON_TOTAL_MT)
                absorption_mts = data.get_reaction_components(
                    NEUTRON_ABSORPTION_MT
                )
                neutron_total += atom_density * collapse_lethargy_weighted(
                    neutron_edges,
                    lambda energy, d=data, mts=total_mts: _sum_neutron_reactions(
                        d, mts, energy
                    ),
                    minimum_eV=1.0e-5,
                    maximum_eV=20.0e6,
                )
                neutron_absorption += atom_density * collapse_lethargy_weighted(
                    neutron_edges,
                    lambda energy, d=data, mts=absorption_mts: _sum_neutron_reactions(
                        d, mts, energy
                    ),
                    minimum_eV=1.0e-5,
                    maximum_eV=20.0e6,
                )
            photon_scatter = np.zeros(len(photon_edges) - 1)
            photon_absorption = np.zeros_like(photon_scatter)
            for element, atom_density in element_atoms.items():
                if element not in photon_records:
                    raise ValueError(
                        f"photon library has no data for {element}"
                    )
                data = openmc.data.IncidentPhoton.from_hdf5(
                    photon_records[element]
                )
                photon_scatter += atom_density * collapse_lethargy_weighted(
                    photon_edges,
                    lambda energy, d=data: _sum_photon_reactions(
                        d, PHOTON_SCATTER_MTS, energy
                    ),
                    minimum_eV=1.0e3,
                    maximum_eV=30.0e6,
                )
                photon_absorption += atom_density * collapse_lethargy_weighted(
                    photon_edges,
                    lambda energy, d=data: _sum_photon_reactions(
                        d, PHOTON_REMOVAL_MTS, energy
                    ),
                    minimum_eV=1.0e3,
                    maximum_eV=30.0e6,
                )
            neutron_scatter = np.maximum(
                neutron_total - neutron_absorption, 0.0
            )
            photon_total = photon_scatter + photon_absorption
            group = root.create_group(spec.name)
            group.attrs["density_g_cm3"] = spec.density_g_cm3
            group.attrs["provenance"] = spec.provenance
            group.attrs["element_weight_fractions_json"] = json.dumps(
                dict(spec.element_weight_fractions), sort_keys=True
            )
            group.create_dataset("neutron_total_cm_1", data=neutron_total)
            group.create_dataset(
                "neutron_absorption_cm_1", data=neutron_absorption
            )
            group.create_dataset(
                "neutron_scattering_cm_1", data=np.diag(neutron_scatter)
            )
            group.create_dataset("photon_total_cm_1", data=photon_total)
            group.create_dataset(
                "photon_absorption_cm_1", data=photon_absorption
            )
            group.create_dataset(
                "photon_scattering_cm_1", data=np.diag(photon_scatter)
            )
    return output


def load_layer_transport(
    path: str | Path,
    layers: Sequence[tuple[str, str, float]],
):
    """Load evaluated coefficients into ``LayerTransport`` objects."""

    from .multigroup_sn import LayerTransport

    result = []
    with h5py.File(path, "r") as stream:
        if stream.attrs.get("schema") != SCHEMA:
            raise ValueError("unsupported evaluated multigroup schema")
        for name, material, thickness_cm in layers:
            if material not in stream["materials"]:
                raise ValueError(f"evaluated material {material!r} is absent")
            group = stream[f"materials/{material}"]
            result.append(
                LayerTransport(
                    name=name,
                    material=material,
                    thickness_cm=thickness_cm,
                    total_xs_cm_1={
                        "neutron": group["neutron_total_cm_1"][...],
                        "photon": group["photon_total_cm_1"][...],
                    },
                    absorption_xs_cm_1={
                        "neutron": group["neutron_absorption_cm_1"][...],
                        "photon": group["photon_absorption_cm_1"][...],
                    },
                    scattering_xs_cm_1={
                        "neutron": group["neutron_scattering_cm_1"][...],
                        "photon": group["photon_scattering_cm_1"][...],
                    },
                )
            )
    return tuple(result)
