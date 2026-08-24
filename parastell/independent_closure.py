"""Independent-history closure for magnet boundary-source handoffs."""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping

import h5py


def _manifest(path: str | Path) -> dict[str, Any]:
    with h5py.File(Path(path), "r") as handle:
        if "manifest_json" not in handle:
            raise ValueError(f"handoff has no manifest_json: {path}")
        value = handle["manifest_json"][()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    manifest = json.loads(value)
    if manifest.get("schema") != "parastell.magnet_boundary_source/v2.1.0":
        raise ValueError("independent closure requires boundary schema v2.1.0")
    closure = manifest.get("bank_metadata", {}).get(
        "same_run_integrity_closure"
    )
    if not isinstance(closure, Mapping):
        raise ValueError(
            "handoff lacks the source tally/bank integrity inventory"
        )
    return manifest


def _identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    envelope = manifest.get("envelope", {})
    provenance = manifest.get("provenance", {})
    return {
        "geometry_sha256": envelope.get("dagmc_geometry_sha256"),
        "source_definition_sha256": provenance.get("source_definition_sha256"),
        "envelope_id": envelope.get("envelope_id"),
        "magnet_id": envelope.get("magnet_component"),
    }


def _entries(
    manifest: Mapping[str, Any],
) -> dict[tuple[str, str, int | None], Any]:
    closure = manifest["bank_metadata"]["same_run_integrity_closure"]
    result = {}
    for particle, particle_closure in closure.items():
        whole = particle_closure["whole_envelope"]
        for sense in ("incoming", "outgoing", "net", "total_crossing"):
            result[(particle, sense, None)] = whole[sense]
        for surface in particle_closure["by_surface"]:
            surface_id = int(surface["surface_id"])
            for sense in ("incoming", "outgoing", "net", "total_crossing"):
                result[(particle, sense, surface_id)] = surface[sense]
    return result


def compare_independent_handoffs(
    reference_path: str | Path,
    replicate_path: str | Path,
    *,
    reference_seed: int,
    replicate_seed: int,
    z_tolerance: float = 3.0,
    familywise_alpha: float | None = 0.05,
    numerical_absolute_tolerance: float = 1.0e-14,
) -> dict[str, Any]:
    """Compare banks and tallies accumulated from distinct random histories.

    Both cross-directions are tested: reference bank against replicate tally,
    and replicate bank against reference tally. No bank is renormalized.
    """

    if int(reference_seed) == int(replicate_seed):
        raise ValueError("independent closure requires distinct random seeds")
    if z_tolerance <= 0.0 or numerical_absolute_tolerance < 0.0:
        raise ValueError("closure tolerances must be positive")
    if familywise_alpha is not None and not 0.0 < familywise_alpha < 1.0:
        raise ValueError("familywise_alpha must be between zero and one")
    reference = _manifest(reference_path)
    replicate = _manifest(replicate_path)
    reference_identity = _identity(reference)
    replicate_identity = _identity(replicate)
    if reference_identity != replicate_identity:
        raise ValueError(
            "independent handoffs do not share geometry, source, envelope, and magnet identity"
        )
    reference_entries = _entries(reference)
    replicate_entries = _entries(replicate)
    if set(reference_entries) != set(replicate_entries):
        raise ValueError(
            "independent handoffs have incompatible closure inventories"
        )

    comparisons = []
    for direction, bank_entries, tally_entries in (
        (
            "reference_bank_vs_replicate_tally",
            reference_entries,
            replicate_entries,
        ),
        (
            "replicate_bank_vs_reference_tally",
            replicate_entries,
            reference_entries,
        ),
    ):
        for (particle, sense, surface_id), bank_entry in sorted(
            bank_entries.items(),
            key=lambda item: (item[0][0], item[0][2] or -1, item[0][1]),
        ):
            tally_entry = tally_entries[(particle, sense, surface_id)]
            bank_mean = float(bank_entry["bank_current_per_source"])
            tally_mean = float(tally_entry["tally_current_per_source"])
            bank_counting_sigma = float(bank_entry["bank_poisson_std_dev"])
            bank_sigma = float(bank_entry["tally_std_dev"])
            tally_sigma = float(tally_entry["tally_std_dev"])
            combined_sigma = sqrt(bank_sigma**2 + tally_sigma**2)
            difference = bank_mean - tally_mean
            z_score = (
                difference / combined_sigma if combined_sigma > 0.0 else 0.0
            )
            tolerance = (
                z_tolerance * combined_sigma + numerical_absolute_tolerance
            )
            comparisons.append(
                {
                    "comparison": direction,
                    "particle": particle,
                    "sense": sense,
                    "surface_id": surface_id,
                    "bank_current_per_source": bank_mean,
                    "independent_tally_current_per_source": tally_mean,
                    "bank_std_dev": bank_sigma,
                    "bank_counting_std_dev_diagnostic": bank_counting_sigma,
                    "independent_tally_std_dev": tally_sigma,
                    "combined_std_dev": combined_sigma,
                    "difference": difference,
                    "z_score": z_score,
                    "acceptance_tolerance": tolerance,
                    "passes": abs(difference) <= tolerance,
                }
            )
    raw_failure_count = sum(not item["passes"] for item in comparisons)
    effective_z_tolerance = float(z_tolerance)
    familywise_method = "none"
    if familywise_alpha is not None and comparisons:
        bonferroni_z = NormalDist().inv_cdf(
            1.0 - familywise_alpha / (2.0 * len(comparisons))
        )
        effective_z_tolerance = max(effective_z_tolerance, bonferroni_z)
        familywise_method = "two-sided Bonferroni"
    for item in comparisons:
        item["acceptance_tolerance"] = (
            effective_z_tolerance * item["combined_std_dev"]
            + numerical_absolute_tolerance
        )
        item["passes"] = (
            abs(item["difference"]) <= item["acceptance_tolerance"]
        )
    failures = [item for item in comparisons if not item["passes"]]
    return {
        "schema": "parastell.magnet_independent_closure/v1.0.0",
        "identity": reference_identity,
        "reference": {
            "path": str(Path(reference_path)),
            "seed": int(reference_seed),
        },
        "replicate": {
            "path": str(Path(replicate_path)),
            "seed": int(replicate_seed),
        },
        "statistical_contract": "distinct OpenMC random seeds; bank and tally values are accumulated from independent histories",
        "z_tolerance": effective_z_tolerance,
        "raw_z_tolerance": float(z_tolerance),
        "familywise_alpha": familywise_alpha,
        "familywise_method": familywise_method,
        "raw_failure_count": raw_failure_count,
        "numerical_absolute_tolerance": float(numerical_absolute_tolerance),
        "comparison_count": len(comparisons),
        "failure_count": len(failures),
        "maximum_absolute_z_score": max(
            (abs(item["z_score"]) for item in comparisons), default=0.0
        ),
        "passes": not failures,
        "comparisons": comparisons,
    }
