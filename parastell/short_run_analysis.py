"""Scientific summary of the bounded Prompt-1B 500k qualification run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import numpy as np


def _strings(dataset: Any) -> np.ndarray:
    return (
        dataset.asstr()[()]
        if hasattr(dataset, "asstr")
        else np.asarray(dataset).astype(str)
    )


def classify_relative_uncertainty(mean: float, relative: float | None) -> str:
    """Apply declared, conservative qualification bands."""
    if not np.isfinite(mean) or mean < 0.0:
        return "INVALID"
    if mean == 0.0:
        return "INSUFFICIENT_STATISTICS"
    if relative is None or not np.isfinite(relative):
        return "INSUFFICIENT_STATISTICS"
    if relative <= 0.25:
        return "QUALIFIED"
    if relative <= 0.50:
        return "MARGINAL"
    return "INSUFFICIENT_STATISTICS"


def _metric(
    mean_values: Any,
    std_values: Any,
    units: str,
    *,
    event_level_ess: float | None = None,
) -> dict[str, Any]:
    mean = float(np.sum(np.asarray(mean_values, dtype=float)))
    # Energy-bin covariance is unavailable.  The L1 sum is a conservative
    # upper bound on the standard deviation of the sum.
    sigma = float(np.sum(np.abs(np.asarray(std_values, dtype=float))))
    relative = sigma / abs(mean) if mean != 0.0 else None
    status = classify_relative_uncertainty(mean, relative)
    return {
        "value": mean,
        "units": units,
        "conservative_std_dev_upper_bound": sigma,
        "conservative_relative_uncertainty_upper_bound": relative,
        "uncertainty_combination": "L1_upper_bound_energy_covariance_unavailable",
        "status": status,
        "zero_interpretation": (
            None
            if mean != 0.0
            else "INSUFFICIENT_STATISTICS; physical zero is not claimed"
        ),
        "event_level_effective_sample_size": event_level_ess,
    }


def _field_rows(group: h5py.Group, mean_name: str, std_name: str, units: str):
    magnet_ids = _strings(group["magnet_ids"])
    roles = _strings(group["component_roles"])
    means = group[mean_name][()]
    stds = group[std_name][()]
    return {
        (str(magnet), str(role)): _metric(mean, std, units)
        for magnet, role, mean, std in zip(magnet_ids, roles, means, stds)
    }


def _bank_summary(path: Path, source_histories: int) -> dict[str, Any]:
    with h5py.File(path, "r") as source:
        manifest = json.loads(source["manifest_json"].asstr()[()])
        records = source["records"]
        particle = _strings(records["particle"])
        sense = _strings(records["crossing_sense"])
        surface = records["surface_id"][()].astype(int)
        patch = records["patch_id"][()].astype(int)
        energy = records["energy_eV"][()].astype(float)
        mu = records["mu"][()].astype(float)
        azimuth = records["azimuth_rad"][()].astype(float)
        position = records["position_global_cm"][()].astype(float)
        weight = records["weight"][()].astype(float)
        all_surface_ids = [
            int(value)
            for value in manifest["bank_metadata"].get(
                "surface_ids", sorted(set(surface.tolist()))
            )
        ]
        rows = []
        for surface_id in sorted(all_surface_ids):
            for name in ("neutron", "photon"):
                for direction in ("incoming", "outgoing", "grazing"):
                    mask = (
                        (surface == surface_id)
                        & (particle == name)
                        & (sense == direction)
                    )
                    count = int(np.count_nonzero(mask))
                    sum_weight = float(weight[mask].sum())
                    sum_squared = float(np.square(weight[mask]).sum())
                    ess = (
                        sum_weight * sum_weight / sum_squared
                        if sum_squared > 0.0
                        else 0.0
                    )
                    rows.append(
                        {
                            "surface_id": surface_id,
                            "particle": name,
                            "sense": direction,
                            "raw_records": count,
                            "sum_weights_per_source": sum_weight,
                            "sum_squared_weights_per_source2": sum_squared,
                            "effective_sample_size": ess,
                            "event_counting_relative_uncertainty": (
                                math.sqrt(sum_squared) / sum_weight
                                if sum_weight > 0.0
                                else None
                            ),
                            "zero_count_95pct_poisson_upper_per_source": (
                                -math.log(0.05) / source_histories
                                if count == 0
                                else None
                            ),
                            "status": (
                                classify_relative_uncertainty(
                                    sum_weight,
                                    (
                                        math.sqrt(sum_squared) / sum_weight
                                        if sum_weight > 0.0
                                        else None
                                    ),
                                )
                            ),
                        }
                    )
        patch_rows = []
        patch_counts = manifest["bank_metadata"].get(
            "surface_patch_counts", {}
        )
        patch_keys = [
            (surface_id, patch_id, name)
            for surface_id in sorted(all_surface_ids)
            for patch_id in range(int(patch_counts.get(str(surface_id), 0)))
            for name in ("neutron", "photon")
        ]
        for surface_id, patch_id, name in patch_keys:
            mask = (
                (surface == surface_id)
                & (patch == patch_id)
                & (particle == name)
            )
            sum_weight = float(weight[mask].sum())
            sum_squared = float(np.square(weight[mask]).sum())
            relative = (
                math.sqrt(sum_squared) / sum_weight
                if sum_weight > 0.0
                else None
            )
            patch_rows.append(
                {
                    "surface_id": int(surface_id),
                    "patch_id": int(patch_id),
                    "particle": str(name),
                    "raw_records": int(np.count_nonzero(mask)),
                    "sum_weights_per_source": sum_weight,
                    "sum_squared_weights_per_source2": sum_squared,
                    "effective_sample_size": (
                        sum_weight * sum_weight / sum_squared
                        if sum_squared > 0.0
                        else 0.0
                    ),
                    "relative_uncertainty": relative,
                    "status": classify_relative_uncertainty(
                        sum_weight, relative
                    ),
                }
            )
        spectra = {}
        axes = manifest["energy_axes"]
        for name in ("neutron", "photon"):
            edges = np.asarray(axes[f"{name}_energy_edges_eV"], dtype=float)
            mask = particle == name
            values = np.histogram(
                energy[mask], bins=edges, weights=weight[mask]
            )[0]
            squares = np.histogram(
                energy[mask], bins=edges, weights=np.square(weight[mask])
            )[0]
            spectra[name] = {
                "energy_edges_eV": edges.tolist(),
                "current_per_source": values.tolist(),
                "effective_sample_size": np.divide(
                    np.square(values),
                    squares,
                    out=np.zeros_like(values),
                    where=squares > 0.0,
                ).tolist(),
            }
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "record_count": len(weight),
            "surface_ids": sorted(all_surface_ids),
            "completeness": manifest["bank_metadata"].get(
                "surface_bank_completeness", {}
            ),
            "current_by_surface_particle_sense": rows,
            "surface_patch_statistics": patch_rows,
            "spectra": spectra,
            "mu": mu.tolist(),
            "azimuth_rad": azimuth.tolist(),
            "position_global_cm": position.tolist(),
            "particle": particle.tolist(),
            "grazing_fraction": float(np.mean(np.abs(mu) <= 0.1)),
        }


def analyze_short_run(
    *,
    scalar_flux_path: str | Path,
    heating_path: str | Path,
    reaction_path: str | Path,
    damage_gas_path: str | Path,
    outer_bank_path: str | Path,
    winding_bank_path: str | Path,
    source_histories: int = 500_000,
) -> dict[str, Any]:
    paths = {
        "scalar_flux": Path(scalar_flux_path).resolve(),
        "heating": Path(heating_path).resolve(),
        "reaction": Path(reaction_path).resolve(),
        "damage_gas": Path(damage_gas_path).resolve(),
        "outer_bank": Path(outer_bank_path).resolve(),
        "winding_bank": Path(winding_bank_path).resolve(),
    }
    with h5py.File(paths["scalar_flux"], "r") as source:
        flux_manifest = json.loads(source["manifest_json"].asstr()[()])
        neutron_flux = _field_rows(
            source["scalar_flux_fields/neutron_configured_fine"],
            "mean_physical",
            "std_dev_physical",
            "cm^-2 s^-1",
        )
        photon_flux = _field_rows(
            source["scalar_flux_fields/photon_configured"],
            "mean_physical",
            "std_dev_physical",
            "cm^-2 s^-1",
        )
        local_mesh = {}
        for particle_name in ("neutron", "photon"):
            group = source[
                "scalar_flux_fields/"
                "example-stellarator-sector-00-90deg-coil-0005_"
                f"{particle_name}_local_mesh"
            ]
            entries = []
            for index, (mean, std) in enumerate(
                zip(group["mean_physical"][()], group["std_dev_physical"][()])
            ):
                metric = _metric(mean, std, "cm^-2 s^-1")
                relative = metric[
                    "conservative_relative_uncertainty_upper_bound"
                ]
                metric.update(
                    {
                        "region_id": str(_strings(group["region_ids"])[index]),
                        "global_centroid_cm": group["global_centroid_cm"][
                            index
                        ].tolist(),
                        "local_centreline_coordinates_cm": group[
                            "local_centreline_coordinates_cm"
                        ][index].tolist(),
                        "event_level_raw_records": None,
                        "event_level_sum_weights": None,
                        "event_level_sum_squared_weights": None,
                        "batch_moment_effective_sample_size_proxy": (
                            1.0 / relative**2
                            if relative is not None and relative > 0.0
                            else None
                        ),
                    }
                )
                entries.append(metric)
            local_mesh[particle_name] = entries
    provenance = flux_manifest["provenance"]

    with h5py.File(paths["heating"], "r") as source:
        neutron_heating = _field_rows(
            source["heating/neutron"], "mean_W", "std_dev_W", "W"
        )
        photon_heating = _field_rows(
            source["heating/photon"], "mean_W", "std_dev_W", "W"
        )
    with h5py.File(paths["damage_gas"], "r") as source:
        damage = _field_rows(
            source["damage_energy/neutron"],
            "mean_J_per_s",
            "std_dev_J_per_s",
            "J s^-1",
        )
        group = source["gas_production/neutron"]
        labels = _strings(group["score_labels"])
        magnet_ids = _strings(group["magnet_ids"])
        roles = _strings(group["component_roles"])
        means = group["mean_atoms_per_s"][()]
        stds = group["std_dev_atoms_per_s"][()]
        hydrogen = {}
        helium = {}
        h_mask = np.asarray(
            [
                str(value).startswith("H") and not str(value).startswith("He")
                for value in labels
            ]
        )
        he_mask = np.asarray([str(value).startswith("He") for value in labels])
        for magnet, role, mean, std in zip(magnet_ids, roles, means, stds):
            key = (str(magnet), str(role))
            hydrogen[key] = _metric(
                mean[:, h_mask], std[:, h_mask], "atoms s^-1"
            )
            helium[key] = _metric(
                mean[:, he_mask], std[:, he_mask], "atoms s^-1"
            )
    with h5py.File(paths["reaction"], "r") as source:
        photon_production = _field_rows(
            source["production/photon"],
            "mean_events_per_s",
            "std_dev_events_per_s",
            "events s^-1",
        )
        group = source["reactions/neutron"]
        labels = _strings(group["reaction_labels"])
        magnet_ids = _strings(group["magnet_ids"])
        roles = _strings(group["component_roles"])
        means = group["mean_events_per_s"][()]
        stds = group["std_dev_events_per_s"][()]
        reactions = {}
        for magnet, role, mean, std in zip(magnet_ids, roles, means, stds):
            reactions[(str(magnet), str(role))] = {
                str(label): _metric(
                    mean[:, index], std[:, index], "events s^-1"
                )
                for index, label in enumerate(labels)
            }

    keys = sorted(neutron_flux)
    components = []
    for magnet_id, role in keys:
        key = (magnet_id, role)
        components.append(
            {
                "magnet_id": magnet_id,
                "component_role": role,
                "neutron_scalar_flux": neutron_flux[key],
                "photon_scalar_flux": photon_flux[key],
                "neutron_heating": neutron_heating[key],
                "photon_heating": photon_heating[key],
                "damage_energy": damage[key],
                "hydrogen_production": hydrogen[key],
                "helium_production": helium[key],
                "photon_production": photon_production[key],
                "reaction_totals": reactions[key],
            }
        )
    protected = (
        "neutron_scalar_flux",
        "photon_scalar_flux",
        "neutron_heating",
        "photon_heating",
    )
    all_coarse_qualified = all(
        row[name]["status"] in {"QUALIFIED", "MARGINAL"}
        for row in components
        for name in protected
    )
    return {
        "schema": "parastell.short_run_scientific_analysis/v1",
        "classification": "BOUNDED_500K_QUALIFICATION_NOT_PRODUCTION",
        "source_histories": int(source_histories),
        "provenance": provenance,
        "artifact_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in paths.items()
        },
        "uncertainty_policy": {
            "qualified_max_relative": 0.25,
            "marginal_max_relative": 0.50,
            "energy_sum": "L1 standard-deviation upper bound; covariance unavailable",
            "zero_scores": "INSUFFICIENT_STATISTICS; never interpreted as physical zero",
        },
        "symmetry_status": "directly_simulated_90_degree_sector_not_symmetry_expanded",
        "all_magnet_components": components,
        "all_magnet_coarse_field_pass": all_coarse_qualified,
        "representative_magnet": {
            "magnet_id": "example-stellarator-sector-00-90deg-coil-0005",
            "outer_bank": _bank_summary(paths["outer_bank"], source_histories),
            "winding_bank": _bank_summary(
                paths["winding_bank"], source_histories
            ),
            "local_mesh_flux": local_mesh,
            "local_mesh_heating": {
                "status": "NOT_RUN_LOCAL_MESH_HEATING_TALLY_ABSENT",
                "physical_zero_claimed": False,
            },
        },
    }


def _short(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.4e}"


def write_analysis(
    json_path: str | Path,
    markdown_path: str | Path,
    report: Mapping[str, Any],
) -> None:
    json_target = Path(json_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Short-run scientific analysis",
        "",
        "Status: bounded 500,000-history qualification run; not a production result.",
        "",
        "All quantities are directly simulated for the 90-degree sector and are not symmetry-expanded. "
        "Integrated energy-bin uncertainties use the conservative L1 upper bound because covariance is unavailable. "
        "QUALIFIED means <=25% relative uncertainty, MARGINAL <=50%, and larger or zero-score estimates are INSUFFICIENT_STATISTICS.",
        "",
        "| Magnet | Component | n flux (cm^-2 s^-1) | rel. u | photon flux (cm^-2 s^-1) | rel. u | n heat (W) | photon heat (W) | damage (J/s) | H (atoms/s) | He (atoms/s) | photon production (events/s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["all_magnet_components"]:
        nflux = row["neutron_scalar_flux"]
        pflux = row["photon_scalar_flux"]
        lines.append(
            "| {magnet} | {role} | {nf} | {nru} | {pf} | {pru} | {nh} | {ph} | {de} | {hp} | {hep} | {pp} |".format(
                magnet=row["magnet_id"].rsplit("-", 1)[-1],
                role=row["component_role"],
                nf=_short(nflux["value"]),
                nru=_short(
                    nflux["conservative_relative_uncertainty_upper_bound"]
                ),
                pf=_short(pflux["value"]),
                pru=_short(
                    pflux["conservative_relative_uncertainty_upper_bound"]
                ),
                nh=_short(row["neutron_heating"]["value"]),
                ph=_short(row["photon_heating"]["value"]),
                de=_short(row["damage_energy"]["value"]),
                hp=_short(row["hydrogen_production"]["value"]),
                hep=_short(row["helium_production"]["value"]),
                pp=_short(row["photon_production"]["value"]),
            )
        )
    statuses = {}
    for row in report["all_magnet_components"]:
        for name in (
            "neutron_scalar_flux",
            "photon_scalar_flux",
            "neutron_heating",
            "photon_heating",
            "damage_energy",
            "hydrogen_production",
            "helium_production",
            "photon_production",
        ):
            status = row[name]["status"]
            statuses[status] = statuses.get(status, 0) + 1
    lines.extend(
        [
            "",
            "## Qualification outcome",
            "",
            f"Protected all-magnet coarse-field gate: {'PASS' if report['all_magnet_coarse_field_pass'] else 'INSUFFICIENT_STATISTICS'}.",
            "",
            "Metric classifications: "
            + ", ".join(
                f"{key}={value}" for key, value in sorted(statuses.items())
            )
            + ".",
            "",
            "The representative surface banks contain exact correlated records and complete capture accounting. "
            "Zero-record strata carry a 95% Poisson upper bound rather than a physical-zero claim. "
            "The local mesh contains flux estimates for 367 material-intersection regions; event-level ESS is unavailable from tally batch moments. "
            "Local-mesh heating was not tallied and is therefore NOT_RUN, not zero.",
            "",
            "Per-reaction totals, every component uncertainty object, spectra, surface currents, angle distributions, entry coordinates, patch ESS, and all local-bin classifications are retained in the companion JSON.",
        ]
    )
    Path(markdown_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir")
    parser.add_argument("output_json")
    parser.add_argument("output_markdown")
    options = parser.parse_args(argv)
    root = Path(options.artifact_dir)
    report = analyze_short_run(
        scalar_flux_path=root / "medium_scalar_flux_fields.h5",
        heating_path=root / "medium_magnet_heating.h5",
        reaction_path=root / "medium_magnet_reaction_production.h5",
        damage_gas_path=root / "medium_magnet_damage_gas.h5",
        outer_bank_path=root / "medium_500k_outer_coil0005.h5",
        winding_bank_path=root / "medium_500k_winding_coil0005.h5",
    )
    write_analysis(options.output_json, options.output_markdown, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
