"""Portable ParaStell-to-ALARA activation handoff and deck renderer.

This module prepares bounded ALARA inputs from validated OpenMC volume scalar
flux.  It never treats a surface bank as flux, never applies the physical
source rate more than once, and does not authorize production activation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .activation_campaign import (
    COOLING_OFFSETS_S,
    IRRADIATION_CHECKPOINTS_S,
    validate_activation_campaign,
)
from .energy_groups import (
    VITAMIN_J_175_EDGES_EV,
    VITAMIN_J_175_EDGES_SHA256,
)


SCHEMA = "parastell.alara_activation_handoff/v1.0.0"
GROUP_STRUCTURE = "VITAMIN-J-175"
GROUP_COUNT = 175
DELAYED_PHOTON_EDGES_EV = (0.0, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7)
ALARA_SOURCE_COMMIT = "4d01679a9837d9e8a2882c7efa71bc0b5f9ade64"
ALARA_BINARY_SHA256 = (
    "a28b8413a829e2df22e2a6a26d67328275511bfd66b6f85271b108a1c831e2d0"
)
ALARA_PATCH_SHA256 = (
    "cf93de3e57c52ed4a26a731daed94ceaf458ed79528b7dbcc78a173830d6fd0b"
)
FENDL_COMPONENT_SHA256 = {
    "fendl2.0-175.lib": (
        "b6e39841d89e2077510ee1c00e418c4fcb6049d315c2a81e33bfd3e3fd748488"
    ),
    "fendl2.0-175.idx": (
        "994507205078f712047c2ff2e9880cbbec6d071d3a40bb860bfb55e87abc2d4a"
    ),
    "fendl2.0-175.gam": (
        "aa6fb8c64acb9cf86e224dcb7e698c6d54a2b134858007c9cb8b3e7512223bd8"
    ),
    "fendl2.0-175.gdx": (
        "695ef08febfd361ec86e5bf2f83f317820112d80ca09471b4e74d569cd38eb0d"
    ),
    "nuclib.std": (
        "9f7ea31135938a708bad12308fd74741ef3e24593bbdadcea654bd9fce2a47f1"
    ),
}
_ALARA_ELEMENT = re.compile(r"^[a-z]{1,2}:[0-9]+$")
_ISOTOPE_LABEL = re.compile(r"^[a-z]{1,2}-[0-9]+(?:m[0-9]*)?$")


def qualified_alara_runtime() -> dict[str, Any]:
    return {
        "host": "poly-bateman",
        "alara_version": "2.9.2",
        "source_repository": "FusionSandwich/ALARA",
        "source_commit": ALARA_SOURCE_COMMIT,
        "binary_path": (
            "/home/apollon/josma/opt/"
            "alara-fusionsandwich-4d01679a9837-isonamefix-20260829/bin/alara"
        ),
        "binary_sha256": ALARA_BINARY_SHA256,
        "compiler_patch_path": (
            "/home/apollon/josma/codex-acquisitions/"
            "alara-isoname-overlap-fix-20260829.patch"
        ),
        "compiler_patch_sha256": ALARA_PATCH_SHA256,
        "activation_library": "FENDL/A-2.0",
        "decay_library": "FENDL/D-2.0",
        "library_base_path": (
            "/home/apollon/josma/data/"
            "alara-fendl2.0-175-vitj-e-20260828/binary/fendl2.0-175"
        ),
        "element_library_path": (
            "/home/apollon/josma/data/"
            "alara-fendl2.0-175-vitj-e-20260828/text/nuclib.std"
        ),
        "component_sha256": dict(FENDL_COMPONENT_SHA256),
        "group_structure": GROUP_STRUCTURE,
        "group_count": GROUP_COUNT,
        "cross_library_classification": (
            "OPENMC_TRANSPORT_PLUS_FENDL2_ACTIVATION_SMOKE_BASIS"
        ),
    }


def _positive(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _digest(value: Any, name: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return text


def _validated_constituents(
    material: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = material.get("alara_constituents")
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "each ALARA material needs explicit isotope constituents"
        )
    result = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("ALARA material constituent must be a mapping")
        symbol = str(row.get("element_symbol", "")).lower()
        if not _ALARA_ELEMENT.fullmatch(symbol):
            raise ValueError(
                "ALARA constituents must use explicit isotope symbols"
            )
        result.append(
            {
                "element_symbol": symbol,
                "relative_density": _positive(
                    row.get("relative_density"), "relative density"
                ),
                "volume_fraction": _positive(
                    row.get("volume_fraction"), "volume fraction"
                ),
            }
        )
    if not math.isclose(
        sum(row["volume_fraction"] for row in result),
        1.0,
        rel_tol=1.0e-10,
        abs_tol=1.0e-12,
    ):
        raise ValueError("ALARA constituent volume fractions must sum to one")
    return result


def build_alara_handoff(
    *,
    activation_rows: Sequence[Mapping[str, Any]],
    campaign: Mapping[str, Any],
    group_structure_sha256: str,
    openmc_provenance: Mapping[str, Any],
    delayed_photon_energy_edges_eV: Sequence[float] = (
        DELAYED_PHOTON_EDGES_EV
    ),
) -> dict[str, Any]:
    """Convert per-source OpenMC spectra to ALARA-order physical spectra."""
    validate_activation_campaign(campaign)
    runtime = qualified_alara_runtime()
    photon_edges = [float(value) for value in delayed_photon_energy_edges_eV]
    if (
        len(photon_edges) < 2
        or photon_edges[0] != 0.0
        or photon_edges != sorted(set(photon_edges))
        or any(
            not math.isfinite(value) or value < 0.0 for value in photon_edges
        )
    ):
        raise ValueError(
            "delayed-photon energy edges must increase from zero eV"
        )
    if group_structure_sha256 != VITAMIN_J_175_EDGES_SHA256:
        raise ValueError("group-structure hash is not OpenMC VITAMIN-J-175")
    rate = _positive(
        openmc_provenance.get("physical_source_rate_per_s"),
        "physical source rate",
    )
    if openmc_provenance.get("normalization") != "per_source_history":
        raise ValueError(
            "OpenMC activation spectra must be per source history"
        )
    if openmc_provenance.get("statistics_classification") not in {
        "WORKFLOW_SMOKE_ONLY",
        "INSUFFICIENT_STATISTICS",
        "QUALIFIED",
    }:
        raise ValueError("OpenMC statistics classification is missing")
    if openmc_provenance.get("provenance_origin") not in {
        "PHYSICAL_OPENMC_STATEPOINT",
        "SYNTHETIC_WORKFLOW_FIXTURE",
    }:
        raise ValueError("OpenMC provenance origin is missing")
    if not str(openmc_provenance.get("source_rate_scope", "")).strip():
        raise ValueError("OpenMC source-rate scope is missing")
    for key in (
        "raw_h5m_sha256",
        "canonical_geometry_fingerprint",
        "statepoint_sha256",
        "nuclear_data_sha256",
    ):
        _digest(openmc_provenance.get(key), key)

    zones = []
    domain_ids: set[str] = set()
    for source in activation_rows:
        domain_id = str(source.get("domain_id", "")).strip()
        if not domain_id or domain_id in domain_ids:
            raise ValueError("ALARA domain IDs must be nonempty and unique")
        domain_ids.add(domain_id)
        edges = [float(value) for value in source.get("energy_edges_eV", ())]
        per_source = [
            float(value)
            for value in source.get("bin_integrated_flux_per_source", ())
        ]
        if (
            len(edges) != GROUP_COUNT + 1
            or len(per_source) != GROUP_COUNT
            or edges != sorted(set(edges))
            or edges[0] < 0.0
            or any(
                not math.isfinite(value) or value < 0.0 for value in per_source
            )
        ):
            raise ValueError(
                "ALARA requires an ascending 176-edge VITAMIN-J-175 spectrum"
            )
        if tuple(edges) != VITAMIN_J_175_EDGES_EV:
            raise ValueError(
                "spectrum edges do not match OpenMC VITAMIN-J-175"
            )
        if not math.isclose(
            float(source.get("physical_source_rate_per_s", math.nan)),
            rate,
            rel_tol=1.0e-14,
            abs_tol=0.0,
        ):
            raise ValueError(
                "activation row source rate disagrees with provenance"
            )
        material = source.get("material")
        if not isinstance(material, Mapping):
            raise ValueError("ALARA zone material is missing")
        volume = _positive(source.get("volume_cm3"), "ALARA zone volume")
        constituents = _validated_constituents(material)
        zones.append(
            {
                "domain_id": domain_id,
                "volume_cm3": volume,
                "material_id": str(material.get("material_id", "")).strip(),
                "composition_sha256": _digest(
                    material.get("composition_sha256"), "material composition"
                ),
                "constituents": constituents,
                "physical_flux_descending_per_cm2_s": [
                    value * rate for value in reversed(per_source)
                ],
                "zero_flux": not any(per_source),
            }
        )
    if not zones:
        raise ValueError("ALARA handoff has no activation zones")
    result = {
        "schema": SCHEMA,
        "status": "READY_FOR_BOUNDED_EXECUTION",
        "claim": "WORKFLOW_SMOKE_ONLY",
        "runtime": runtime,
        "openmc_provenance": dict(openmc_provenance),
        "group_structure": {
            "name": GROUP_STRUCTURE,
            "group_count": GROUP_COUNT,
            "energy_edges_sha256": _digest(
                group_structure_sha256, "group structure"
            ),
            "openmc_order": "ascending_energy",
            "alara_order": "descending_energy",
            "interpolation_allowed": False,
        },
        "normalization": {
            "input": "per_source_history",
            "output": "physical_flux_per_cm2_s",
            "operation": "multiply_once_by_physical_source_rate",
            "physical_source_rate_per_s": rate,
            "surface_bank_used_as_flux": False,
        },
        "campaign": dict(campaign),
        "zones": zones,
        "delayed_photon_output": {
            "particle": "photon",
            "spatial_resolution": "activation_zone",
            "alara_output_normalization": "per_cm3_from_units_Bq_cm3",
            "nuclide_total_label": "TOTAL",
            "group_count": len(photon_edges) - 1,
            "energy_edges_eV": photon_edges,
            "group_order": "ascending_energy",
            "value_semantics": "bin_integrated_emission_rate_density",
            "unit": "photons/cm3/s/bin",
            "zone_integrated_rate_operation": (
                "multiply_once_by_bound_zone_volume_cm3"
            ),
        },
        "output_units": "Bq cm3",
        "production_activation_authorized": False,
    }
    validate_alara_handoff(result)
    return result


def validate_alara_handoff(handoff: Mapping[str, Any]) -> None:
    if handoff.get("schema") != SCHEMA:
        raise ValueError("unsupported ALARA handoff schema")
    if handoff.get("status") != "READY_FOR_BOUNDED_EXECUTION":
        raise ValueError("ALARA handoff is not ready for bounded execution")
    if handoff.get("claim") != "WORKFLOW_SMOKE_ONLY":
        raise ValueError("ALARA workflow may claim smoke execution only")
    runtime = handoff.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime != qualified_alara_runtime()
    ):
        raise ValueError("ALARA runtime identity is not the qualified runtime")
    group = handoff.get("group_structure")
    if not isinstance(group, Mapping) or (
        group.get("name") != GROUP_STRUCTURE
        or int(group.get("group_count", -1)) != GROUP_COUNT
        or group.get("openmc_order") != "ascending_energy"
        or group.get("alara_order") != "descending_energy"
        or group.get("interpolation_allowed") is not False
    ):
        raise ValueError("ALARA group-structure contract is invalid")
    _digest(group.get("energy_edges_sha256"), "group structure")
    if group.get("energy_edges_sha256") != VITAMIN_J_175_EDGES_SHA256:
        raise ValueError("ALARA group-structure hash is not qualified")
    normalization = handoff.get("normalization")
    if normalization != {
        "input": "per_source_history",
        "output": "physical_flux_per_cm2_s",
        "operation": "multiply_once_by_physical_source_rate",
        "physical_source_rate_per_s": (
            normalization.get("physical_source_rate_per_s")
            if isinstance(normalization, Mapping)
            else None
        ),
        "surface_bank_used_as_flux": False,
    }:
        raise ValueError("ALARA normalization contract is invalid")
    _positive(normalization["physical_source_rate_per_s"], "source rate")
    validate_activation_campaign(handoff.get("campaign", {}))
    zones = handoff.get("zones")
    if not isinstance(zones, list) or not zones:
        raise ValueError("ALARA handoff has no zones")
    for zone in zones:
        _positive(zone.get("volume_cm3"), "ALARA zone volume")
        if (
            len(zone.get("physical_flux_descending_per_cm2_s", ()))
            != GROUP_COUNT
        ):
            raise ValueError("ALARA zone spectrum does not have 175 groups")
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in zone["physical_flux_descending_per_cm2_s"]
        ):
            raise ValueError("ALARA zone spectrum contains an invalid value")
        _validated_constituents(
            {"alara_constituents": zone.get("constituents")}
        )
    photon_output = handoff.get("delayed_photon_output")
    if not isinstance(photon_output, Mapping):
        raise ValueError("ALARA delayed-photon output contract is invalid")
    photon_edges = [
        float(value) for value in photon_output.get("energy_edges_eV", ())
    ]
    if (
        len(photon_edges) < 2
        or photon_edges[0] != 0.0
        or photon_edges != sorted(set(photon_edges))
        or any(
            not math.isfinite(value) or value < 0.0 for value in photon_edges
        )
        or photon_output
        != {
            "particle": "photon",
            "spatial_resolution": "activation_zone",
            "alara_output_normalization": "per_cm3_from_units_Bq_cm3",
            "nuclide_total_label": "TOTAL",
            "group_count": len(photon_edges) - 1,
            "energy_edges_eV": photon_edges,
            "group_order": "ascending_energy",
            "value_semantics": "bin_integrated_emission_rate_density",
            "unit": "photons/cm3/s/bin",
            "zone_integrated_rate_operation": (
                "multiply_once_by_bound_zone_volume_cm3"
            ),
        }
    ):
        raise ValueError("ALARA delayed-photon output contract is invalid")
    if handoff.get("output_units") != "Bq cm3":
        raise ValueError("ALARA output must use Bq cm3 units")
    if handoff.get("production_activation_authorized") is not False:
        raise ValueError("production activation is not authorized")


def _alara_duration(value: int) -> str:
    """Render exact schedule seconds using ALARA's compact time labels."""
    seconds = int(value)
    compact = {
        0: "0 s",
        1: "1 s",
        60: "60 s",
        3_600: "1 h",
        86_400: "1 d",
        604_800: "1 w",
        2_592_000: "30 d",
        31_557_600: "1 y",
        157_788_000: "5 y",
        315_576_000: "10 y",
    }
    if seconds not in compact:
        raise ValueError("duration has no exact compact ALARA representation")
    return compact[seconds]


def _render_flux(zones: Sequence[Mapping[str, Any]]) -> str:
    values = [
        float(value)
        for zone in zones
        for value in zone["physical_flux_descending_per_cm2_s"]
    ]
    lines = []
    for index in range(0, len(values), 6):
        lines.append(
            " ".join(f"{value:.12E}" for value in values[index : index + 6])
        )
    return "\n".join(lines) + "\n"


def _render_input(handoff: Mapping[str, Any], *, irradiation_s: int) -> str:
    runtime = handoff["runtime"]
    zones = handoff["zones"]
    cooling = [
        int(value)
        for value in handoff["campaign"]["cooling_offset_s"]
        if int(value) > 0
    ]
    lines = ["geometry rectangular", "", "volumes"]
    for zone in zones:
        lines.append(f"  {zone['volume_cm3']:.12E}  {zone['domain_id']}")
    lines.extend(["end", "", "mat_loading"])
    for index, zone in enumerate(zones):
        lines.append(f"  {zone['domain_id']}  material_{index:04d}")
    lines.extend(["end", "", "spatial_norm"])
    lines.extend("  1.0" for _ in zones)
    photon_edges = handoff["delayed_photon_output"]["energy_edges_eV"]
    photon_source = (
        "  photon_source "
        f"{runtime['library_base_path']} delayed_photon_source.txt "
        f"{len(photon_edges) - 1} "
        + " ".join(f"{float(value):.12g}" for value in photon_edges[1:])
    )
    lines.extend(
        [
            "end",
            "",
            f"element_lib {runtime['element_library_path']}",
            "",
        ]
    )
    for index, zone in enumerate(zones):
        lines.append(f"mixture material_{index:04d}")
        for row in zone["constituents"]:
            lines.append(
                "  element {element_symbol} {relative_density:.12E} "
                "{volume_fraction:.12E}".format(**row)
            )
        lines.extend(["end", ""])
    lines.extend(
        [
            "flux full_power_flux flux.in 1.0 0 default",
            "",
            f"schedule irradiation_{irradiation_s}s",
            f"  {_alara_duration(irradiation_s)} full_power_flux steady_state 0 s",
            "end",
            "",
            "pulsehistory steady_state",
            "  1 0 s",
            "end",
            "",
            "cooling",
        ]
    )
    lines.extend(f"  {_alara_duration(value)}" for value in cooling)
    lines.extend(
        [
            "end",
            "",
            f"data_library alaralib {runtime['library_base_path']}",
            "dump_file activation.dump",
            "",
            "output zone",
            "  units Bq cm3",
            "  specific_activity",
            "  total_heat",
            photon_source,
            "end",
            "",
            "truncation 1e-8",
            "impurity 1e-8 1e-3",
            "",
        ]
    )
    return "\n".join(lines)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_alara_package(
    directory: str | Path, handoff: Mapping[str, Any]
) -> list[Path]:
    """Write create-only ALARA inputs for every independent checkpoint."""
    validate_alara_handoff(handoff)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    written: list[Path] = []
    for irradiation_s in handoff["campaign"]["irradiation_checkpoint_s"]:
        branch = root / f"irradiation_{int(irradiation_s):012d}s"
        branch.mkdir()
        flux = branch / "flux.in"
        input_path = branch / "alara.inp"
        flux.write_text(_render_flux(handoff["zones"]), encoding="utf-8")
        input_path.write_text(
            _render_input(handoff, irradiation_s=int(irradiation_s)),
            encoding="utf-8",
        )
        written.extend([flux, input_path])
    manifest = {
        "schema": "parastell.alara_input_package/v1.0.0",
        "status": "READY_FOR_BOUNDED_EXECUTION",
        "claim": "WORKFLOW_SMOKE_ONLY",
        "handoff": handoff,
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for path in written
        ],
        "production_activation_authorized": False,
    }
    manifest_path = root / "alara_input_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    written.append(manifest_path)
    return written


def validate_alara_output_text(text: str) -> dict[str, Any]:
    """Apply portable output sanity gates without inventing physical claims."""
    if "Response Units: Bq /cm3" not in text:
        raise ValueError("ALARA output does not record Bq/cm3 response units")
    if "Photon Source Distribution [gammas/s/cm3]" not in text:
        raise ValueError(
            "ALARA output does not record gammas/s/cm3 photon-source units"
        )
    labels = sorted(
        set(re.findall(r"\b[a-z]{1,2}-[0-9]+(?:m[0-9]*)?\b", text))
    )
    malformed = [
        label for label in labels if not _ISOTOPE_LABEL.fullmatch(label)
    ]
    if malformed:
        raise ValueError("ALARA output contains malformed isotope labels")
    if not labels:
        raise ValueError("ALARA output contains no isotope labels")
    required_section_markers = (
        "*** Specific Activity [Bq/cm3] ***",
        "*** Total Decay Heat [W/cm3] ***",
        "*** Photon Source Distribution",
    )
    if any(marker not in text for marker in required_section_markers):
        raise ValueError("ALARA output is missing a required response section")
    sections = {
        "specific_activity": text.split(
            "*** Specific Activity [Bq/cm3] ***", 1
        )[-1].split("*** Total Decay Heat [W/cm3] ***", 1)[0],
        "total_decay_heat": text.split("*** Total Decay Heat [W/cm3] ***", 1)[
            -1
        ].split("*** Photon Source Distribution", 1)[0],
    }
    cooling_labels = [
        _alara_duration(int(value)) for value in COOLING_OFFSETS_S if value > 0
    ]
    output_columns = ["pre-irrad", "shutdown", *cooling_labels]
    expected_header = " ".join(["isotope", "t_1/2(s)", *output_columns])
    totals: dict[str, dict[str, float]] = {}
    for name, section in sections.items():
        header_match = re.search(
            r"^\s*isotope\s+(.+?)\s*$", section, flags=re.MULTILINE
        )
        if header_match is None:
            raise ValueError(f"ALARA output is missing the {name} header")
        actual_header = " ".join(("isotope " + header_match.group(1)).split())
        if actual_header != expected_header:
            raise ValueError(
                f"ALARA {name} cooling columns differ from the contract"
            )
        match = re.search(r"^total\s+(.+)$", section, flags=re.MULTILINE)
        if match is None:
            raise ValueError(f"ALARA output is missing the {name} total row")
        values = [float(value) for value in match.group(1).split()]
        expected_value_count = 1 + len(output_columns)
        if len(values) != expected_value_count:
            raise ValueError(
                f"ALARA {name} total row does not match the cooling columns"
            )
        response_values = values[1:]
        if any(
            not math.isfinite(value) or value < 0.0
            for value in response_values
        ):
            raise ValueError(f"ALARA {name} total row contains invalid data")
        values_by_column = dict(zip(output_columns, response_values))
        pre_irradiation = values_by_column["pre-irrad"]
        shutdown = values_by_column["shutdown"]
        one_day = values_by_column["1 d"]
        if pre_irradiation != 0.0:
            raise ValueError(f"ALARA {name} pre-irradiation value is nonzero")
        if min(shutdown, one_day) <= 0.0:
            raise ValueError(f"ALARA {name} endpoint is not positive")
        totals[name] = dict(values_by_column)
    return {
        "schema": "parastell.alara_output_text_audit/v1.0.0",
        "status": "PASS",
        "input_units_directive": "Bq cm3",
        "reported_activity_units": "Bq/cm3",
        "reported_heat_units": "W/cm3",
        "reported_delayed_photon_units": "gammas/s/cm3/bin",
        "isotope_label_count": len(labels),
        "all_isotope_labels_well_formed": True,
        "pre_irradiation_activity_and_heat_zero": True,
        "shutdown_and_one_day_activity_and_heat_positive": True,
        "cooling_schedule_columns_validated": True,
        "cooling_offset_s": list(COOLING_OFFSETS_S),
        "cooling_output_columns": output_columns,
        "totals": totals,
        "scientific_claim": "WORKFLOW_SMOKE_ONLY",
    }


def validate_alara_result_files(
    *,
    output_path: str | Path,
    stderr_path: str | Path,
    delayed_photon_path: str | Path,
    input_manifest_sha256: str,
    remote_run_root: str,
    provenance_origin: str,
    irradiation_checkpoint_s: int | None = None,
    alara_input_sha256: str | None = None,
    activation_zone_ids: Sequence[str] | None = None,
    cooling_offset_s: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Seal a terminal bounded result from the qualified ALARA runtime."""
    output = Path(output_path)
    stderr = Path(stderr_path)
    photon = Path(delayed_photon_path)
    for path in (output, stderr, photon):
        if not path.is_file():
            raise FileNotFoundError(path)
    if stderr.stat().st_size != 0:
        raise ValueError("ALARA stderr is not empty")
    if photon.stat().st_size <= 0:
        raise ValueError("ALARA delayed-photon output is empty")
    if provenance_origin not in {
        "PHYSICAL_OPENMC_STATEPOINT",
        "SYNTHETIC_WORKFLOW_FIXTURE",
    }:
        raise ValueError("ALARA result provenance origin is invalid")
    audit = validate_alara_output_text(output.read_text(encoding="utf-8"))
    result = {
        "schema": "parastell.alara_bounded_result/v1.0.0",
        "status": "PASSED",
        "claim": "WORKFLOW_SMOKE_ONLY",
        "provenance_origin": provenance_origin,
        "provenance_assignment": "explicit_at_result_sealing",
        "host": "poly-bateman",
        "remote_run_root": str(remote_run_root),
        "runtime": qualified_alara_runtime(),
        "input_manifest_sha256": _digest(
            input_manifest_sha256, "ALARA input manifest"
        ),
        "files": {
            "output": {
                "size_bytes": output.stat().st_size,
                "sha256": _file_sha256(output),
            },
            "stderr": {
                "size_bytes": stderr.stat().st_size,
                "sha256": _file_sha256(stderr),
            },
            "delayed_photon_source": {
                "size_bytes": photon.stat().st_size,
                "sha256": _file_sha256(photon),
            },
        },
        "output_audit": audit,
        "resource_controls": {
            "requested_cores": 1,
            "memory_max_bytes": 8 * 1024**3,
            "memory_swap_max_bytes": 0,
            "observed_peak_bytes": None,
            "observed_peak_status": "NOT_RETAINED_AFTER_SCOPE_REAP",
        },
        "cross_library_limitation": (
            "OpenMC transport and FENDL-2 activation/decay are a workflow "
            "smoke basis, not a qualified production comparison"
        ),
        "mcnp_executed": False,
        "production_activation_authorized": False,
    }
    execution_fields = (
        irradiation_checkpoint_s,
        alara_input_sha256,
        activation_zone_ids,
        cooling_offset_s,
    )
    if provenance_origin == "PHYSICAL_OPENMC_STATEPOINT" and any(
        value is None for value in execution_fields
    ):
        raise ValueError(
            "physical ALARA results require a complete execution binding"
        )
    if any(value is not None for value in execution_fields):
        if any(value is None for value in execution_fields):
            raise ValueError(
                "ALARA delayed-source execution binding must be complete"
            )
        checkpoint = int(irradiation_checkpoint_s)
        if checkpoint not in IRRADIATION_CHECKPOINTS_S:
            raise ValueError(
                "irradiation checkpoint differs from the campaign contract"
            )
        zone_ids = [str(value).strip() for value in activation_zone_ids]
        if (
            not zone_ids
            or any(not value for value in zone_ids)
            or len(set(zone_ids)) != len(zone_ids)
        ):
            raise ValueError("activation zone IDs must be nonempty and unique")
        offsets = [int(value) for value in cooling_offset_s]
        if tuple(offsets) != COOLING_OFFSETS_S:
            raise ValueError(
                "cooling offsets differ from the campaign contract"
            )
        result["execution_binding"] = {
            "irradiation_checkpoint_s": checkpoint,
            "alara_input_sha256": _digest(
                alara_input_sha256, "ALARA checkpoint input"
            ),
            "activation_zone_ids": zone_ids,
            "cooling_offset_s": offsets,
        }
    return result
