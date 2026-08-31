"""Run a bounded full-response OpenMC 0.16 smoke on selected magnets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from parastell.parametric_openmc16_model import (
    _canonical_sha,
    _statepoint_batches,
)
from parastell.openmc16 import add_reactor_component_tallies
from parastell.selected_case_instrumentation import instrument_selected_case
from parastell.surface_source_instrumentation import (
    build_surface_instrumentation_spec,
)
from parastell.transport_response_plan import validate_response_plan
from scripts.prepare_accepted_magnet_inventory import _validated_magnets
from scripts.run_openmc16_dagmc_surface_qualification import (
    _replace_dagmc_filename,
)
from scripts.run_parametric_openmc16_geometry_debug import (
    _load_bound_receipt,
    _nuclear_data_hashes,
    _runtime_inputs,
    sha256_file,
)


SCHEMA = "parastell.parametric_openmc16_full_response_smoke/v1.0.0"


def _location_mesh_filters(
    openmc,
    model,
    *,
    cell_ids: list[int],
    bins_per_axis: int,
) -> tuple[dict[int, Any], dict[str, Any] | None]:
    if isinstance(bins_per_axis, bool) or not isinstance(bins_per_axis, int):
        raise ValueError("local mesh bins per axis must be an integer")
    if bins_per_axis == 0:
        return {}, None
    if bins_per_axis < 2 or bins_per_axis > 64:
        raise ValueError("local mesh bins per axis must be between 2 and 64")
    lower, upper = model.geometry.bounding_box
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if (
        lower.shape != (3,)
        or upper.shape != (3,)
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(upper <= lower)
    ):
        raise ValueError("model bounding box is not finite for location mesh")
    mesh = openmc.RegularMesh(name="pstl_global_location_mesh")
    mesh.lower_left = lower.tolist()
    mesh.upper_right = upper.tolist()
    mesh.dimension = [bins_per_axis] * 3
    mesh_filter = openmc.MeshFilter(mesh)
    unique_cells = sorted(set(int(value) for value in cell_ids))
    if not unique_cells or unique_cells[0] <= 0:
        raise ValueError("location mesh cell IDs are invalid")
    return (
        {cell: mesh_filter for cell in unique_cells},
        {
            "scope": "global_mesh_cell_filtered_to_selected_magnet",
            "lower_left_cm": mesh.lower_left,
            "upper_right_cm": mesh.upper_right,
            "dimension": mesh.dimension,
            "selected_cell_ids": unique_cells,
            "empty_bins_mean_outside_selected_cell_not_zero_response": True,
        },
    )


def _validate_smoke_controls(
    *,
    particles: int,
    batches: int,
    seed: int,
    max_particles: int,
    threads: int,
    timeout_seconds: int,
) -> None:
    controls = (
        particles,
        batches,
        seed,
        max_particles,
        threads,
        timeout_seconds,
    )
    if min(controls) <= 0 or particles * batches > 100_000:
        raise ValueError(
            "smoke controls are invalid or exceed 100,000 histories"
        )
    if threads > 4:
        raise ValueError("bounded smoke is capped at four OpenMP threads")
    if max_particles > 1_000_000:
        raise ValueError("bounded smoke bank capacity is capped at 1,000,000")
    if timeout_seconds > 3_600:
        raise ValueError("bounded smoke timeout is capped at 3,600 seconds")


def _load_response_plan(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("response-plan file hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    plan = payload.get("response_plan", payload)
    if not isinstance(plan, dict):
        raise ValueError("response-plan payload is invalid")
    validate_response_plan(plan)
    return plan


def _selected_surface_inputs(
    path: Path,
    dagmc_h5m: Path,
    magnet_ids: list[str],
    expected_sha256: str,
) -> tuple[dict[int, int], dict[str, int], str]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("surface-manifest file hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        == "parastell.continuous_magnet_surface_manifest/v1.0.0"
    ):
        capture = payload.get("capture_bank", {})
        if (
            payload.get("status") != "PASS"
            or payload.get("dagmc_sha256") != sha256_file(dagmc_h5m)
            or payload.get("geometry", {}).get("volume_count") != 9
            or payload.get("geometry", {}).get("continuous_magnet_volume_id")
            != 9
            or capture.get("closed") is not True
            or capture.get("records_all_entering_and_leaving_crossings")
            is not True
            or payload.get("physical_h5m_mutation") is not False
            or magnet_ids != ["continuous-magnet-layer"]
        ):
            raise ValueError("continuous surface manifest is not accepted")
        declared = sorted(
            int(value) for value in capture.get("surface_ids", ())
        )
        signs = {
            int(row["dagmc_surface_id"]): int(
                row["magnet_outward_normal_multiplier"]
            )
            for row in payload.get("surfaces", ())
            if row.get("complete_boundary_capture") is True
        }
        if sorted(signs) != declared or any(
            value not in {-1, 1} for value in signs.values()
        ):
            raise ValueError("continuous capture surface signs are invalid")
        return (
            signs,
            {"continuous-magnet-layer": 9},
            "continuous_magnet_complete_boundary",
        )
    if (
        payload.get("schema")
        != "parastell.parametric_magnet_surface_manifest/v1.0.0"
        or payload.get("status") != "PASS"
        or payload.get("dagmc_sha256") != sha256_file(dagmc_h5m)
        or payload.get("coupling_interface")
        != "homogenized_magnet_outer_boundary"
        or payload.get("magnet_count") != 18
        or payload.get("all_envelopes_close") is not True
        or payload.get("physical_h5m_mutation") is not False
    ):
        raise ValueError("surface manifest is not accepted")
    magnets = _validated_magnets(payload)
    by_id = {row["magnet"]: row for row in magnets}
    if not magnet_ids or len(magnet_ids) != len(set(magnet_ids)):
        raise ValueError("selected magnet IDs must be nonempty and unique")
    if any(item not in by_id for item in magnet_ids):
        raise ValueError("response plan selects an unknown magnet ID")
    signs: dict[int, int] = {}
    cell_ids: dict[str, int] = {}
    for magnet_id in magnet_ids:
        row = by_id[magnet_id]
        declared = sorted(int(value) for value in row["dagmc_surface_ids"])
        observed = []
        for surface in row.get("surfaces", []):
            surface_id = int(surface["dagmc_surface_id"])
            sign = int(surface["magnet_outward_normal_multiplier"])
            if sign not in {-1, 1} or surface_id in signs:
                raise ValueError(
                    "selected surface identities/signs are invalid"
                )
            observed.append(surface_id)
            signs[surface_id] = sign
        if sorted(observed) != declared:
            raise ValueError(
                "selected surface rows disagree with declared IDs"
            )
        cell_ids[magnet_id] = int(row["dagmc_volume_id"])
    return signs, cell_ids, "homogenized_magnet_outer_boundary"


def _run_openmc(
    output: Path, *, threads: int, timeout_seconds: int
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["openmc", "-s", str(threads)],
            cwd=output,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
            timeout=timeout_seconds,
        )
        return {
            "exit_code": int(completed.returncode),
            "timed_out": False,
            "output": completed.stdout,
        }
    except subprocess.TimeoutExpired as error:
        value = error.stdout or ""
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        return {"exit_code": -1, "timed_out": True, "output": value}


def main() -> None:
    import openmc

    parser = argparse.ArgumentParser()
    parser.add_argument("model_xml", type=Path)
    parser.add_argument("model_receipt", type=Path)
    parser.add_argument("dagmc_h5m", type=Path)
    parser.add_argument("surface_manifest", type=Path)
    parser.add_argument("response_plan", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--expected-model-receipt-sha256", required=True)
    parser.add_argument("--expected-surface-manifest-sha256", required=True)
    parser.add_argument("--expected-response-plan-sha256", required=True)
    parser.add_argument("--particles", type=int, default=500)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--statepoint-interval-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=8_290_831)
    parser.add_argument("--max-particles", type=int, default=100_000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--local-mesh-bins-per-axis", type=int, default=0)
    args = parser.parse_args()

    if openmc.__version__ != "0.16.0":
        raise RuntimeError(f"OpenMC 0.16.0 required, got {openmc.__version__}")
    _validate_smoke_controls(
        particles=args.particles,
        batches=args.batches,
        seed=args.seed,
        max_particles=args.max_particles,
        threads=args.threads,
        timeout_seconds=args.timeout_seconds,
    )

    model_xml = args.model_xml.resolve(strict=True)
    model_receipt = args.model_receipt.resolve(strict=True)
    dagmc_h5m = args.dagmc_h5m.resolve(strict=True)
    surface_manifest = args.surface_manifest.resolve(strict=True)
    response_plan_path = args.response_plan.resolve(strict=True)
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt = _load_bound_receipt(
        model_receipt,
        args.expected_model_receipt_sha256,
        model_xml=model_xml,
        dagmc_h5m=dagmc_h5m,
    )
    response_plan = _load_response_plan(
        response_plan_path, args.expected_response_plan_sha256
    )
    signs, magnet_cell_ids, coupling_interface = _selected_surface_inputs(
        surface_manifest,
        dagmc_h5m,
        list(response_plan["magnet_ids"]),
        args.expected_surface_manifest_sha256,
    )
    before = {
        "model_xml": sha256_file(model_xml),
        "model_receipt": sha256_file(model_receipt),
        "surface_manifest": sha256_file(surface_manifest),
        "response_plan": sha256_file(response_plan_path),
        "runtime_inputs": _runtime_inputs(model_xml, dagmc_h5m, receipt),
        "nuclear_data": _nuclear_data_hashes(receipt),
    }

    model = openmc.Model.from_model_xml(model_xml)
    dagmc_universe_id = _replace_dagmc_filename(model, dagmc_h5m)
    model.settings.run_mode = "fixed source"
    model.settings.particles = args.particles
    model.settings.batches = args.batches
    model.settings.seed = args.seed
    expected_statepoint_batches = _statepoint_batches(
        {
            "batches": args.batches,
            "statepoint_interval_batches": args.statepoint_interval_batches,
        }
    )
    model.settings.statepoint = {"batches": expected_statepoint_batches}
    model.settings.output = {"tallies": False}
    model.tallies = openmc.Tallies()
    local_mesh_filters, location_mesh = _location_mesh_filters(
        openmc,
        model,
        cell_ids=list(magnet_cell_ids.values()),
        bins_per_axis=args.local_mesh_bins_per_axis,
    )
    surface_spec = build_surface_instrumentation_spec(
        surface_ids=sorted(signs),
        energy_edges_by_particle=response_plan["energy_axes_eV"],
        openmc_normal_sign_by_surface=signs,
        max_particles_per_process=args.max_particles,
        max_source_files=1,
        mpi_ranks=1,
        coupling_interface=coupling_interface,
    )
    wiring = instrument_selected_case(
        model,
        response_plan=response_plan,
        surface_spec=surface_spec,
        magnet_cell_ids=magnet_cell_ids,
        tally_profile="activation_ready",
        local_mesh_filters_by_cell=local_mesh_filters,
    )
    component_cell_ids = receipt.get("component_cell_ids")
    if not isinstance(component_cell_ids, dict):
        raise ValueError("model receipt lacks physical component cell IDs")
    reactor_wiring = add_reactor_component_tallies(
        model,
        component_cell_ids=component_cell_ids,
        neutron_edges_eV=response_plan["energy_axes_eV"]["neutron"],
        photon_edges_eV=response_plan["energy_axes_eV"]["photon"],
    )
    run_model = output / "model.xml"
    model.export_to_model_xml(run_model)
    run_runtime_inputs_before = _runtime_inputs(run_model, dagmc_h5m, receipt)
    run = _run_openmc(
        output, threads=args.threads, timeout_seconds=args.timeout_seconds
    )
    log = output / "openmc.log"
    log.write_text(run["output"], encoding="utf-8")
    after = {
        "model_xml": sha256_file(model_xml),
        "model_receipt": sha256_file(model_receipt),
        "surface_manifest": sha256_file(surface_manifest),
        "response_plan": sha256_file(response_plan_path),
        "runtime_inputs": _runtime_inputs(model_xml, dagmc_h5m, receipt),
        "nuclear_data": _nuclear_data_hashes(receipt),
    }
    run_runtime_inputs_after = _runtime_inputs(run_model, dagmc_h5m, receipt)
    immutable = bool(
        before == after
        and run_runtime_inputs_before == before["runtime_inputs"]
        and run_runtime_inputs_after == run_runtime_inputs_before
    )
    statepoint_rows = []
    for path in output.glob("statepoint.*.h5"):
        try:
            batch = int(path.name.split(".")[1])
        except (IndexError, ValueError):
            batch = -1
        statepoint_rows.append((batch, path))
    statepoint_rows.sort(key=lambda row: (row[0], row[1].name))
    observed_statepoint_batches = [row[0] for row in statepoint_rows]
    statepoints = [row[1] for row in statepoint_rows]
    banks = sorted(output.glob("surface_source*.h5"))
    passed = bool(
        not run["timed_out"]
        and run["exit_code"] == 0
        and immutable
        and observed_statepoint_batches == expected_statepoint_batches
        and banks
    )
    result = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "BLOCKED_FULL_RESPONSE_SMOKE",
        "claim": "BOUNDED_INFRASTRUCTURE_SMOKE_ONLY",
        "run_mode": "fixed source",
        "openmc_version": openmc.__version__,
        "histories": args.particles * args.batches,
        "particles_per_batch": args.particles,
        "batches": args.batches,
        "statepoint_policy": {
            "interval_batches": args.statepoint_interval_batches,
            "expected_batches": expected_statepoint_batches,
            "observed_batches": observed_statepoint_batches,
            "complete": observed_statepoint_batches
            == expected_statepoint_batches,
        },
        "seed": args.seed,
        "location_mesh": location_mesh,
        "threads": args.threads,
        "timeout_seconds": args.timeout_seconds,
        "timed_out": run["timed_out"],
        "exit_code": run["exit_code"],
        "dagmc_universe_id": dagmc_universe_id,
        "source_normalization": receipt["source"],
        "input_hashes": before,
        "output_model_runtime_inputs": run_runtime_inputs_before,
        "inputs_immutable": immutable,
        "all_bound_inputs_immutable": immutable,
        "selected_magnet_ids": list(response_plan["magnet_ids"]),
        "selected_magnet_cell_ids": magnet_cell_ids,
        "surface_instrumentation": surface_spec,
        "full_response_wiring": wiring,
        "reactor_component_wiring": reactor_wiring,
        "instrumented_model_xml": {
            "path": str(run_model),
            "sha256": sha256_file(run_model),
            "tallies_instrumented": True,
            "surface_bank_instrumented": True,
            "response_plan_sha256": before["response_plan"],
            "surface_manifest_sha256": before["surface_manifest"],
        },
        "output_model_xml_sha256": sha256_file(run_model),
        "openmc_log_sha256": sha256_file(log),
        "statepoints": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in statepoints
        ],
        "surface_source_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in banks
        ],
        "surface_bank_classification": "POSTPROCESSING_REQUIRED",
        "statistics_qualified": False,
        "production_run_authorized": False,
    }
    result["receipt_content_sha256"] = _canonical_sha(result)
    result_path = output / "FULL_RESPONSE_SMOKE_RECEIPT.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(result_path)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
