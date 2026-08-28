"""Run and aggregate the two preregistered bounded OpenMC geometry checks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time

from parastell.openmc_geometry_debug import aggregate_geometry_debug_replicas
from parastell.openmc_geometry_debug import (
    categorical_geometry_debug_signature,
)
from parastell.openmc_geometry_debug import parse_openmc_geometry_debug_log
from parastell.reference_geometry import sha256_file
from scripts.generate_openmc_geometry_debug import generate
from scripts.generate_openmc_geometry_debug import model_xml_contract
from scripts.generate_openmc_geometry_debug import nuclear_data_manifest


REQUIRED_OPENMC_VERSION = "0.16.0"
REQUIRED_OPENMC_COMMIT = "617d35a5063c57796b43428bc401e627d2011046"


def _summary_material_id_names(summary) -> list[list[int | str]]:
    materials = summary.materials
    if isinstance(materials, dict):
        rows = [
            [int(identifier), material.name]
            for identifier, material in materials.items()
        ]
    else:
        rows = [[int(material.id), material.name] for material in materials]
    return sorted(rows)


def _command(
    command: list[str], *, cwd: Path | None, timeout_seconds: int
) -> dict:
    clock = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=int(timeout_seconds),
        )
        return {
            "command": command,
            "exit_code": int(result.returncode),
            "timed_out": False,
            "wall_time_seconds": time.monotonic() - clock,
            "output": result.stdout,
        }
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return {
            "command": command,
            "exit_code": None,
            "timed_out": True,
            "wall_time_seconds": time.monotonic() - clock,
            "output": output,
        }


def _openmc_identity(timeout_seconds: int) -> dict:
    result = _command(
        ["openmc", "--version"], cwd=None, timeout_seconds=timeout_seconds
    )
    version = re.search(r"OpenMC version\s+([^\s]+)", result["output"])
    commit = re.search(r"Commit hash:\s*([0-9a-f]{40})", result["output"])
    identity = {
        **result,
        "version": version.group(1) if version else None,
        "commit": commit.group(1) if commit else None,
    }
    identity["pass"] = (
        result["exit_code"] == 0
        and identity["version"] == REQUIRED_OPENMC_VERSION
        and identity["commit"] == REQUIRED_OPENMC_COMMIT
    )
    return identity


def _read_statepoint(
    openmc,
    statepoint: Path,
    summary: Path,
    *,
    expected_seed: int,
    expected_model_contract: dict,
) -> dict:
    result = {
        "statepoint_path": str(statepoint),
        "statepoint_exists": statepoint.is_file(),
        "summary_path": str(summary),
        "summary_exists": summary.is_file(),
        "readable": False,
        "error": None,
    }
    if not statepoint.is_file() or not summary.is_file():
        return result
    try:
        import h5py

        with openmc.StatePoint(statepoint) as loaded:
            result["statepoint_batches"] = int(loaded.n_batches)
            result["statepoint_particles"] = int(loaded.n_particles)
            run_mode = loaded.run_mode
            if isinstance(run_mode, bytes):
                run_mode = run_mode.decode("utf-8")
            result["statepoint_run_mode"] = str(run_mode)
        with h5py.File(statepoint, "r") as handle:
            result["statepoint_seed"] = int(handle["seed"][()])
            raw_mode = handle["run_mode"][()]
            if isinstance(raw_mode, bytes):
                raw_mode = raw_mode.decode("utf-8")
            result["statepoint_run_mode"] = str(raw_mode)
        result["statepoint_seed_matches_expected"] = result[
            "statepoint_seed"
        ] == int(expected_seed)
        loaded_summary = openmc.Summary(summary)
        geometry = loaded_summary.geometry
        result["summary_geometry_present"] = geometry is not None
        if geometry is not None:
            root = geometry.root_universe
            all_cells = geometry.get_all_cells()
            all_surfaces = geometry.get_all_surfaces()
            all_universes = geometry.get_all_universes()
            result["summary_contract"] = {
                "root_universe_id": int(root.id),
                "cell_ids": sorted(int(value) for value in all_cells),
                "material_id_names": _summary_material_id_names(
                    loaded_summary
                ),
                "surface_ids": sorted(int(value) for value in all_surfaces),
                "universe_ids": sorted(int(value) for value in all_universes),
            }
            expected_wrapper = expected_model_contract["wrapper_cell"]
            expected_surface = expected_model_contract["wrapper_surface"]
            expected_materials = sorted(
                [
                    [row["id"], row["name"]]
                    for row in expected_model_contract["materials"]
                ]
            )
            result["summary_contract_pass"] = bool(
                result["summary_contract"]["root_universe_id"]
                == expected_wrapper["universe"]
                and expected_wrapper["id"]
                in result["summary_contract"]["cell_ids"]
                and expected_surface["id"]
                in result["summary_contract"]["surface_ids"]
                and expected_model_contract["dagmc_universe_id"]
                in result["summary_contract"]["universe_ids"]
                and result["summary_contract"]["material_id_names"]
                == expected_materials
            )
        else:
            result["summary_contract_pass"] = False
        result["readable"] = True
    except Exception as error:  # fail-closed receipt records the exact error
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def _run_replica(
    *,
    openmc,
    dagmc: Path,
    source_mesh: Path,
    output: Path,
    cross_sections: Path,
    expected_dagmc_sha256: str,
    expected_source_mesh_sha256: str,
    acceptance_criteria: Path,
    expected_acceptance_criteria_sha256: str,
    source_domain_audit: Path,
    expected_source_domain_audit_sha256: str,
    expected_reference_source_mesh_sha256: str,
    expected_reference_source_fingerprint: str,
    expected_nuclear_data_manifest_sha256: str,
    seed: int,
    threads: int,
    timeout_seconds: int,
    identity: dict,
) -> dict:
    generate(
        dagmc,
        source_mesh,
        output,
        str(cross_sections),
        expected_dagmc_sha256=expected_dagmc_sha256,
        expected_source_mesh_sha256=expected_source_mesh_sha256,
        acceptance_criteria=acceptance_criteria,
        expected_acceptance_criteria_sha256=(
            expected_acceptance_criteria_sha256
        ),
        source_domain_audit=source_domain_audit,
        expected_source_domain_audit_sha256=(
            expected_source_domain_audit_sha256
        ),
        expected_reference_source_mesh_sha256=(
            expected_reference_source_mesh_sha256
        ),
        expected_reference_source_fingerprint=(
            expected_reference_source_fingerprint
        ),
        expected_nuclear_data_manifest_sha256=(
            expected_nuclear_data_manifest_sha256
        ),
        seed=seed,
    )
    input_receipt = output / "input_receipt.json"
    model_xml = output / "model.xml"
    h5m_before = sha256_file(dagmc)
    source_before = sha256_file(source_mesh)
    criteria_before = sha256_file(acceptance_criteria)
    source_domain_before = sha256_file(source_domain_audit)
    cross_sections_before = sha256_file(cross_sections)
    nuclear_data_before = nuclear_data_manifest(cross_sections)
    model_xml_before = sha256_file(model_xml)
    model_contract_before = model_xml_contract(model_xml)
    receipt = json.loads(input_receipt.read_text(encoding="utf-8"))
    run = _command(
        ["openmc", "-g", "-s", str(int(threads))],
        cwd=output,
        timeout_seconds=timeout_seconds,
    )
    log = output / "openmc_geometry_debug.log"
    log.write_text(run["output"], encoding="utf-8")
    h5m_after = sha256_file(dagmc)
    source_after = sha256_file(source_mesh)
    criteria_after = sha256_file(acceptance_criteria)
    source_domain_after = sha256_file(source_domain_audit)
    cross_sections_after = sha256_file(cross_sections)
    nuclear_data_after = nuclear_data_manifest(cross_sections)
    model_xml_after = sha256_file(model_xml)
    model_contract_after = model_xml_contract(model_xml)
    parsed = parse_openmc_geometry_debug_log(
        run["output"],
        exit_code=run["exit_code"] if run["exit_code"] is not None else -1,
        expected_threads=threads,
        required_cell_ids=receipt["native_geometry_ids"]["volume_ids"],
    )
    state = _read_statepoint(
        openmc,
        output / "statepoint.2.h5",
        output / "summary.h5",
        expected_seed=seed,
        expected_model_contract=model_contract_before,
    )
    binding_pass = (
        receipt.get("schema") == "parastell.openmc_geometry_debug_input/v1.2.0"
        and receipt["seed"] == int(seed)
        and receipt["raw_h5m_sha256_before"] == expected_dagmc_sha256
        and receipt["raw_h5m_sha256_after_model_export"]
        == expected_dagmc_sha256
        and receipt["source_mesh_sha256"] == expected_source_mesh_sha256
        and receipt["acceptance_criteria_sha256"]
        == expected_acceptance_criteria_sha256
        and receipt["source_domain_audit_sha256"]
        == expected_source_domain_audit_sha256
        == source_domain_before
        and receipt["expected_source_domain_audit_sha256"]
        == expected_source_domain_audit_sha256
        and receipt["reference_source_mesh_sha256"]
        == expected_reference_source_mesh_sha256
        and receipt["reference_source_fingerprint"]
        == expected_reference_source_fingerprint
        and receipt["cross_sections_sha256"] == cross_sections_before
        and receipt["nuclear_data_manifest"] == nuclear_data_before
        and receipt["nuclear_data_manifest_sha256"]
        == nuclear_data_before["manifest_sha256"]
        == expected_nuclear_data_manifest_sha256
        and receipt["expected_nuclear_data_manifest_sha256"]
        == expected_nuclear_data_manifest_sha256
        and receipt["model_xml_contract_pass"] is True
        and receipt["model_xml_contract"] == model_contract_before
        and receipt["model_xml_contract"]["model_xml_sha256"]
        == model_xml_before
        and receipt["histories"] == 4000
        and receipt["batches"] == 2
        and receipt["particles_per_batch"] == 2000
    )
    replica = {
        "schema": "parastell.openmc_geometry_debug_replica/v1.2.0",
        "seed": int(seed),
        "threads": int(threads),
        "raw_h5m_sha256": expected_dagmc_sha256,
        "source_mesh_sha256": expected_source_mesh_sha256,
        "acceptance_criteria_sha256": expected_acceptance_criteria_sha256,
        "openmc_version": identity["version"],
        "openmc_commit": identity["commit"],
        "source_domain_audit_sha256": source_domain_before,
        "source_point_cm": receipt["source_point_cm"],
        "reference_source_mesh_sha256": (
            expected_reference_source_mesh_sha256
        ),
        "reference_source_fingerprint": expected_reference_source_fingerprint,
        "cross_sections_sha256": cross_sections_before,
        "nuclear_data_manifest": nuclear_data_before,
        "nuclear_data_manifest_sha256": nuclear_data_before["manifest_sha256"],
        "model_xml_seed_normalized_fingerprint": model_contract_before[
            "seed_normalized_fingerprint"
        ],
        "histories": receipt["histories"],
        "batches": receipt["batches"],
        "particles_per_batch": receipt["particles_per_batch"],
        "command": run["command"],
        "exit_code": run["exit_code"],
        "timed_out": run["timed_out"],
        "wall_time_seconds": run["wall_time_seconds"],
        "input_receipt_sha256": sha256_file(input_receipt),
        "model_xml_sha256_before_transport": model_xml_before,
        "model_xml_sha256_after_transport": model_xml_after,
        "model_xml_contract_before_transport": model_contract_before,
        "model_xml_contract_after_transport": model_contract_after,
        "log_sha256": sha256_file(log),
        "raw_h5m_sha256_before_transport": h5m_before,
        "raw_h5m_sha256_after_transport": h5m_after,
        "source_mesh_sha256_before_transport": source_before,
        "source_mesh_sha256_after_transport": source_after,
        "acceptance_criteria_sha256_before_transport": criteria_before,
        "acceptance_criteria_sha256_after_transport": criteria_after,
        "source_domain_audit_sha256_before_transport": source_domain_before,
        "source_domain_audit_sha256_after_transport": source_domain_after,
        "cross_sections_sha256_before_transport": cross_sections_before,
        "cross_sections_sha256_after_transport": cross_sections_after,
        "nuclear_data_manifest_after_transport": nuclear_data_after,
        "input_binding_pass": binding_pass,
        "transport_input_immutability_pass": (
            h5m_before == h5m_after == expected_dagmc_sha256
            and source_before == source_after == expected_source_mesh_sha256
            and criteria_before
            == criteria_after
            == expected_acceptance_criteria_sha256
            and source_domain_before == source_domain_after
            and cross_sections_before == cross_sections_after
            and nuclear_data_before == nuclear_data_after
        ),
        "model_xml_immutability_pass": (
            model_xml_before == model_xml_after
            and model_contract_before == model_contract_after
        ),
        "log_qualification": parsed,
        "statepoint_summary_qualification": state,
    }
    replica["lost_particle_restart_files"] = sorted(
        path.name for path in output.glob("particle_*.h5")
    )
    if state["statepoint_exists"]:
        replica["statepoint_sha256"] = sha256_file(output / "statepoint.2.h5")
    if state["summary_exists"]:
        replica["summary_sha256"] = sha256_file(output / "summary.h5")
    replica["replica_gate_pass"] = bool(
        identity["pass"]
        and binding_pass
        and replica["transport_input_immutability_pass"]
        and replica["model_xml_immutability_pass"]
        and parsed["pass"]
        and state["readable"]
        and state.get("statepoint_batches") == 2
        and state.get("statepoint_particles") == 2000
        and state.get("statepoint_run_mode") == "fixed source"
        and state.get("statepoint_seed") == int(seed)
        and state.get("statepoint_seed_matches_expected") is True
        and state.get("summary_geometry_present") is True
        and state.get("summary_contract_pass") is True
        and replica["lost_particle_restart_files"] == []
    )
    (output / "replica_qualification.json").write_text(
        json.dumps(replica, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return replica


def run(
    dagmc: Path,
    source_mesh: Path,
    output: Path,
    cross_sections: Path,
    *,
    expected_dagmc_sha256: str,
    expected_source_mesh_sha256: str,
    acceptance_criteria: Path,
    expected_acceptance_criteria_sha256: str,
    source_domain_audit: Path,
    expected_source_domain_audit_sha256: str,
    expected_reference_source_mesh_sha256: str,
    expected_reference_source_fingerprint: str,
    expected_nuclear_data_manifest_sha256: str,
    threads: int = 4,
    timeout_seconds: int = 1800,
) -> dict:
    import openmc

    if output.exists():
        raise FileExistsError(f"create-only output exists: {output}")
    if sha256_file(dagmc) != expected_dagmc_sha256:
        raise ValueError("DAGMC H5M hash mismatch")
    if sha256_file(source_mesh) != expected_source_mesh_sha256:
        raise ValueError("source mesh hash mismatch")
    if sha256_file(acceptance_criteria) != expected_acceptance_criteria_sha256:
        raise ValueError("acceptance-criteria hash mismatch")
    if sha256_file(source_domain_audit) != expected_source_domain_audit_sha256:
        raise ValueError("source-domain receipt hash mismatch")
    initial_nuclear_data = nuclear_data_manifest(cross_sections)
    if (
        initial_nuclear_data["manifest_sha256"]
        != expected_nuclear_data_manifest_sha256
    ):
        raise ValueError("nuclear-data manifest hash mismatch")
    criteria = json.loads(acceptance_criteria.read_text(encoding="utf-8"))
    required_seeds = [
        int(value) for value in criteria["openmc_geometry_debug"]["seeds"]
    ]
    if required_seeds != [20260827, 20260828]:
        raise ValueError("required geometry-debug seeds changed")
    identity = _openmc_identity(timeout_seconds)
    if not identity["pass"]:
        raise RuntimeError("qualified OpenMC 0.16.0 executable is unavailable")
    output.mkdir(parents=True, exist_ok=False)
    replicas = []
    for seed in required_seeds:
        replicas.append(
            _run_replica(
                openmc=openmc,
                dagmc=dagmc,
                source_mesh=source_mesh,
                output=output / f"seed-{seed}",
                cross_sections=cross_sections,
                expected_dagmc_sha256=expected_dagmc_sha256,
                expected_source_mesh_sha256=expected_source_mesh_sha256,
                acceptance_criteria=acceptance_criteria,
                expected_acceptance_criteria_sha256=(
                    expected_acceptance_criteria_sha256
                ),
                source_domain_audit=source_domain_audit,
                expected_source_domain_audit_sha256=(
                    expected_source_domain_audit_sha256
                ),
                expected_reference_source_mesh_sha256=(
                    expected_reference_source_mesh_sha256
                ),
                expected_reference_source_fingerprint=(
                    expected_reference_source_fingerprint
                ),
                expected_nuclear_data_manifest_sha256=(
                    expected_nuclear_data_manifest_sha256
                ),
                seed=seed,
                threads=threads,
                timeout_seconds=timeout_seconds,
                identity=identity,
            )
        )
    signatures = [
        categorical_geometry_debug_signature(row["log_qualification"])
        for row in replicas
    ]
    confirmation = None
    if signatures[0] != signatures[1]:
        confirmation = _run_replica(
            openmc=openmc,
            dagmc=dagmc,
            source_mesh=source_mesh,
            output=output / "one-thread-confirmation-seed-20260827",
            cross_sections=cross_sections,
            expected_dagmc_sha256=expected_dagmc_sha256,
            expected_source_mesh_sha256=expected_source_mesh_sha256,
            acceptance_criteria=acceptance_criteria,
            expected_acceptance_criteria_sha256=(
                expected_acceptance_criteria_sha256
            ),
            source_domain_audit=source_domain_audit,
            expected_source_domain_audit_sha256=(
                expected_source_domain_audit_sha256
            ),
            expected_reference_source_mesh_sha256=(
                expected_reference_source_mesh_sha256
            ),
            expected_reference_source_fingerprint=(
                expected_reference_source_fingerprint
            ),
            expected_nuclear_data_manifest_sha256=(
                expected_nuclear_data_manifest_sha256
            ),
            seed=required_seeds[0],
            threads=1,
            timeout_seconds=timeout_seconds,
            identity=identity,
        )
    aggregate = aggregate_geometry_debug_replicas(
        replicas, required_seeds=required_seeds
    )
    diagnostics_agree = signatures[0] == signatures[1]
    confirmation_signature = (
        categorical_geometry_debug_signature(confirmation["log_qualification"])
        if confirmation is not None
        else None
    )
    confirmation_binding_pass = bool(
        confirmation is not None
        and confirmation.get("seed") == replicas[0].get("seed")
        and confirmation.get("raw_h5m_sha256")
        == replicas[0].get("raw_h5m_sha256")
        and confirmation.get("source_mesh_sha256")
        == replicas[0].get("source_mesh_sha256")
        and confirmation.get("source_domain_audit_sha256")
        == replicas[0].get("source_domain_audit_sha256")
        and confirmation.get("nuclear_data_manifest_sha256")
        == replicas[0].get("nuclear_data_manifest_sha256")
        and confirmation.get("model_xml_seed_normalized_fingerprint")
        == replicas[0].get("model_xml_seed_normalized_fingerprint")
    )
    disagreement_resolution_pass = diagnostics_agree or (
        confirmation is not None
        and confirmation.get("replica_gate_pass") is True
        and confirmation_binding_pass
        and confirmation_signature == signatures[0]
    )
    aggregate["openmc_geometry_debug_gate_pass"] = bool(
        aggregate["openmc_geometry_debug_gate_pass"]
        and disagreement_resolution_pass
    )
    aggregate.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "openmc_executable_identity": identity,
            "replica_diagnostics_agree": diagnostics_agree,
            "categorical_diagnostic_signatures": signatures,
            "confirmation_categorical_signature": confirmation_signature,
            "confirmation_binding_pass": confirmation_binding_pass,
            "disagreement_resolution_pass": disagreement_resolution_pass,
            "one_thread_confirmation": confirmation,
            "cross_sections_path": str(cross_sections),
            "cross_sections_sha256": sha256_file(cross_sections),
            "expected_nuclear_data_manifest_sha256": (
                expected_nuclear_data_manifest_sha256
            ),
        }
    )
    (output / "openmc_geometry_debug_aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dagmc", type=Path)
    parser.add_argument("source_mesh", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cross-sections", required=True, type=Path)
    parser.add_argument("--expected-dagmc-sha256", required=True)
    parser.add_argument("--expected-source-mesh-sha256", required=True)
    parser.add_argument("--acceptance-criteria", required=True, type=Path)
    parser.add_argument("--expected-acceptance-criteria-sha256", required=True)
    parser.add_argument("--source-domain-audit", required=True, type=Path)
    parser.add_argument("--expected-source-domain-audit-sha256", required=True)
    parser.add_argument(
        "--expected-reference-source-mesh-sha256", required=True
    )
    parser.add_argument(
        "--expected-reference-source-fingerprint", required=True
    )
    parser.add_argument(
        "--expected-nuclear-data-manifest-sha256", required=True
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    arguments = parser.parse_args()
    run(
        arguments.dagmc.resolve(),
        arguments.source_mesh.resolve(),
        arguments.output.resolve(),
        arguments.cross_sections.resolve(),
        expected_dagmc_sha256=arguments.expected_dagmc_sha256,
        expected_source_mesh_sha256=arguments.expected_source_mesh_sha256,
        acceptance_criteria=arguments.acceptance_criteria.resolve(),
        expected_acceptance_criteria_sha256=(
            arguments.expected_acceptance_criteria_sha256
        ),
        source_domain_audit=arguments.source_domain_audit.resolve(),
        expected_source_domain_audit_sha256=(
            arguments.expected_source_domain_audit_sha256
        ),
        expected_reference_source_mesh_sha256=(
            arguments.expected_reference_source_mesh_sha256
        ),
        expected_reference_source_fingerprint=(
            arguments.expected_reference_source_fingerprint
        ),
        expected_nuclear_data_manifest_sha256=(
            arguments.expected_nuclear_data_manifest_sha256
        ),
        threads=arguments.threads,
        timeout_seconds=arguments.timeout_seconds,
    )


if __name__ == "__main__":
    main()
