"""Command-line workflow for ParaStell magnet spectral handoffs."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
from typing import Any, Sequence

import h5py
import numpy as np
import openmc

from .dagmc_envelope import discover_magnet_volumes
from .dagmc_envelope import select_winding_pack_volumes
from .magnet_spectral_handoff import (
    MagnetSpectralHandoff,
    available_energy_group_structures,
    load_energy_group_edges,
)
from .hts_multilayer import (
    constant_response_library,
    replay_phase_space,
    verification_rebco_stack,
    zero_response_library,
)
from .production_handoff import load_and_validate_no_port_configuration


_DEFAULT_MANIFEST = "magnet_spectral_handoff_manifest.json"
_DEFAULT_TALLY_OUTPUT = "magnet_spectra.h5"
_DEFAULT_PHASE_SPACE_OUTPUT = "hts_phase_space_source.h5"


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _load_model(path: str | Path) -> openmc.Model:
    model_path = Path(path).expanduser().resolve()
    if model_path.is_file():
        if model_path.suffix.lower() != ".xml":
            raise ValueError("an OpenMC model file must have an .xml suffix")
        with _working_directory(model_path.parent):
            return openmc.Model.from_model_xml(model_path.name)

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"OpenMC model path does not exist: {model_path}"
        )

    combined = model_path / "model.xml"
    if combined.is_file():
        with _working_directory(model_path):
            return openmc.Model.from_model_xml("model.xml")

    paths = {
        "geometry": model_path / "geometry.xml",
        "materials": model_path / "materials.xml",
        "settings": model_path / "settings.xml",
        "tallies": model_path / "tallies.xml",
        "plots": model_path / "plots.xml",
    }
    missing = [
        str(paths[name])
        for name in ("geometry", "materials", "settings")
        if not paths[name].is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "OpenMC model directory is missing required XML files: "
            + ", ".join(missing)
        )
    relative_paths = {name: item.name for name, item in paths.items()}
    with _working_directory(model_path):
        return openmc.Model.from_xml(**relative_paths)


def _select_region(
    handoff: MagnetSpectralHandoff, requested: str | None
) -> str:
    if requested is not None:
        return handoff.region(requested).name
    if len(handoff.regions) == 1:
        return handoff.regions[0].name
    names = ", ".join(region.name for region in handoff.regions)
    raise ValueError(
        "--region is required when the configuration has multiple regions; "
        f"available regions: {names}"
    )


def _natural_source_key(path: Path) -> tuple[int, int, str]:
    if path.name == "surface_source.h5":
        return (0, 0, path.name)
    match = re.search(r"\.(\d+)\.h5$", path.name)
    return (1, int(match.group(1)) if match else 0, path.name)


def _surface_source_paths(directory: str | Path) -> list[Path]:
    paths = Path(directory).glob("surface_source*.h5")
    return sorted(paths, key=_natural_source_key)


def _file_fingerprint(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size


def _surface_source_snapshot(
    directory: str | Path,
) -> dict[Path, tuple[int, int, int]]:
    return {
        path.resolve(): _file_fingerprint(path)
        for path in _surface_source_paths(directory)
    }


def _fresh_surface_sources(
    directory: str | Path,
    before: dict[Path, tuple[int, int, int]],
) -> list[Path]:
    fresh: list[Path] = []
    for path in _surface_source_paths(directory):
        resolved = path.resolve()
        if before.get(resolved) != _file_fingerprint(path):
            fresh.append(path)
    return fresh


def _contract_summary(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path),
        "kind": "missing",
        "schema": None,
        "schema_version": None,
    }
    with h5py.File(path, "r") as source:
        if "manifest_json" in source:
            summary["kind"] = "spectra"
            summary["schema"] = str(source.attrs.get("schema", ""))
            summary["schema_version"] = str(
                source.attrs.get("schema_version", "")
            )
            manifest = json.loads(source["manifest_json"].asstr()[()])
            summary["interface_status"] = manifest.get("interface_status")
            summary["energy_group_count"] = (
                len(manifest.get("energy_group_edges_eV", [])) - 1
            )
            summary["particles"] = manifest.get("particles", [])
            summary["angular_bin_count"] = (
                len(manifest.get("mu_edges", [])) - 1
            ) * (len(manifest.get("azimuth_edges_rad", [])) - 1)
            summary["coupling_planes"] = [
                plane.get("plane_id")
                for region in manifest.get("regions", [])
                for plane in region.get("coupling_planes", [])
            ]
            return summary
        if "phase_space" in source:
            summary["kind"] = "phase_space"
            summary["schema"] = str(source.attrs.get("schema", ""))
            summary["schema_version"] = str(
                source.attrs.get("schema_version", "")
            )
            phase = source["phase_space"]
            for field in ("position_local_cm", "direction_local"):
                if field in phase:
                    summary[f"count_{field}"] = int(len(phase[field]))
            summary["selection"] = str(source.attrs.get("selection", ""))
            summary["region"] = str(source.attrs.get("region", ""))
            return summary
    return summary


def _inspect_contract(
    spectra_path: Path, phase_space_path: Path | None
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "spectra": _contract_summary(spectra_path),
        "phase_space": None,
    }
    if phase_space_path is not None:
        summary["phase_space"] = _contract_summary(phase_space_path)
    return summary


def _validate_phase_space_contract(args: argparse.Namespace) -> dict[str, Any]:
    handoff = MagnetSpectralHandoff.from_yaml(args.config)
    return handoff.validate_exported_contract(
        args.spectra,
        phase_space_path=args.phase_space,
    )


def _replay_summary(
    args: argparse.Namespace,
    handoff: MagnetSpectralHandoff,
) -> dict[str, Any]:
    if args.response_mode == "zero":
        response = zero_response_library(
            stack=args.stack,
            energy_bounds_eV=handoff.energy_bounds_eV,
            particles=handoff.particles,
        )
    else:
        response = constant_response_library(
            stack=args.stack,
            energy_bounds_eV=handoff.energy_bounds_eV,
            particles=handoff.particles,
            removal_xs_cm_1=args.removal_xs_cm_1,
            deposition_fraction=args.deposition_fraction,
        )
    summary = replay_phase_space(
        phase_space_path=args.phase_space,
        spectra_path=args.spectra,
        output_path=args.output,
        stack=args.stack,
        response_library=response,
        tally_direction=args.tally_direction,
        record_direction=args.record_direction,
        fixed_inward_normal_global=args.fixed_inward_normal,
        allow_local_axis_proxy=args.allow_local_axis_proxy,
        minimum_incident_cosine=args.minimum_incident_cosine,
    )
    return {
        "replay_summary": summary.to_dict(),
        "response_mode": args.response_mode,
        "stack": args.stack.to_dict(),
    }


def _resolve_run_output(path: str | Path, directory: Path) -> Path:
    result = Path(path)
    if result.is_absolute():
        return result
    candidate = directory / result
    return candidate if candidate.exists() else result


def _mpi_arguments(args: argparse.Namespace) -> list[str] | None:
    if args.mpi_processes is None:
        return None
    if args.mpi_processes <= 0:
        raise ValueError("--mpi-processes must be positive")
    return [args.mpi_launcher, "-n", str(args.mpi_processes)]


def _write_result(result: dict[str, Any]) -> None:
    print(json.dumps(result, indent=2, sort_keys=True))


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    handoff = MagnetSpectralHandoff.from_yaml(args.config)
    model = _load_model(args.model)
    handoff.attach_to_model(
        model,
        enable_photon_transport=not args.disable_photon_transport,
    )

    surface_definition: dict[str, Any] = {}
    region_name: str | None = None
    if not args.no_surface_source:
        region_name = _select_region(handoff, args.region)
        surface_config = handoff.configure_surface_source(
            model.settings,
            region_name,
            direction=args.direction,
            max_particles=args.max_particles,
            max_source_files=args.max_source_files,
            overwrite=args.overwrite_surface_source_setting,
        )
        surface_definition = {
            "region": region_name,
            "selection": args.direction,
            "openmc_settings": surface_config,
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.export_to_xml(directory=output_dir)

    result: dict[str, Any] = {
        "output_dir": str(output_dir),
        "prepared_model": str(output_dir),
        "run_performed": bool(args.run),
    }
    statepoint_path: Path | None = None
    source_paths: list[Path] = []
    source_snapshot = _surface_source_snapshot(output_dir)

    if args.run:
        if args.threads is not None and args.threads <= 0:
            raise ValueError("--threads must be positive")
        statepoint = model.run(
            threads=args.threads,
            cwd=output_dir,
            openmc_exec=args.openmc_exec,
            mpi_args=_mpi_arguments(args),
            export_model_xml=False,
        )
        statepoint_path = _resolve_run_output(statepoint, output_dir)
        tally_output = output_dir / args.tally_output
        handoff.export_statepoint(statepoint_path, tally_output)
        result["statepoint"] = str(statepoint_path)
        result["tally_handoff"] = str(tally_output)

        if region_name is not None:
            source_paths = _fresh_surface_sources(output_dir, source_snapshot)
            if not source_paths:
                raise RuntimeError(
                    "OpenMC completed without a new or updated HDF5 "
                    "surface-source file; check surf_source_write settings, "
                    "crossing statistics, and run-directory freshness"
                )
            phase_space_output = output_dir / args.phase_space_output
            handoff.export_surface_source(
                source_paths,
                phase_space_output,
                region_name=region_name,
                selection=args.direction,
            )
            result["surface_source_files"] = [
                str(path) for path in source_paths
            ]
            result["phase_space_handoff"] = str(phase_space_output)
            surface_definition["source_files"] = [
                str(path) for path in source_paths
            ]

    manifest_path = output_dir / args.manifest_output
    handoff.write_manifest(
        manifest_path,
        statepoint_path=statepoint_path,
        surface_source=surface_definition,
    )
    result["manifest"] = str(manifest_path)
    _write_result(result)
    return result


def _postprocess(args: argparse.Namespace) -> dict[str, Any]:
    handoff = MagnetSpectralHandoff.from_yaml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tally_output = output_dir / args.tally_output
    handoff.export_statepoint(args.statepoint, tally_output)
    result: dict[str, Any] = {
        "statepoint": str(args.statepoint),
        "tally_handoff": str(tally_output),
    }

    surface_definition: dict[str, Any] = {}
    if args.surface_source:
        region_name = _select_region(handoff, args.region)
        phase_space_output = output_dir / args.phase_space_output
        handoff.export_surface_source(
            args.surface_source,
            phase_space_output,
            region_name=region_name,
            selection=args.selection,
        )
        surface_definition = {
            "region": region_name,
            "selection": args.selection,
            "source_files": [str(path) for path in args.surface_source],
        }
        result["phase_space_handoff"] = str(phase_space_output)

    manifest_path = output_dir / args.manifest_output
    handoff.write_manifest(
        manifest_path,
        statepoint_path=args.statepoint,
        surface_source=surface_definition,
    )
    result["manifest"] = str(manifest_path)
    _write_result(result)
    return result


def _run(args: argparse.Namespace) -> dict[str, Any]:
    args.run = True
    return _prepare(args)


def _export(args: argparse.Namespace) -> dict[str, Any]:
    return _postprocess(args)


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    result = _validate_phase_space_contract(args)
    _write_result(result)
    return result


def _inspect(args: argparse.Namespace) -> dict[str, Any]:
    result = _inspect_contract(
        spectra_path=args.spectra,
        phase_space_path=args.phase_space,
    )
    _write_result(result)
    return result


def _validate_production_config(args: argparse.Namespace) -> dict[str, Any]:
    _, audit = load_and_validate_no_port_configuration(args.config)
    result = audit.to_dict()
    _write_result(result)
    return result


def _list_magnets(args: argparse.Namespace) -> dict[str, Any]:
    inventory = discover_magnet_volumes(
        args.dagmc,
        winding_pack_materials=args.winding_pack_material,
        casing_materials=args.casing_material,
    )
    result = inventory.to_dict()
    _write_result(result)
    return result


def _inspect_magnet(args: argparse.Namespace) -> dict[str, Any]:
    inventory = discover_magnet_volumes(
        args.dagmc,
        winding_pack_materials=args.winding_pack_material,
        casing_materials=args.casing_material,
    )
    selected = select_winding_pack_volumes(inventory, [args.volume_id])
    result = {
        "dagmc_geometry_sha256": inventory.dagmc_geometry_sha256,
        "magnet": selected[0].to_dict(),
    }
    _write_result(result)
    return result


def _inspect_planes(args: argparse.Namespace) -> dict[str, Any]:
    handoff = MagnetSpectralHandoff.from_yaml(args.config)
    result = {
        "interface_status": handoff.interface_status,
        "planes": [
            plane.to_dict()
            for region in handoff.regions
            for plane in region.coupling_planes
        ],
    }
    _write_result(result)
    return result


def _validate_planes(args: argparse.Namespace) -> dict[str, Any]:
    handoff = MagnetSpectralHandoff.from_yaml(args.config)
    handoff.validate_plane_contract(require_production=args.production)
    result = {
        "valid": True,
        "production": bool(args.production),
        "interface_status": handoff.interface_status,
        "plane_count": sum(
            len(region.coupling_planes) for region in handoff.regions
        ),
    }
    _write_result(result)
    return result


def _list_energy_groups(args: argparse.Namespace) -> dict[str, Any]:
    del args
    result = available_energy_group_structures()
    _write_result(result)
    return result


def _validate_energy_groups(args: argparse.Namespace) -> dict[str, Any]:
    edges = (
        load_energy_group_edges(args.path, units=args.units)
        if args.path is not None
        else np.asarray(args.edges, dtype=float)
        * {"eV": 1.0, "keV": 1.0e3, "MeV": 1.0e6}[args.units]
    )
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("energy-group structure requires at least two edges")
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("energy-group edges must be finite and increasing")
    result = {
        "valid": True,
        "units": "eV",
        "group_count": int(len(edges) - 1),
        "edges_eV": edges.tolist(),
    }
    _write_result(result)
    return result


def _replay(args: argparse.Namespace) -> dict[str, Any]:
    handoff = MagnetSpectralHandoff.from_yaml(args.config)
    stack = verification_rebco_stack()
    output = Path(args.output)
    args.stack = stack
    result = _replay_summary(args, handoff=handoff)
    args.output = output
    _write_result(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parastell-magnet-handoff",
        description=(
            "Add reactor-to-magnet spectral tallies to an OpenMC model and "
            "export deterministic HTS boundary inputs."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    production_config = subparsers.add_parser(
        "validate-production-config",
        help="enforce the port-free production geometry contract",
    )
    production_config.add_argument("--config", required=True, type=Path)
    production_config.set_defaults(handler=_validate_production_config)

    list_magnets = subparsers.add_parser(
        "list-magnets",
        help="list stable winding-pack and casing DAGMC volume IDs",
    )
    list_magnets.add_argument("--dagmc", required=True, type=Path)
    list_magnets.add_argument(
        "--winding-pack-material", action="append", default=["winding_pack"]
    )
    list_magnets.add_argument(
        "--casing-material", action="append", default=["magnet_casing"]
    )
    list_magnets.set_defaults(handler=_list_magnets)

    inspect_magnet = subparsers.add_parser(
        "inspect-magnet",
        help="inspect one explicitly selected winding-pack DAGMC volume",
    )
    inspect_magnet.add_argument("--dagmc", required=True, type=Path)
    inspect_magnet.add_argument("--volume-id", required=True, type=int)
    inspect_magnet.add_argument(
        "--winding-pack-material", action="append", default=["winding_pack"]
    )
    inspect_magnet.add_argument(
        "--casing-material", action="append", default=["magnet_casing"]
    )
    inspect_magnet.set_defaults(handler=_inspect_magnet)

    prepare = subparsers.add_parser(
        "prepare",
        help="augment an OpenMC model and optionally execute it",
    )
    prepare.add_argument("--config", required=True, type=Path)
    prepare.add_argument(
        "--model",
        required=True,
        type=Path,
        help="model.xml or a directory containing OpenMC XML files",
    )
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument("--region")
    prepare.add_argument(
        "--direction",
        choices=("incoming", "outgoing", "both", "all"),
        default="both",
    )
    prepare.add_argument("--max-particles", type=int, default=100_000)
    prepare.add_argument("--max-source-files", type=int, default=1)
    prepare.add_argument("--no-surface-source", action="store_true")
    prepare.add_argument(
        "--overwrite-surface-source-setting",
        action="store_true",
    )
    prepare.add_argument(
        "--disable-photon-transport",
        action="store_true",
    )
    prepare.add_argument("--run", action="store_true")
    prepare.add_argument("--threads", type=int)
    prepare.add_argument("--mpi-processes", type=int)
    prepare.add_argument("--mpi-launcher", default="mpiexec")
    prepare.add_argument("--openmc-exec", default="openmc")
    prepare.add_argument(
        "--manifest-output",
        default=_DEFAULT_MANIFEST,
    )
    prepare.add_argument(
        "--tally-output",
        default=_DEFAULT_TALLY_OUTPUT,
    )
    prepare.add_argument(
        "--phase-space-output",
        default=_DEFAULT_PHASE_SPACE_OUTPUT,
    )
    prepare.set_defaults(handler=_prepare)

    postprocess = subparsers.add_parser(
        "postprocess",
        help="export statepoint and surface-source data after an HPC run",
    )
    postprocess.add_argument("--config", required=True, type=Path)
    postprocess.add_argument("--statepoint", required=True, type=Path)
    postprocess.add_argument("--output-dir", required=True, type=Path)
    postprocess.add_argument(
        "--surface-source",
        nargs="+",
        type=Path,
    )
    postprocess.add_argument("--region")
    postprocess.add_argument(
        "--selection",
        choices=("incoming", "outgoing", "both", "all"),
        default="both",
    )
    postprocess.add_argument(
        "--manifest-output",
        default=_DEFAULT_MANIFEST,
    )
    postprocess.add_argument(
        "--tally-output",
        default=_DEFAULT_TALLY_OUTPUT,
    )
    postprocess.add_argument(
        "--phase-space-output",
        default=_DEFAULT_PHASE_SPACE_OUTPUT,
    )
    postprocess.set_defaults(handler=_postprocess)

    run = subparsers.add_parser(
        "run",
        help="prepare an OpenMC model and execute it",
    )
    run.add_argument("--config", required=True, type=Path)
    run.add_argument(
        "--model",
        required=True,
        type=Path,
        help="model.xml or a directory containing OpenMC XML files",
    )
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--region")
    run.add_argument(
        "--direction",
        choices=("incoming", "outgoing", "both", "all"),
        default="both",
    )
    run.add_argument("--max-particles", type=int, default=100_000)
    run.add_argument("--max-source-files", type=int, default=1)
    run.add_argument("--disable-photon-transport", action="store_true")
    run.add_argument("--threads", type=int)
    run.add_argument("--mpi-processes", type=int)
    run.add_argument("--mpi-launcher", default="mpiexec")
    run.add_argument("--openmc-exec", default="openmc")
    run.add_argument(
        "--tally-output",
        default=_DEFAULT_TALLY_OUTPUT,
    )
    run.add_argument(
        "--phase-space-output",
        default=_DEFAULT_PHASE_SPACE_OUTPUT,
    )
    run.add_argument("--manifest-output", default=_DEFAULT_MANIFEST)
    run.add_argument("--no-surface-source", action="store_true")
    run.set_defaults(
        handler=_run,
        overwrite_surface_source_setting=False,
        no_surface_source=False,
    )

    export = subparsers.add_parser(
        "export",
        help="export handoff files from existing statepoint/source banks",
    )
    export.add_argument("--config", required=True, type=Path)
    export.add_argument("--statepoint", required=True, type=Path)
    export.add_argument("--output-dir", required=True, type=Path)
    export.add_argument(
        "--surface-source",
        nargs="+",
        type=Path,
    )
    export.add_argument("--region")
    export.add_argument(
        "--selection",
        choices=("incoming", "outgoing", "both", "all"),
        default="both",
    )
    export.add_argument("--tally-output", default=_DEFAULT_TALLY_OUTPUT)
    export.add_argument(
        "--phase-space-output", default=_DEFAULT_PHASE_SPACE_OUTPUT
    )
    export.add_argument("--manifest-output", default=_DEFAULT_MANIFEST)
    export.set_defaults(handler=_export)

    validate = subparsers.add_parser(
        "validate",
        help="validate spectra and optional phase-space handoff files",
    )
    validate.add_argument("--config", required=True, type=Path)
    validate.add_argument("--spectra", required=True, type=Path)
    validate.add_argument("--phase-space", type=Path)
    validate.set_defaults(handler=_validate)

    inspect = subparsers.add_parser(
        "inspect",
        help="print a compact summary of handoff files",
    )
    inspect.add_argument("--spectra", required=True, type=Path)
    inspect.add_argument("--phase-space", type=Path)
    inspect.set_defaults(handler=_inspect)

    inspect_planes = subparsers.add_parser(
        "inspect-planes",
        help="print resolved finite magnet-coupling planes",
    )
    inspect_planes.add_argument("--config", required=True, type=Path)
    inspect_planes.set_defaults(handler=_inspect_planes)

    validate_plane = subparsers.add_parser(
        "validate-plane",
        help="validate plane geometry and optional production metadata",
    )
    validate_plane.add_argument("--config", required=True, type=Path)
    validate_plane.add_argument("--production", action="store_true")
    validate_plane.set_defaults(handler=_validate_planes)

    list_groups = subparsers.add_parser(
        "list-energy-groups",
        help="list built-in neutron energy-group structures",
    )
    list_groups.set_defaults(handler=_list_energy_groups)

    validate_groups = subparsers.add_parser(
        "validate-energy-groups",
        help="validate inline or file-based energy-group edges",
    )
    group_input = validate_groups.add_mutually_exclusive_group(required=True)
    group_input.add_argument("--path", type=Path)
    group_input.add_argument("--edges", nargs="+", type=float)
    validate_groups.add_argument(
        "--units", choices=("eV", "keV", "MeV"), default="eV"
    )
    validate_groups.set_defaults(handler=_validate_energy_groups)

    replay = subparsers.add_parser(
        "replay",
        help="replay phase-space through an explicit HTS stack",
    )
    replay.add_argument("--config", required=True, type=Path)
    replay.add_argument("--spectra", required=True, type=Path)
    replay.add_argument("--phase-space", required=True, type=Path)
    replay.add_argument("--output", required=True, type=Path)
    replay.add_argument(
        "--response-mode",
        choices=("zero", "constant"),
        default="constant",
    )
    replay.add_argument("--removal-xs-cm-1", type=float, default=0.0)
    replay.add_argument("--deposition-fraction", type=float, default=0.0)
    replay.add_argument(
        "--tally-direction",
        choices=("incoming", "outgoing", "both"),
        default="incoming",
    )
    replay.add_argument(
        "--record-direction",
        choices=("incoming", "outgoing", "both"),
        default="incoming",
    )
    replay.add_argument(
        "--fixed-inward-normal",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
    )
    replay.add_argument(
        "--allow-local-axis-proxy",
        action="store_true",
    )
    replay.add_argument(
        "--minimum-incident-cosine",
        type=float,
        default=1.0e-8,
    )
    replay.set_defaults(handler=_replay)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
