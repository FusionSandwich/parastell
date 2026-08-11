"""Command-line workflow for ParaStell magnet spectral handoffs."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
from typing import Any, Sequence

import openmc

from .magnet_spectral_handoff import MagnetSpectralHandoff


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
            source_paths = _fresh_surface_sources(
                output_dir, source_snapshot
            )
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parastell-magnet-handoff",
        description=(
            "Add reactor-to-magnet spectral tallies to an OpenMC model and "
            "export deterministic HTS boundary inputs."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

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
