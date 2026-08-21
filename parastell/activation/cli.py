"""Command-line tools for activation environment and interchange audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backends import detect_activation_backends, select_activation_backend
from .chain_audit import audit_activation_chain
from .openmc_r2s import openmc_r2s_capability_report
from .spectrum_export import ActivationSpectrum, load_activation_spectrum


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _audit_environment(args) -> int:
    capabilities = detect_activation_backends(
        openmc_report=openmc_r2s_capability_report(),
        alara_executable=args.alara,
        alara_data=tuple(args.alara_data),
        fispact_executable=args.fispact,
        fispact_data=tuple(args.fispact_data),
    )
    payload = {
        item.value: value.as_dict() for item, value in capabilities.items()
    }
    try:
        payload["selection"] = select_activation_backend(capabilities)
    except RuntimeError as error:
        payload["selection_error"] = str(error)
    _print(payload)
    return 0


def _audit_chain(args) -> int:
    report = audit_activation_chain(
        args.chain,
        args.cross_sections,
        args.nuclide,
        chain_release=args.chain_release,
        transport_release=args.transport_release,
        allow_release_mismatch=args.allow_release_mismatch,
    )
    if args.output:
        report.write(args.output)
    _print(report.as_dict())
    return 0 if report.passes else 2


def _export_spectrum(args) -> int:
    data = json.loads(Path(args.input).read_text(encoding="ascii"))
    spectrum = ActivationSpectrum.create(
        data["name"],
        data.get("particle", "neutron"),
        data["edges_eV"],
        data["group_flux_cm2_s"],
        data["region_id"],
        data["reference_source_rate_n_s"],
        data.get("source"),
    )
    if args.format == "json":
        spectrum.write_json(args.output)
    elif args.format == "alara":
        spectrum.write_alara_flux(args.output, ordering=args.ordering)
    else:
        spectrum.write_fispact_arb_flux(args.output)
    _print(spectrum.as_dict())
    return 0


def _inspect_spectrum(args) -> int:
    _print(load_activation_spectrum(args.path).as_dict())
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parastell activation",
        description="Audit and prepare ParaStell activation workflows.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    environment = sub.add_parser("audit-environment")
    environment.add_argument("--alara")
    environment.add_argument("--alara-data", action="append", default=[])
    environment.add_argument("--fispact")
    environment.add_argument("--fispact-data", action="append", default=[])
    environment.set_defaults(function=_audit_environment)

    chain = sub.add_parser("audit-chain")
    chain.add_argument("--chain", required=True)
    chain.add_argument("--cross-sections", required=True)
    chain.add_argument("--nuclide", action="append", required=True)
    chain.add_argument("--chain-release")
    chain.add_argument("--transport-release")
    chain.add_argument("--allow-release-mismatch", action="store_true")
    chain.add_argument("--output")
    chain.set_defaults(function=_audit_chain)

    export = sub.add_parser("export-spectrum")
    export.add_argument("--input", required=True)
    export.add_argument("--output", required=True)
    export.add_argument(
        "--format", choices=("json", "alara", "fispact-arb"), required=True
    )
    export.add_argument(
        "--ordering", choices=("ascending", "descending"), default="descending"
    )
    export.set_defaults(function=_export_spectrum)

    inspect = sub.add_parser("inspect-spectrum")
    inspect.add_argument("path")
    inspect.set_defaults(function=_inspect_spectrum)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
