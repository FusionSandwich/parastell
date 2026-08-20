from pathlib import Path

import pytest

from parastell.magnet_handoff_cli import _load_model
from parastell.magnet_handoff_cli import _natural_source_key
from parastell.magnet_handoff_cli import _fresh_surface_sources
from parastell.magnet_handoff_cli import _resolve_run_output
from parastell.magnet_handoff_cli import _select_region
from parastell.magnet_handoff_cli import _surface_source_snapshot
from parastell.magnet_handoff_cli import build_parser
from parastell.magnet_spectral_handoff import MagnetSpectralHandoff


def _mapping(region_count=1):
    regions = []
    for index in range(region_count):
        regions.append(
            {
                "name": f"coil {index}",
                "cell_ids": [10 + index],
                "surface_ids": [20 + index],
            }
        )
    return {
        "energy_bounds_eV": [0.0, 1.0e6, 2.0e7],
        "regions": regions,
    }


def test_prepare_parser_defaults_to_bidirectional_bank():
    args = build_parser().parse_args(
        [
            "prepare",
            "--config",
            "handoff.yaml",
            "--model",
            "model.xml",
            "--output-dir",
            "run",
        ]
    )

    assert args.direction == "both"
    assert args.max_particles == 100_000
    assert args.run is False


def test_selects_only_region_implicitly():
    handoff = MagnetSpectralHandoff.from_mapping(_mapping())

    assert _select_region(handoff, None) == "coil 0"


def test_requires_region_for_multiple_interfaces():
    handoff = MagnetSpectralHandoff.from_mapping(_mapping(region_count=2))

    with pytest.raises(ValueError, match="--region is required"):
        _select_region(handoff, None)


def test_surface_source_files_sort_numerically():
    paths = [
        Path("surface_source.10.h5"),
        Path("surface_source.2.h5"),
        Path("surface_source.h5"),
    ]

    assert [path.name for path in sorted(paths, key=_natural_source_key)] == [
        "surface_source.h5",
        "surface_source.2.h5",
        "surface_source.10.h5",
    ]


def test_fresh_surface_sources_excludes_stale_files(tmp_path):
    stale = tmp_path / "surface_source.h5"
    changed = tmp_path / "surface_source.2.h5"
    stale.write_bytes(b"old")
    changed.write_bytes(b"before")

    snapshot = _surface_source_snapshot(tmp_path)
    changed.write_bytes(b"after-and-larger")
    new_file = tmp_path / "surface_source.3.h5"
    new_file.write_bytes(b"new")

    fresh = _fresh_surface_sources(tmp_path, snapshot)

    assert [path.name for path in fresh] == [
        "surface_source.2.h5",
        "surface_source.3.h5",
    ]


def test_load_model_resolves_relative_files_from_model_directory(
    tmp_path, monkeypatch
):
    model_dir = tmp_path / "reactor"
    model_dir.mkdir()
    (model_dir / "model.xml").write_text("<model/>", encoding="utf-8")
    observed = {}

    def fake_from_model_xml(path):
        observed["cwd"] = Path.cwd()
        observed["path"] = Path(path)
        return object()

    monkeypatch.setattr(
        "parastell.magnet_handoff_cli.openmc.Model.from_model_xml",
        fake_from_model_xml,
        raising=False,
    )
    original = Path.cwd()

    result = _load_model(model_dir)

    assert result is not None
    assert observed == {"cwd": model_dir.resolve(), "path": Path("model.xml")}
    assert Path.cwd() == original


def test_resolve_run_output_prefers_run_directory(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    candidate = run_dir / "statepoint.10.h5"
    candidate.write_bytes(b"new")
    stale = tmp_path / "statepoint.10.h5"
    stale.write_bytes(b"stale")
    monkeypatch.chdir(tmp_path)

    resolved = _resolve_run_output("statepoint.10.h5", run_dir)

    assert resolved == candidate


def test_plane_and_energy_group_subcommands_are_public():
    parser = build_parser()

    plane_args = parser.parse_args(
        ["validate-plane", "--config", "handoff.yaml", "--production"]
    )
    group_args = parser.parse_args(
        [
            "validate-energy-groups",
            "--edges",
            "0",
            "13.5",
            "14.1",
            "20",
            "--units",
            "MeV",
        ]
    )

    assert plane_args.production is True
    assert group_args.edges == [0.0, 13.5, 14.1, 20.0]
    assert parser.parse_args(["list-energy-groups"]).command == (
        "list-energy-groups"
    )
