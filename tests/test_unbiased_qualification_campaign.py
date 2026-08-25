import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from parastell.magnet_stage_handlers import qualify_production_statistics
from parastell.magnet_stage_handlers import (
    run_unbiased_qualification_campaign,
)


OPENMC_COMMIT = "617d35a5063c57796b43428bc401e627d2011046"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _prepared_model(tmp_path, *, capacity=20):
    model = tmp_path / "prepared"
    model.mkdir()
    xml_files = {}
    for name in (
        "geometry.xml",
        "materials.xml",
        "settings.xml",
        "tallies.xml",
    ):
        path = model / name
        path.write_text(f"<{name}/>", encoding="utf-8")
        xml_files[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    manifest = model / "magnet_openmc_model_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "parastell.magnet_openmc_model/v1.0.0",
                "execution_performed": False,
                "xml_files": xml_files,
                "surface_source": {
                    "max_particles": capacity,
                    "max_source_files": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    return model, manifest


def _install_fake_openmc(
    monkeypatch, *, bank_count=2, navigation_failure=False
):
    exported_seeds = {}
    run_calls = []

    class FakeModel:
        def __init__(self):
            self.settings = SimpleNamespace()

        @classmethod
        def from_xml(cls, **paths):
            assert set(paths) == {
                "geometry",
                "materials",
                "settings",
                "tallies",
            }
            return cls()

        def export_to_xml(self, directory):
            directory = Path(directory)
            exported_seeds[directory.resolve()] = self.settings.seed
            for name in (
                "geometry.xml",
                "materials.xml",
                "settings.xml",
                "tallies.xml",
            ):
                (directory / name).write_text(
                    f"{name}:seed={self.settings.seed}", encoding="utf-8"
                )

    class FakeStatePoint:
        def __init__(self, path, autolink=False):
            with h5py.File(path) as stream:
                self.seed = int(stream["seed"][()])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get_tally(self, *, name):
            assert name == "key_tally"
            offset = (self.seed % 7) * 0.01
            return SimpleNamespace(
                mean=np.asarray([1.0 + offset, 2.0 + offset]),
                std_dev=np.asarray([0.10, 0.20]),
            )

    vector = np.dtype([("x", "f8"), ("y", "f8"), ("z", "f8")])
    bank_dtype = np.dtype(
        [
            ("particle", "i8"),
            ("surf_id", "i8"),
            ("wgt", "f8"),
            ("u", vector),
        ]
    )

    def run(*, cwd, **kwargs):
        directory = Path(cwd).resolve()
        seed = exported_seeds[directory]
        run_calls.append((directory, seed))
        with h5py.File(directory / "statepoint.2.h5", "w") as stream:
            stream.attrs["tallies_present"] = True
            stream.attrs["photon_transport"] = True
            stream.create_dataset("current_batch", data=2)
            stream.create_dataset("n_batches", data=2)
            stream.create_dataset("n_particles", data=5)
            stream.create_dataset("seed", data=seed)
        bank = np.zeros(bank_count, dtype=bank_dtype)
        bank["particle"] = [
            2112 if index % 2 == 0 else 22 for index in range(bank_count)
        ]
        bank["surf_id"] = 11
        bank["wgt"] = np.arange(1, bank_count + 1, dtype=float)
        bank["u"]["x"] = 1.0
        with h5py.File(directory / "surface_source.h5", "w") as stream:
            stream.create_dataset("source_bank", data=bank)
        if navigation_failure:
            print("DAGMC ERROR: could not find the cell")

    fake = SimpleNamespace(
        Model=FakeModel,
        StatePoint=FakeStatePoint,
        run=run,
    )
    monkeypatch.setitem(sys.modules, "openmc", fake)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=(
                "OpenMC version 0.16.0\n" f"Commit hash: {OPENMC_COMMIT}\n"
            )
        ),
    )
    return run_calls


def _stage(tmp_path, model, manifest, *, capacity=20):
    seeds = [101, 103, 107]
    reports = [tmp_path / f"seed-{seed}.json" for seed in seeds]
    return {
        "model_directory": str(model),
        "model_manifest_path": str(manifest),
        "campaign_directory": str(tmp_path / "campaign-runs"),
        "campaign_report_path": str(tmp_path / "campaign.json"),
        "result_report_paths": [str(path) for path in reports],
        "seeds": seeds,
        "minimum_seed_count": 3,
        "particles_per_batch": 5,
        "batches": 2,
        "max_surface_particles": capacity,
        "max_surface_files": 1,
        "expected_openmc_version": "0.16.0",
        "expected_openmc_commit": OPENMC_COMMIT,
        "responses": [
            {
                "response_id": "magnet_flux",
                "kind": "tally",
                "tally_name": "key_tally",
                "particle": "neutron",
                "quantity": "cell_track_length_flux_sum",
                "units": "particle-cm per source",
                "normalization": "per_openmc_source_history",
                "reduction": "sum",
            },
            {
                "response_id": "incoming_photons",
                "kind": "surface_bank_weight",
                "particle": "photon",
                "quantity": "surface_crossing_weight",
                "units": "particle per source",
                "normalization": "per_openmc_source_history",
                "surface_ids": [11],
            },
        ],
    }


def test_unbiased_campaign_runs_three_isolated_seeds_and_reuses_reports(
    tmp_path, monkeypatch
):
    model, manifest = _prepared_model(tmp_path)
    run_calls = _install_fake_openmc(monkeypatch)
    stage = _stage(tmp_path, model, manifest)

    first = run_unbiased_qualification_campaign(stage, tmp_path)
    assert first["status"] == "PASS"
    assert first["transport"] == "UNBIASED_ONLY"
    assert first["seeds"] == [101, 103, 107]
    assert first["histories_per_seed"] == 10
    assert len(run_calls) == 3
    assert len({path for path, _ in run_calls}) == 3
    assert all("attempt-1" in str(path) for path, _ in run_calls)

    for seed, path in zip(stage["seeds"], stage["result_report_paths"]):
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        assert report["seed"] == seed
        assert report["histories"] == 10
        assert {row["response_id"] for row in report["responses"]} == {
            "magnet_flux",
            "incoming_photons",
        }
        tally = next(
            row
            for row in report["responses"]
            if row["response_id"] == "magnet_flux"
        )
        assert tally["within_run_std_dev"] == pytest.approx(0.3)
        assert tally["effective_sample_size"] is None
        surface = next(
            row
            for row in report["responses"]
            if row["response_id"] == "incoming_photons"
        )
        assert surface["raw_count"] == 1
        assert surface["effective_sample_size"] == pytest.approx(1.0)

    second = run_unbiased_qualification_campaign(stage, tmp_path)
    assert second == first
    assert len(run_calls) == 3

    qualification_path = tmp_path / "qualification.json"
    qualification = qualify_production_statistics(
        {
            "result_report_paths": stage["result_report_paths"],
            "campaign_report_path": stage["campaign_report_path"],
            "qualification_path": str(qualification_path),
            "minimum_seed_count": 3,
            "qualification_thresholds": {},
        },
        tmp_path,
    )
    assert qualification["source_report_count"] == 3
    assert qualification["seed_count"] == 3
    assert qualification_path.is_file()


def test_unbiased_campaign_rejects_surface_bank_at_capacity(
    tmp_path, monkeypatch
):
    model, manifest = _prepared_model(tmp_path, capacity=2)
    _install_fake_openmc(monkeypatch, bank_count=2)
    with pytest.raises(
        RuntimeError, match="reached the surface-source capacity"
    ):
        run_unbiased_qualification_campaign(
            _stage(tmp_path, model, manifest, capacity=2), tmp_path
        )


def test_unbiased_campaign_rejects_navigation_diagnostics(
    tmp_path, monkeypatch
):
    model, manifest = _prepared_model(tmp_path)
    _install_fake_openmc(monkeypatch, navigation_failure=True)
    with pytest.raises(RuntimeError, match="DAGMC navigation failure"):
        run_unbiased_qualification_campaign(
            _stage(tmp_path, model, manifest), tmp_path
        )
