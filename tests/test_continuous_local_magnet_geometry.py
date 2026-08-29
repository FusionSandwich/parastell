from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import parastell.continuous_local_magnet_geometry as geometry
from parastell.coil_filament_inventory import coordinate_identity


class _Vector:
    def __init__(self, values):
        self.values = values

    def toTuple(self):
        return tuple(self.values)


class _Bounds:
    xmin, ymin, zmin = -1.0, -2.0, -3.0
    xmax, ymax, zmax = 1.0, 2.0, 3.0


class _Shape:
    def __init__(self, name, volume, *, overlap=0.0, error=0.0):
        self.name, self.volume = name, volume
        self.overlap, self.error = overlap, error

    def Volume(self):
        return self.volume

    def isValid(self):
        return True

    def Center(self):
        return _Vector((0.0, 0.0, 0.0))

    def BoundingBox(self):
        return _Bounds()

    def fuse(self, other):
        return _Shape(
            "union", self.volume + other.volume + self.error, error=self.error
        )

    def intersect(self, other):
        return _Shape("intersection", self.overlap)

    def cut(self, other):
        if self.name == "union":
            return _Shape("extra", abs(self.error))
        if other.name == "union":
            return _Shape("missing", abs(other.error))
        return _Shape("difference", 0.0)


class _Imported:
    def __init__(self, shapes):
        self.shapes = shapes

    def solids(self):
        return self

    def vals(self):
        return self.shapes


def _install_fakes(monkeypatch, *, overlap=0.0, error=0.0):
    coords = np.asarray(
        [
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [-100.0, 0.0, 0.0],
            [0.0, -100.0, 0.0],
            [100.0, 0.0, 0.0],
        ]
    )
    exports = {}

    class _MagnetSet:
        def __init__(self, *_args, **kwargs):
            self.case_thickness = kwargs["case_thickness"]
            self.indices = kwargs.get("filament_indices")

        def populate_magnet_coils(self):
            filament = SimpleNamespace(coords=coords)
            self.filaments = [filament]
            self.magnet_coils = [SimpleNamespace(filament=filament)]

        def build_magnet_coils(self):
            if self.case_thickness:
                self.coil_solids = [
                    [
                        _Shape("case", 20.0, overlap=overlap, error=error),
                        _Shape("winding", 80.0),
                    ]
                ]
            else:
                self.coil_solids = [[_Shape("unsplit", 100.0)]]

    class _Exporters:
        @staticmethod
        def export(shape, path):
            exports[path] = shape if isinstance(shape, list) else [shape]
            Path(path).write_bytes(b"step")

    class _Compound:
        @staticmethod
        def makeCompound(shapes):
            return list(shapes)

    class _Importers:
        @staticmethod
        def importStep(path):
            return _Imported(exports[path])

    cq = SimpleNamespace(
        exporters=_Exporters, importers=_Importers, Compound=_Compound
    )
    monkeypatch.setattr(geometry, "_dependencies", lambda: (cq, _MagnetSet))
    return coords


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan(tmp_path, coords):
    coil = tmp_path / "coils"
    coil.write_text("bound", encoding="utf-8")
    case = tmp_path / "case.json"
    case.write_text(json.dumps({"case": 1}), encoding="utf-8")
    bindings = {
        "case": {
            "path": str(case),
            "bytes": case.stat().st_size,
            "sha256": _sha(case),
        }
    }
    return {
        "schema": "parastell.continuous_local_magnet_case_plan/v1.0.0",
        "status": "CONTINUOUS_GLOBAL_AND_LOCAL_COIL_COUPLING_BOUND_NOT_EXECUTED",
        "case_id": "case-1",
        "local_geometry_request": {
            "filament_id": "filament-test",
            "filament_source_ordinal": 0,
            "filament_coordinates_sha256": coordinate_identity(coords)[
                "coordinates_sha256"
            ],
            "coil_input": {
                "path": str(coil),
                "bytes": coil.stat().st_size,
                "sha256": _sha(coil),
            },
            "parser": {"start_line": 3, "scale_to_cm": 100.0},
            "sector": {"start_degrees": 0.0, "extent_degrees": 90.0},
            "outer_width_cm": 40.0,
            "outer_thickness_cm": 50.0,
            "case_thickness_cm": 2.0,
            "inner_width_cm": 36.0,
            "inner_thickness_cm": 46.0,
            "sample_mod": 1,
            "dimension_authority": "user_supplied_local_model",
            "global_volume_parity_claim": False,
        },
        "surface_source": {
            "transform": {
                "rotation_global_to_local": np.eye(3).tolist(),
                "translation_cm": [0.0, 0.0, 0.0],
            }
        },
        "bindings": bindings,
        "claims": {"global_swept_coils": False},
    }


def test_builds_local_split_with_only_same_local_input_parity(
    tmp_path, monkeypatch
):
    coords = _install_fakes(monkeypatch)
    manifest = geometry.build_continuous_local_magnet_geometry(
        _plan(tmp_path, coords), tmp_path / "output"
    )
    assert manifest["same_local_input_unsplit_parity"]["pass"] is True
    assert (
        manifest["global_geometry_comparison"]["volume_parity_claim"] is False
    )
    assert manifest["claims"]["global_swept_coils"] is False
    assert manifest["claims"]["transport_executed"] is False


@pytest.mark.parametrize(
    ("overlap", "error", "match"),
    [(1.0, 0.0, "interiors overlap"), (0.0, 1.0, "same-input unsplit")],
)
def test_local_parity_fails_closed(
    tmp_path, monkeypatch, overlap, error, match
):
    coords = _install_fakes(monkeypatch, overlap=overlap, error=error)
    with pytest.raises(RuntimeError, match=match):
        geometry.build_continuous_local_magnet_geometry(
            _plan(tmp_path, coords), tmp_path / "output"
        )


def test_builder_rejects_unapplied_nonidentity_transform(
    tmp_path, monkeypatch
):
    coords = _install_fakes(monkeypatch)
    plan = _plan(tmp_path, coords)
    plan["surface_source"]["transform"]["translation_cm"] = [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="identity coupling transform"):
        geometry.build_continuous_local_magnet_geometry(
            plan, tmp_path / "output"
        )
