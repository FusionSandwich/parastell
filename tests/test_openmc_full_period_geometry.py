import hashlib
import sys
from types import SimpleNamespace

import pytest

import parastell.reference_geometry as reference_geometry_module
from parastell.reference_geometry import ReferenceGeometry


class _Region:
    def __init__(self, expression):
        self.expression = expression

    def __and__(self, other):
        return _Region(("and", self.expression, other.expression))


class _Surface:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.id = kwargs["surface_id"]
        self.periodic_surface = None

    def __pos__(self):
        return _Region(("+", self.id))

    def __neg__(self):
        return _Region(("-", self.id))


class _OpenMC:
    __version__ = "0.16.0"

    class YPlane(_Surface):
        pass

    class Plane(_Surface):
        pass

    class Sphere(_Surface):
        pass

    class DAGMCUniverse:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Cell:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Universe:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Geometry:
        def __init__(self, root):
            self.root = root


def _reference(tmp_path):
    path = tmp_path / "dagmc.h5m"
    path.write_bytes(b"immutable-test-h5m")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return ReferenceGeometry(path=path, accepted_sha256=digest)


def test_one_period_geometry_uses_stock_openmc_rotational_periodicity(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(sys.modules, "openmc", _OpenMC)
    monkeypatch.setattr(
        reference_geometry_module,
        "native_dagmc_id_inventory",
        lambda _path: {"native_id_gate_pass": True, "maximum_native_id": 100},
    )

    geometry = _reference(tmp_path).openmc_one_period_geometry(
        n_field_periods=4,
        external_vacuum_radius_cm=2000.0,
    )

    cell = geometry.root.kwargs["cells"][0]
    dagmc = cell.kwargs["fill"]
    assert dagmc.kwargs["auto_geom_ids"] is False
    assert dagmc.kwargs["universe_id"] == 106
    assert cell.kwargs["cell_id"] == 104
    assert cell.kwargs["region"].expression == (
        "and",
        ("and", ("-", 103), ("+", 101)),
        ("+", 102),
    )


def test_one_period_geometry_rejects_non_openmc_016(tmp_path, monkeypatch):
    wrong = SimpleNamespace(__version__="0.15.2")
    monkeypatch.setitem(sys.modules, "openmc", wrong)

    with pytest.raises(RuntimeError, match="expected 0.16.0"):
        _reference(tmp_path).openmc_one_period_geometry(
            n_field_periods=4,
            external_vacuum_radius_cm=2000.0,
        )
