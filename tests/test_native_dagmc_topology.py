import json
import math
import sys
from types import SimpleNamespace

import numpy as np

import pytest

from parastell.native_dagmc_topology import _optional_tag_value
from parastell.native_dagmc_topology import _native_volume_closure
from parastell.native_dagmc_topology import _required_tag
from parastell.native_dagmc_topology import audit_native_moab_topology


class _MissingTagMesh:
    def tag_get_data(self, *args, **kwargs):
        raise RuntimeError("MB_TAG_NOT_FOUND")


class _BrokenMesh:
    def tag_get_data(self, *args, **kwargs):
        raise RuntimeError("corrupt sparse tag storage")


def test_only_explicit_missing_tag_is_treated_as_optional():
    assert _optional_tag_value(_MissingTagMesh(), object(), 1) is None
    with pytest.raises(RuntimeError, match="corrupt sparse"):
        _optional_tag_value(_BrokenMesh(), object(), 1)


def test_optional_tag_uses_pymoab_uint64_entity_handle_array():
    observed = {}

    class Mesh:
        def tag_get_data(self, tag, handles, *, flat):
            observed["tag"] = tag
            observed["handles"] = handles
            observed["flat"] = flat
            return np.asarray([7], dtype=np.int32)

    assert _optional_tag_value(Mesh(), "TAG", 42) == 7
    assert observed["tag"] == "TAG"
    assert observed["handles"].dtype == np.dtype(np.uint64)
    assert observed["handles"].tolist() == [42]
    assert observed["flat"] is True


def test_required_tag_preserves_storage_type_and_never_creates(monkeypatch):
    calls = []

    class Mesh:
        def tag_get_handle(self, *args, **kwargs):
            calls.append((args, kwargs))
            return object()

    monkeypatch.setitem(
        sys.modules,
        "pymoab",
        SimpleNamespace(types=SimpleNamespace(MB_TAG_STORE=0x4000)),
    )
    _required_tag(Mesh(), "CATEGORY", 32, 7, 0x20)

    # MB_TAG_STORE is a query flag, not a storage type.  OR-ing it into the
    # storage_type argument makes real PyMOAB reject the request with
    # MB_TYPE_OUT_OF_RANGE.
    assert calls[0][0][3] == 0x20
    assert calls[0][1]["create_if_missing"] is False


@pytest.mark.parametrize("tolerance", [0.0, -1.0, math.inf, math.nan, True])
def test_invalid_vector_area_tolerance_fails_before_pymoab_import(
    tmp_path, tolerance
):
    candidate = tmp_path / "candidate.h5m"
    candidate.write_bytes(b"tolerance-validation-only")
    with pytest.raises(ValueError, match="finite and positive"):
        audit_native_moab_topology(
            candidate,
            expected_material_counts={},
            vector_area_relative_tolerance=tolerance,
        )


class _FacetMesh:
    def __init__(self, triangles, coordinates):
        self.triangles = triangles
        self.coordinates = coordinates

    def get_entities_by_type(self, surface_handle, entity_type):
        return list(self.triangles)

    def get_connectivity(self, triangle_handle):
        triangle_handle = int(np.asarray(triangle_handle).reshape(-1)[0])
        return self.triangles[triangle_handle]

    def get_coords(self, connectivity):
        return np.asarray(
            [self.coordinates[value] for value in connectivity]
        ).reshape(-1)


def _install_triangle_type(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "pymoab",
        SimpleNamespace(types=SimpleNamespace(MBTRI=2)),
    )


def test_nonfinite_facet_produces_finite_serializable_failure(monkeypatch):
    _install_triangle_type(monkeypatch)
    mesh = _FacetMesh(
        {11: [1, 2, 3]},
        {1: [math.nan, 0, 0], 2: [1, 0, 0], 3: [0, 1, 0]},
    )

    report = _native_volume_closure(
        mesh,
        7,
        [5],
        {5: (7, 0)},
        vector_area_relative_tolerance=1.0e-8,
    )

    assert report["pass"] is False
    assert report["degenerate_triangle_count"] == 1
    assert report["vector_area_closure_relative"] is None
    json.dumps(report, allow_nan=False)


def test_inward_tetrahedron_fails_signed_volume_gate(monkeypatch):
    _install_triangle_type(monkeypatch)
    coordinates = {
        0: [0, 0, 0],
        1: [1, 0, 0],
        2: [0, 1, 0],
        3: [0, 0, 1],
    }
    inward = {
        10: [1, 3, 2],
        11: [0, 2, 3],
        12: [0, 3, 1],
        13: [0, 1, 2],
    }

    report = _native_volume_closure(
        _FacetMesh(inward, coordinates),
        7,
        [5],
        {5: (7, 0)},
        vector_area_relative_tolerance=1.0e-8,
    )

    assert report["native_edge_multiplicity_error_count"] == 0
    assert report["native_directed_edge_error_count"] == 0
    assert report["vector_area_closure_relative"] == pytest.approx(0.0)
    assert report["signed_volume_cm3"] < 0.0
    assert report["pass"] is False


class _FakeCore:
    def __init__(self, *, invalid_sense=False, invalid_group_member=False):
        self.invalid_sense = invalid_sense
        self.invalid_group_member = invalid_group_member
        self.volume = 10
        self.surfaces = [20, 21, 22, 23]
        self.groups = [30, 31]
        self.obb = 99
        self.triangles = {
            20: {100: [1, 2, 3]},
            21: {101: [0, 3, 2]},
            22: {102: [0, 1, 3]},
            23: {103: [0, 2, 1]},
        }
        self.coordinates = {
            0: [0, 0, 0],
            1: [1, 0, 0],
            2: [0, 1, 0],
            3: [0, 0, 1],
        }

    def load_file(self, path):
        return None

    def get_root_set(self):
        return 0

    def tag_get_handle(self, name, *args, **kwargs):
        return name

    def tag_get_data(self, tag, handles, flat=True):
        handle = int(handles[0])
        if tag == "CATEGORY":
            categories = {
                10: b"Volume",
                20: b"Surface",
                21: b"Surface",
                22: b"Surface",
                23: b"Surface",
                30: b"Group",
                31: b"Group",
            }
            if handle not in categories:
                raise RuntimeError("MB_TAG_NOT_FOUND")
            return np.asarray([categories[handle]], dtype=object)
        if tag == "GEOM_DIMENSION":
            dimensions = {
                10: 3,
                20: 2,
                21: 2,
                22: 2,
                23: 2,
                30: -1,
                31: -1,
                99: 0,
            }
            return np.asarray([dimensions[handle]])
        if tag == "GLOBAL_ID":
            ids = {10: 1, 20: 1, 21: 2, 22: 3, 23: 4, 30: 0, 31: 0}
            return np.asarray([ids[handle]])
        if tag == "GEOM_SENSE_2":
            forward = 999 if self.invalid_sense and handle == 20 else 10
            return np.asarray([forward, 0])
        if tag == "NAME":
            names = {30: b"mat:magnets", 31: b"metadata"}
            return np.asarray([names[handle]], dtype=object)
        raise AssertionError(tag)

    def get_entities_by_type(self, handle, entity_type):
        if entity_type == 1:
            return [10, 20, 21, 22, 23, 30, 31, 99] if handle == 0 else []
        if entity_type == 2:
            if handle == 0:
                return [100, 101, 102, 103]
            return list(self.triangles.get(handle, {}))
        raise AssertionError(entity_type)

    def get_parent_meshsets(self, surface):
        return [10]

    def get_child_meshsets(self, volume):
        return self.surfaces if volume == 10 else []

    def get_connectivity(self, triangle):
        triangle = int(np.asarray(triangle).reshape(-1)[0])
        for rows in self.triangles.values():
            if triangle in rows:
                return rows[triangle]
        raise KeyError(triangle)

    def get_coords(self, connectivity):
        return np.asarray(
            [self.coordinates[value] for value in connectivity]
        ).reshape(-1)

    def get_entities_by_handle(self, group):
        if group == 30:
            return [10, 20] if self.invalid_group_member else [10]
        return []


def _install_fake_pymoab(monkeypatch, fake):
    types = SimpleNamespace(
        MB_TAG_STORE=0x4000,
        MB_TAG_SPARSE=0x20,
        MB_TAG_DENSE=0x40,
        MB_TYPE_OPAQUE=1,
        MB_TYPE_INTEGER=2,
        MB_TYPE_HANDLE=3,
        MBENTITYSET=1,
        MBTRI=2,
        CATEGORY_TAG_NAME="CATEGORY",
        CATEGORY_TAG_SIZE=32,
        GEOM_DIMENSION_TAG_NAME="GEOM_DIMENSION",
        GLOBAL_ID_TAG_NAME="GLOBAL_ID",
        NAME_TAG_NAME="NAME",
        NAME_TAG_SIZE=32,
    )
    monkeypatch.setitem(
        sys.modules,
        "pymoab",
        SimpleNamespace(
            core=SimpleNamespace(Core=lambda: fake),
            types=types,
        ),
    )


def test_end_to_end_skips_obb_and_allows_zero_duplicate_group_ids(
    tmp_path, monkeypatch
):
    candidate = tmp_path / "candidate.h5m"
    candidate.write_bytes(b"fake-native-topology")
    _install_fake_pymoab(monkeypatch, _FakeCore())

    report = audit_native_moab_topology(
        candidate,
        expected_material_counts={"magnets": 1},
    )

    assert report["native_topology_gate_pass"] is True
    assert 99 not in [
        row["handle"] for row in report["classified_entity_sets"]
    ]
    assert report["id_errors"] == {}


def test_invalid_sense_handle_fails_without_raising(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate.h5m"
    candidate.write_bytes(b"fake-invalid-sense")
    _install_fake_pymoab(monkeypatch, _FakeCore(invalid_sense=True))

    report = audit_native_moab_topology(
        candidate,
        expected_material_counts={"magnets": 1},
    )

    assert report["native_topology_gate_pass"] is False
    row = next(
        item
        for item in report["surface_senses"]
        if item["surface_handle"] == 20
    )
    assert row["sense_shape_pass"] is False
    assert row["sense_volume_global_ids"] == [None, None]


def test_material_group_direct_nonvolume_member_fails(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate.h5m"
    candidate.write_bytes(b"fake-invalid-material-group")
    _install_fake_pymoab(monkeypatch, _FakeCore(invalid_group_member=True))

    report = audit_native_moab_topology(
        candidate,
        expected_material_counts={"magnets": 1},
    )

    assert report["native_topology_gate_pass"] is False
    material = next(
        row
        for row in report["material_groups"]
        if row["material"] == "magnets"
    )
    assert material["invalid_member_handles"] == [20]
    assert material["pass"] is False
