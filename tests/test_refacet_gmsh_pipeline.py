from types import SimpleNamespace

import pytest

import parastell.source_cad_refaceting as refacet


class FakeOptions:
    def __init__(self, calls, *, mismatch=None):
        self.calls = calls
        self.values = {}
        self.mismatch = mismatch

    def setNumber(self, name, value):
        self.calls.append(("set", name, value))
        self.values[name] = value

    def getNumber(self, name):
        self.calls.append(("get", name))
        if name == self.mismatch:
            return self.values[name] + 1
        return self.values[name]


class FakeGmsh:
    def __init__(self, calls, *, mismatch=None):
        self.calls = calls
        self.option = FakeOptions(calls, mismatch=mismatch)
        self.finalize_count = 0

    def finalize(self):
        self.calls.append(("finalize",))
        self.finalize_count += 1


def test_thread_contract_freezes_exact_option_sets():
    calls = []
    gmsh = FakeGmsh(calls)
    before = refacet._set_and_read_gmsh_thread_contract(
        gmsh, 4, phase="PRE_IMPORT_FRAGMENT"
    )
    assert before == {"General.NumThreads": 4, "Geometry.OCCParallel": 1}
    mesh = refacet._set_and_read_gmsh_thread_contract(
        gmsh, 32, phase="PRE_MESH"
    )
    assert mesh == {
        "General.NumThreads": 32,
        "Geometry.OCCParallel": 1,
        "Mesh.MaxNumThreads1D": 32,
        "Mesh.MaxNumThreads2D": 32,
        "Mesh.MaxNumThreads3D": 32,
    }


def test_thread_readback_mismatch_fails_closed():
    gmsh = FakeGmsh([], mismatch="General.NumThreads")
    with pytest.raises(RuntimeError, match="readback failed"):
        refacet._set_and_read_gmsh_thread_contract(
            gmsh, 4, phase="PRE_IMPORT_FRAGMENT"
        )


def test_shared_context_sets_threads_before_source_load_and_finalizes(
    monkeypatch, tmp_path
):
    calls = []
    gmsh = FakeGmsh(calls)
    cad = SimpleNamespace(init_gmsh=lambda: gmsh)
    cq = SimpleNamespace(
        Compound=SimpleNamespace(
            makeCompound=lambda solids: calls.append(("compound",)) or object()
        )
    )

    def load(_cq, _path, **_kwargs):
        calls.append(("load_source_solids",))
        return [object()] * 24

    monkeypatch.setattr(refacet, "_load_ordered_source_solids", load)
    monkeypatch.setattr(
        refacet,
        "_cadquery_solid_signature",
        lambda _solid: {
            "mass_cm3": 1.0,
            "center_of_mass_cm": [0.0, 0.0, 0.0],
            "bounding_box_cm": [0.0] * 6,
            "matrix_of_inertia_cm5": [1.0] * 9,
        },
    )
    monkeypatch.setattr(
        refacet, "_valid_volume_signature", lambda _value: True
    )

    def shared(*_args, **_kwargs):
        calls.append(("shared_import_fragment",))
        return {
            "volumes": [(3, index + 1) for index in range(24)],
            "source_import_mapping_evidence": [],
            "volume_mapping_evidence": [],
            "gmsh_topology_contract": {"pass": True},
            "pre_import_thread_options": {},
        }

    monkeypatch.setattr(refacet, "_run_import_fragment_topology", shared)
    with refacet.source_cad_import_fragment_context(
        tmp_path,
        {"pass": True},
        threads=4,
        cq_module=cq,
        cad_to_dagmc_module=cad,
    ):
        calls.append(("caller",))
    first_set = next(
        index for index, row in enumerate(calls) if row[0] == "set"
    )
    load_index = calls.index(("load_source_solids",))
    shared_index = calls.index(("shared_import_fragment",))
    assert first_set < load_index < shared_index
    assert gmsh.finalize_count == 1


def test_shared_context_finalizes_once_on_source_load_error(
    monkeypatch, tmp_path
):
    calls = []
    gmsh = FakeGmsh(calls)
    cad = SimpleNamespace(init_gmsh=lambda: gmsh)
    monkeypatch.setattr(
        refacet,
        "_load_ordered_source_solids",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("load failed")
        ),
    )
    with pytest.raises(RuntimeError, match="load failed"):
        with refacet.source_cad_import_fragment_context(
            tmp_path,
            {"pass": True},
            threads=4,
            cq_module=SimpleNamespace(),
            cad_to_dagmc_module=cad,
        ):
            pass
    assert gmsh.finalize_count == 1


def test_post_sizing_contract_repairs_reset_options():
    calls = []
    gmsh = FakeGmsh(calls)
    refacet._set_and_read_gmsh_thread_contract(
        gmsh, 32, phase="PRE_IMPORT_FRAGMENT"
    )
    gmsh.option.values["General.NumThreads"] = 0
    repaired = refacet._set_and_read_gmsh_thread_contract(
        gmsh, 32, phase="POST_SIZING"
    )
    assert repaired["General.NumThreads"] == 32
    assert repaired["Mesh.MaxNumThreads2D"] == 32
