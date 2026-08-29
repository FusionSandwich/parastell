from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

import parastell
import parastell.parametric_geometry as parametric_geometry
from parastell.parametric_geometry import (
    PLAN_SCHEMA,
    SAFETY_POLICY,
    ParametricGeometryError,
    build_source_cad,
    docker_build_argv,
    load_plan,
    resolve_geometry,
    validate_container_attestation,
    validate_vmec_metadata,
)


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _fixture(tmp_path: Path, *, nfp: int = 4) -> tuple[dict, Path, Path]:
    inputs = tmp_path / "inputs"
    vmec_hash = _write(inputs / "device.nc", b"synthetic-vmec")
    coils_hash = _write(inputs / "coils.txt", b"synthetic-coils")
    arrays = np.ones((5, 7), dtype=float)
    arrays[:, -1] = arrays[:, 0]
    array_path = inputs / "layers.npz"
    np.savez(array_path, blanket=arrays * 40.0, gap=arrays * 5.0)
    array_hash = hashlib.sha256(array_path.read_bytes()).hexdigest()
    config = {
        "schema": PLAN_SCHEMA,
        "model_id": "arbitrary-device-full-period",
        "device": "ARBITRARY-DEVICE",
        "inputs": {
            "vmec": {"path": "device.nc", "sha256": vmec_hash},
            "coils": {"path": "coils.txt", "sha256": coils_hash},
            "layers": {"path": "layers.npz", "sha256": array_hash},
        },
        "sector": {
            "mode": "full_periods",
            "period_count": 1,
            "start_degrees": 0.0,
            "n_field_periods": nfp,
            "stellarator_symmetric": True,
            "cad_grid": [5, 7],
        },
        "wall_s": 1.0,
        "chamber_material": "vacuum",
        "layers": [
            {
                "name": "first_wall",
                "material": "steel",
                "thickness": {"kind": "constant", "value_cm": 4.0},
            },
            {
                "name": "blanket",
                "material": "breeder",
                "thickness": {
                    "kind": "array",
                    "input": "layers",
                    "array_key": "blanket",
                },
            },
            {
                "name": "magnet_region",
                "material": "homogenized_magnet",
                "thickness": {"kind": "constant", "value_cm": 30.0},
            },
        ],
        "magnets": {
            "representation": "radial_envelope",
            "layer": "magnet_region",
        },
        "source_mesh": {"enabled": False},
        "runtime": {
            "kind": "docker",
            "image_id": "sha256:" + "a" * 64,
            "network": "none",
            "cpus": 2,
            "memory_bytes": 4294967296,
        },
        "safety": dict(SAFETY_POLICY),
    }
    config_path = tmp_path / "plan.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config, config_path, inputs


@pytest.mark.parametrize("nfp, extent", [(4, 90.0), (5, 72.0)])
def test_generic_full_period_plan_uses_device_nfp(tmp_path, nfp, extent):
    _, config_path, inputs = _fixture(tmp_path, nfp=nfp)
    plan = load_plan(config_path, inputs)
    assert plan.extent_degrees == extent
    assert plan.periodic_boundary_candidate is True
    assert plan.component_order == (
        "chamber",
        "first_wall",
        "blanket",
        "magnet_region",
    )
    assert validate_vmec_metadata(plan, n_field_periods=nfp, lasym=False)[
        "pass"
    ]


def test_resolved_fields_support_constants_and_named_npz_arrays(tmp_path):
    _, config_path, inputs = _fixture(tmp_path)
    resolved = resolve_geometry(load_plan(config_path, inputs))
    assert tuple(resolved.thickness_cm) == (
        "first_wall",
        "blanket",
        "magnet_region",
    )
    assert resolved.thickness_cm["blanket"].shape == (5, 7)
    assert np.all(resolved.thickness_cm["blanket"] == 40.0)
    assert all(
        row["poloidally_closed"] for row in resolved.statistics().values()
    )


def test_plan_rejects_input_substitution(tmp_path):
    _, config_path, inputs = _fixture(tmp_path)
    (inputs / "device.nc").write_bytes(b"substituted")
    with pytest.raises(ParametricGeometryError, match="vmec hash mismatch"):
        load_plan(config_path, inputs)


def test_half_period_requires_stellarator_symmetry(tmp_path):
    config, config_path, inputs = _fixture(tmp_path)
    config["sector"].update(
        {"mode": "stellarator_half_period", "stellarator_symmetric": False}
    )
    config["sector"].pop("period_count")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ParametricGeometryError, match="requires stellarator"):
        load_plan(config_path, inputs)


@pytest.mark.parametrize("wall_s,valid", [(0.999, False), (1.0, True)])
def test_wall_s_matches_parastell_domain(tmp_path, wall_s, valid):
    config, config_path, inputs = _fixture(tmp_path)
    config["wall_s"] = wall_s
    config_path.write_text(json.dumps(config), encoding="utf-8")
    if valid:
        assert load_plan(config_path, inputs).config["wall_s"] == 1.0
    else:
        with pytest.raises(ParametricGeometryError, match="at least 1.0"):
            load_plan(config_path, inputs)


def test_direct_custom_sector_is_nonperiodic(tmp_path):
    config, config_path, inputs = _fixture(tmp_path)
    config["sector"].update({"mode": "direct", "extent_degrees": 37.5})
    config["sector"].pop("period_count")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    plan = load_plan(config_path, inputs)
    assert plan.extent_degrees == 37.5
    assert plan.periodic_boundary_candidate is False


def test_direct_full_torus_requires_toroidal_closure(tmp_path):
    config, config_path, inputs = _fixture(tmp_path)
    config["sector"].update({"mode": "direct", "extent_degrees": 360.0})
    config["sector"].pop("period_count")
    values = np.ones((5, 7))
    values[-1, 2] = 2.0
    array_path = inputs / "layers.npz"
    np.savez(array_path, blanket=values)
    config["inputs"]["layers"]["sha256"] = hashlib.sha256(
        array_path.read_bytes()
    ).hexdigest()
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ParametricGeometryError, match="toroidally periodic"):
        resolve_geometry(load_plan(config_path, inputs))


def test_nonzero_sector_start_is_fail_closed(tmp_path):
    config, config_path, inputs = _fixture(tmp_path)
    config["sector"]["start_degrees"] = 10.0
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ParametricGeometryError, match="nonzero sector starts"):
        load_plan(config_path, inputs)


@pytest.mark.parametrize(
    "key,value",
    [
        ("ports", True),
        ("post_build_transforms", True),
        ("imported_physical_solids", True),
        ("create_only", False),
    ],
)
def test_unsafe_geometry_features_are_schema_invalid(tmp_path, key, value):
    config, config_path, inputs = _fixture(tmp_path)
    config["safety"][key] = value
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ParametricGeometryError, match="forbidden"):
        load_plan(config_path, inputs)


def test_unknown_schema_key_is_rejected(tmp_path):
    config, config_path, inputs = _fixture(tmp_path)
    config["sector"]["period_counts"] = 1
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ParametricGeometryError, match="unsupported keys"):
        load_plan(config_path, inputs)


def test_swept_filaments_and_radial_envelope_are_mutually_exclusive(tmp_path):
    config, config_path, inputs = _fixture(tmp_path)
    config["magnets"] = {
        "representation": "swept_filaments",
        "width_cm": 40.0,
        "thickness_cm": 50.0,
        "material": "homogenized_magnet",
        "sample_mod": 1,
        "layer": "magnet_region",
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ParametricGeometryError, match="unsupported keys"):
        load_plan(config_path, inputs)


def test_swept_filament_plan_supports_file_conventions(tmp_path):
    config, config_path, inputs = _fixture(tmp_path)
    config["layers"].pop()
    config["magnets"] = {
        "representation": "swept_filaments",
        "material": "homogenized_magnet",
        "width_cm": 40.0,
        "thickness_cm": 50.0,
        "case_thickness_cm": 0.0,
        "sample_mod": 3,
        "start_line": 5,
        "scale": 100.0,
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    plan = load_plan(config_path, inputs)
    assert plan.component_order == (
        "chamber",
        "first_wall",
        "blanket",
        "magnets",
    )


def test_swept_filament_plan_requires_explicit_material(tmp_path):
    config, config_path, inputs = _fixture(tmp_path)
    config["layers"].pop()
    config["magnets"] = {
        "representation": "swept_filaments",
        "width_cm": 40.0,
        "thickness_cm": 50.0,
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ParametricGeometryError, match="explicit material"):
        load_plan(config_path, inputs)


@pytest.mark.parametrize("failure", ["shape", "nonpositive", "closure"])
def test_resolved_arrays_fail_closed(tmp_path, failure):
    config, config_path, inputs = _fixture(tmp_path)
    if failure == "shape":
        values = np.ones((4, 7))
    else:
        values = np.ones((5, 7))
        if failure == "nonpositive":
            values[2, 3] = 0.0
        else:
            values[2, -1] = 2.0
    array_path = inputs / "layers.npz"
    np.savez(array_path, blanket=values)
    config["inputs"]["layers"]["sha256"] = hashlib.sha256(
        array_path.read_bytes()
    ).hexdigest()
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ParametricGeometryError):
        resolve_geometry(load_plan(config_path, inputs))


def test_full_period_array_requires_toroidal_closure(tmp_path):
    config, config_path, inputs = _fixture(tmp_path)
    values = np.ones((5, 7))
    values[-1, 3] = 2.0
    array_path = inputs / "layers.npz"
    np.savez(array_path, blanket=values)
    config["inputs"]["layers"]["sha256"] = hashlib.sha256(
        array_path.read_bytes()
    ).hexdigest()
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ParametricGeometryError, match="toroidally periodic"):
        resolve_geometry(load_plan(config_path, inputs))


def test_half_period_array_requires_reflection_symmetric_endpoints(tmp_path):
    config, config_path, inputs = _fixture(tmp_path)
    config["sector"]["mode"] = "stellarator_half_period"
    config["sector"].pop("period_count")
    values = np.ones((5, 7))
    values[0, 1] = 2.0
    array_path = inputs / "layers.npz"
    np.savez(array_path, blanket=values)
    config["inputs"]["layers"]["sha256"] = hashlib.sha256(
        array_path.read_bytes()
    ).hexdigest()
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ParametricGeometryError, match="reflection symmetry"):
        resolve_geometry(load_plan(config_path, inputs))


def test_swept_casing_must_fit_inside_coil(tmp_path):
    config, config_path, inputs = _fixture(tmp_path)
    config["layers"].pop()
    config["magnets"] = {
        "representation": "swept_filaments",
        "material": "magnet",
        "width_cm": 40.0,
        "thickness_cm": 50.0,
        "case_thickness_cm": 20.0,
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ParametricGeometryError, match="casing thickness"):
        load_plan(config_path, inputs)


def test_live_vmec_metadata_must_match_plan(tmp_path):
    _, config_path, inputs = _fixture(tmp_path)
    plan = load_plan(config_path, inputs)
    with pytest.raises(ParametricGeometryError, match="nfp"):
        validate_vmec_metadata(plan, n_field_periods=5, lasym=False)
    with pytest.raises(ParametricGeometryError, match="symmetry"):
        validate_vmec_metadata(plan, n_field_periods=4, lasym=True)


def test_docker_plan_is_networkless_bounded_and_shell_free(
    tmp_path, monkeypatch
):
    _, config_path, inputs = _fixture(tmp_path)
    plan = load_plan(config_path, inputs)
    output = tmp_path / "outputs" / "attempt-001"
    output.parent.mkdir()
    monkeypatch.setattr(
        parametric_geometry,
        "validate_repository_revision",
        lambda repository_root, source_revision: {
            "repository_root": str(repository_root),
            "head": source_revision,
            "clean": True,
        },
    )
    argv = docker_build_argv(
        plan,
        output,
        repository_root=Path(__file__).resolve().parents[1],
        source_revision="b" * 40,
    )
    assert argv[:3] == ["docker", "run", "--rm"]
    assert argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("--memory") + 1] == "4294967296"
    assert argv[argv.index("--memory-swap") + 1] == "4294967296"
    assert "PARASTELL_PARAMETRIC_NETWORK=none" in argv
    assert any(
        item.startswith("PARASTELL_PARAMETRIC_SOURCE_TREE_SHA256=")
        for item in argv
    )
    assert "--execute" not in argv
    with pytest.raises(ParametricGeometryError, match="full Git SHA"):
        docker_build_argv(
            plan,
            tmp_path / "outputs" / "bad-revision",
            repository_root=Path(__file__).resolve().parents[1],
            source_revision="short",
        )
    output.mkdir()
    with pytest.raises(FileExistsError, match="create-only"):
        docker_build_argv(
            plan,
            output,
            repository_root=Path(__file__).resolve().parents[1],
            source_revision="b" * 40,
        )


def test_source_cad_builder_propagates_materials_and_stays_pending(
    tmp_path, monkeypatch
):
    _, config_path, inputs = _fixture(tmp_path)
    resolved = resolve_geometry(load_plan(config_path, inputs))
    calls = {}

    class FakeSolid:
        def Volume(self):
            return 1.0

        def Solids(self):
            return [self]

        def isValid(self):
            return True

    class FakeInVessel:
        Components = {
            "chamber": FakeSolid(),
            "first_wall": FakeSolid(),
            "blanket": FakeSolid(),
            "magnet_region": FakeSolid(),
        }

    class FakeStellarator:
        def __init__(self, vmec_path):
            calls["vmec_path"] = vmec_path

        def construct_invessel_build(self, *args, **kwargs):
            calls["invessel_args"] = args
            calls["invessel_kwargs"] = kwargs
            self.invessel_build = FakeInVessel()

        def export_invessel_build_step(self, export_dir):
            for name in self.invessel_build.Components:
                Path(export_dir, f"{name}.step").write_bytes(b"source-cad")

    fake_module = types.ModuleType("parastell.parastell")
    fake_module.Stellarator = FakeStellarator
    monkeypatch.setitem(sys.modules, "parastell.parastell", fake_module)
    monkeypatch.setattr(parastell, "parastell", fake_module, raising=False)
    monkeypatch.setattr(
        parametric_geometry,
        "read_vmec_metadata",
        lambda plan: {
            "n_field_periods": 4,
            "lasym_vmec_logical": False,
            "stellarator_symmetric": True,
            "pass": True,
        },
    )
    monkeypatch.setattr(
        parametric_geometry,
        "validate_container_attestation",
        lambda *args, **kwargs: {
            "environment": {"test": "qualified"},
            "implementation_files": {
                "parastell/parametric_geometry.py": {
                    "path": "synthetic",
                    "sha256": "d" * 64,
                }
            },
        },
    )

    manifest = build_source_cad(
        resolved,
        tmp_path / "source-cad-attempt",
        source_revision="c" * 40,
        repository_root=Path(__file__).resolve().parents[1],
    )

    assert calls["vmec_path"] == str(resolved.plan.input_files["vmec"])
    assert calls["invessel_kwargs"]["chamber_mat_tag"] == "vacuum"
    assert calls["invessel_kwargs"]["split_chamber"] is False
    assert manifest["status"] == "SOURCE_CAD_BUILT_GEOMETRY_GATES_PENDING"
    assert manifest["transport_eligible"] is False
    assert manifest["container_attestation"]["environment"] == {
        "test": "qualified"
    }
    assert (
        manifest["artifacts"]["first_wall.step"]["sha256"]
        == hashlib.sha256(b"source-cad").hexdigest()
    )


def test_swept_source_cad_inventory_uses_grouped_coils(tmp_path, monkeypatch):
    config, config_path, inputs = _fixture(tmp_path)
    config["layers"].pop()
    config["magnets"] = {
        "representation": "swept_filaments",
        "material": "magnet",
        "width_cm": 40.0,
        "thickness_cm": 50.0,
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    resolved = resolve_geometry(load_plan(config_path, inputs))

    class FakeSolid:
        def Volume(self):
            return 2.0

        def Solids(self):
            return [self]

        def isValid(self):
            return True

    class FakeStellarator:
        def __init__(self, vmec_path):
            pass

        def construct_invessel_build(self, *args, **kwargs):
            self.invessel_build = types.SimpleNamespace(
                Components={
                    "chamber": FakeSolid(),
                    "first_wall": FakeSolid(),
                    "blanket": FakeSolid(),
                }
            )

        def export_invessel_build_step(self, export_dir):
            for name in self.invessel_build.Components:
                Path(export_dir, f"{name}.step").write_bytes(b"cad")

        def construct_magnets_from_filaments(self, *args, **kwargs):
            grouped = [[FakeSolid()], [FakeSolid()]]
            coordinates = np.asarray(
                [
                    [100.0, 0.0, 0.0],
                    [100.0, 1.0, 0.0],
                    [100.0, 1.0, 1.0],
                    [100.0, 0.0, 0.0],
                ]
            )
            self.magnet_set = types.SimpleNamespace(
                coil_solids=grouped,
                all_coil_solids=[
                    solid for group in grouped for solid in group
                ],
                magnet_coils=[
                    types.SimpleNamespace(
                        filament=types.SimpleNamespace(coords=coordinates)
                    )
                    for _ in grouped
                ],
            )

        def export_magnets_step(self, filename, export_dir):
            Path(export_dir, f"{filename}.step").write_bytes(b"magnets")

    fake_module = types.ModuleType("parastell.parastell")
    fake_module.Stellarator = FakeStellarator
    monkeypatch.setitem(sys.modules, "parastell.parastell", fake_module)
    monkeypatch.setattr(parastell, "parastell", fake_module, raising=False)
    monkeypatch.setattr(
        parametric_geometry,
        "read_vmec_metadata",
        lambda plan: {
            "n_field_periods": 4,
            "lasym_vmec_logical": False,
            "stellarator_symmetric": True,
            "pass": True,
        },
    )
    monkeypatch.setattr(
        parametric_geometry,
        "validate_container_attestation",
        lambda *args, **kwargs: {
            "environment": {"test": "qualified"},
            "implementation_files": {},
        },
    )
    manifest = build_source_cad(
        resolved,
        tmp_path / "swept-source-cad",
        source_revision="1" * 40,
        repository_root=Path(__file__).resolve().parents[1],
    )
    assert manifest["magnet_inventory"]["coil_count"] == 2
    assert manifest["magnet_inventory"]["solid_count"] == 2
    assert manifest["magnet_inventory"]["one_homogenized_solid_per_coil"]
    assert len(manifest["magnet_inventory"]["magnets"]) == 2
    assert manifest["magnet_inventory"]["magnets"][0]["magnet_id"] == (
        "magnet-0000"
    )


def test_plan_and_resolved_arrays_are_rechecked_before_build(tmp_path):
    _, config_path, inputs = _fixture(tmp_path)
    resolved = resolve_geometry(load_plan(config_path, inputs))
    assert resolved.thickness_cm["blanket"].flags.writeable is False
    config_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ParametricGeometryError, match="configuration changed"):
        build_source_cad(
            resolved,
            tmp_path / "must-not-exist",
            source_revision="e" * 40,
            repository_root=Path(__file__).resolve().parents[1],
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_input_bindings_are_read_only_and_canonical(tmp_path):
    _, config_path, inputs = _fixture(tmp_path)
    plan = load_plan(config_path, inputs)
    with pytest.raises(TypeError):
        plan.input_files["vmec"] = inputs / "replacement.nc"
    with pytest.raises(TypeError):
        plan.input_hashes["vmec"] = "0" * 64


def test_wistell_profile_preserves_accepted_component_identity():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "wistell_d_parametric_full_period.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert [layer["name"] for layer in config["layers"]] == [
        "first_wall",
        "breeder",
        "back_wall",
        "high_temperature_shield",
        "vacuum_vessel",
        "low_temperature_shield",
        "vacuum_gap",
    ]
    assert config["magnets"] == {
        "representation": "swept_filaments",
        "material": "homogenized_magnet",
        "width_cm": 30.0,
        "thickness_cm": 30.0,
        "case_thickness_cm": 0.0,
        "sample_mod": 1,
        "start_line": 3,
        "scale": 100.0,
    }
    assert config["construction"] == {
        "radial_build_mode": "isolated_cumulative_shells"
    }


def test_isolated_shells_preserve_exact_cumulative_boundaries(
    tmp_path, monkeypatch
):
    _, config_path, inputs = _fixture(tmp_path)
    resolved = resolve_geometry(load_plan(config_path, inputs))
    calls = []

    class FakeSolid:
        def Volume(self):
            return 1.0

        def Solids(self):
            return [self]

        def isValid(self):
            return True

    class FakeStellarator:
        def __init__(self, vmec_path):
            self.vmec_path = vmec_path

        def construct_invessel_build(self, *args, **kwargs):
            radial_build = args[3]
            calls.append(
                {
                    name: np.array(row["thickness_matrix"], copy=True)
                    for name, row in radial_build.items()
                }
            )
            self.invessel_build = types.SimpleNamespace(
                Components={
                    name: FakeSolid()
                    for name in ("chamber", *radial_build.keys())
                }
            )

    fake_module = types.SimpleNamespace(Stellarator=FakeStellarator)
    monkeypatch.setattr(
        parametric_geometry,
        "_export_step_solid",
        lambda solid, path: path.write_bytes(b"isolated-shell"),
    )
    evidence, stages = (
        parametric_geometry._construct_isolated_cumulative_shells(
            ps=fake_module,
            plan=resolved.plan,
            resolved=resolved,
            config=resolved.plan.config,
            output_root=tmp_path,
        )
    )
    layer_names = [row["name"] for row in resolved.plan.config["layers"]]
    assert len(calls) == len(layer_names)
    assert tuple(calls[0]) == (layer_names[0],)
    cumulative = np.zeros(resolved.plan.grid_shape)
    for index, name in enumerate(layer_names):
        if index:
            assert np.array_equal(
                calls[index]["__cumulative_interior__"], cumulative
            )
        assert np.array_equal(calls[index][name], resolved.thickness_cm[name])
        cumulative += resolved.thickness_cm[name]
    assert set(evidence) == {"chamber", *layer_names}
    assert [row["component"] for row in stages] == layer_names
    assert (tmp_path / "chamber.step").is_file()
    assert all((tmp_path / f"{name}.step").is_file() for name in layer_names)


def test_direct_host_source_cad_build_is_rejected_before_output(tmp_path):
    _, config_path, inputs = _fixture(tmp_path)
    resolved = resolve_geometry(load_plan(config_path, inputs))
    output = tmp_path / "host-build-must-not-exist"
    with pytest.raises(ParametricGeometryError, match="Docker|attestation"):
        build_source_cad(
            resolved,
            output,
            source_revision="f" * 40,
            repository_root=Path(__file__).resolve().parents[1],
        )
    assert not output.exists()


def test_container_attestation_binds_runtime_revision_and_code(
    tmp_path,
):
    _, config_path, inputs = _fixture(tmp_path)
    plan = load_plan(config_path, inputs)
    marker = tmp_path / ".dockerenv"
    marker.touch()
    repository_root = tmp_path / "source-tree"
    _write(repository_root / "parastell" / "parametric_geometry.py", b"one")
    core_path = repository_root / "parastell" / "invessel_build.py"
    _write(core_path, b"original-core")
    _write(
        repository_root / "scripts" / "build_parametric_geometry.py",
        b"builder",
    )
    source_tree_sha256 = parametric_geometry._source_tree_sha256(
        parametric_geometry._source_tree_rows(repository_root)
    )
    environment = {
        "PARASTELL_PARAMETRIC_ATTESTATION": (
            parametric_geometry.CONTAINER_ATTESTATION
        ),
        "PARASTELL_PARAMETRIC_IMAGE_ID": plan.config["runtime"]["image_id"],
        "PARASTELL_PARAMETRIC_NETWORK": "none",
        "PARASTELL_PARAMETRIC_SOURCE_REVISION": "a" * 40,
        "PARASTELL_PARAMETRIC_SOURCE_TREE_SHA256": source_tree_sha256,
    }
    receipt = validate_container_attestation(
        plan,
        "a" * 40,
        repository_root=repository_root,
        container_marker=marker,
        environ=environment,
    )
    assert receipt["environment"]["PARASTELL_PARAMETRIC_NETWORK"] == "none"
    core_path.write_bytes(b"changed-after-command-generation")
    with pytest.raises(ParametricGeometryError, match="source tree differs"):
        validate_container_attestation(
            plan,
            "a" * 40,
            repository_root=repository_root,
            container_marker=marker,
            environ=environment,
        )
    core_path.write_bytes(b"original-core")
    environment["PARASTELL_PARAMETRIC_SOURCE_REVISION"] = "b" * 40
    with pytest.raises(ParametricGeometryError, match="SOURCE_REVISION"):
        validate_container_attestation(
            plan,
            "a" * 40,
            repository_root=repository_root,
            container_marker=marker,
            environ=environment,
        )
