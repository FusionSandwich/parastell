import hashlib
import json
from types import SimpleNamespace

import cadquery as cq
import pytest

from parastell.magnet_source_cad import (
    SCHEMA,
    export_selected_magnet_source_cad,
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def built_set(*, has_casing=True, two_solids=True):
    inner = cq.Workplane("XY").box(1.0, 1.0, 1.0).val()
    outer = cq.Workplane("XY").box(2.0, 2.0, 2.0).val().cut(inner)
    solids = [outer, inner] if two_solids else [inner]
    return SimpleNamespace(
        has_casing=has_casing,
        coil_solids=[solids],
        magnet_coils=[
            SimpleNamespace(
                coords=[
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            )
        ],
    )


def export(tmp_path, **changes):
    coils = tmp_path / "coils.example"
    coils.write_text("source coils\n", encoding="utf-8")
    arguments = {
        "coil_index": 0,
        "source_filament_index": 8,
        "output_dir": tmp_path / "export",
        "magnet_id": "machine-sector-magnet-8",
        "casing_volume_id": 7,
        "winding_pack_volume_id": 8,
        "coils_path": coils,
        "source_revision": "0123456789abcdef",
        "target_geometry_fingerprint": "a" * 64,
        "global_transform": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "source_parameters": {
            "width_cm": 40.0,
            "thickness_cm": 50.0,
            "case_thickness_cm": 5.0,
            "sample_mod": 6,
        },
        "boundary_signature_options": {
            "linear_tolerance_cm": 0.1,
            "angular_tolerance_rad": 0.1,
        },
    }
    arguments.update(changes)
    return export_selected_magnet_source_cad(
        arguments.pop("magnet_set", built_set()), **arguments
    )


def test_export_is_role_explicit_hash_bound_and_unqualified(tmp_path):
    result = export(tmp_path)
    manifest_path = tmp_path / "export" / result["manifest"]["path"]
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert stored["schema"] == SCHEMA
    assert stored["qualification"]["status"] == (
        "UNQUALIFIED_UNTIL_DAGMC_BOUNDARY_COMPARISON"
    )
    assert stored["source_filament_index"] == 8
    assert stored["target_dagmc"]["role_volume_ids"] == {
        "casing": 7,
        "winding_pack": 8,
    }
    for role in ("casing", "winding_pack"):
        for representation in ("step", "brep"):
            record = stored["artifacts"][role][representation]
            artifact = manifest_path.parent / record["path"]
            assert artifact.is_file()
            assert sha256(artifact) == record["sha256"]
        assert (
            stored["artifacts"][role]["cad_boundary_signature"][
                "measurement_method"
            ]
            == "convergence_qualified_cadquery_closed_boundary_divergence"
        )
    assert sha256(manifest_path) == result["manifest"]["sha256"]


@pytest.mark.parametrize(
    "changes, message",
    [
        (
            {"casing_volume_id": 8, "winding_pack_volume_id": 8},
            "must be distinct",
        ),
        ({"magnet_set": built_set(has_casing=False)}, "requires.*casing"),
        ({"magnet_set": built_set(two_solids=False)}, "must contain"),
        ({"coil_index": 1}, "outside"),
        ({"global_transform": [[1.0]]}, "finite 4x4"),
    ],
)
def test_export_rejects_ambiguous_or_incomplete_identity(
    tmp_path, changes, message
):
    with pytest.raises((ValueError, IndexError), match=message):
        export(tmp_path, **changes)


def test_export_is_no_overwrite_by_default(tmp_path):
    export(tmp_path)
    with pytest.raises(FileExistsError, match="would overwrite"):
        export(tmp_path)
