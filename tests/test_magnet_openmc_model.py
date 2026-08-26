import pytest

from parastell.magnet_openmc_model import _material_binding


def test_dagmc_tag_binds_openmc_id_when_resolved_name_differs():
    manifest = {
        "material_tags": {
            "winding_pack": "WindingPackReference",
            "magnet_casing": "SS316L",
        }
    }
    ids_by_tag = {"winding_pack": 17, "magnet_casing": 23}

    assert _material_binding(manifest, ids_by_tag, "winding_pack") == (
        "WindingPackReference",
        17,
    )


def test_dagmc_tag_binding_fails_closed_for_unconstructed_material():
    manifest = {"material_tags": {"winding_pack": "WindingPackReference"}}

    with pytest.raises(ValueError, match="was not constructed"):
        _material_binding(manifest, {}, "winding_pack")
