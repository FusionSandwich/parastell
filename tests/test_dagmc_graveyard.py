import numpy as np
import pytest

from parastell.dagmc_graveyard import _box_triangles, close_with_graveyard


def _one_sided_box():
    pydagmc = pytest.importorskip("pydagmc")
    types = pytest.importorskip("pymoab.types")
    model = pydagmc.Model()
    volume = model.create_volume(global_id=1)
    surface = model.create_surface(global_id=1)
    triangles = _box_triangles((-1, -1, -1), (1, 1, 1))
    vertices = model.mb.create_vertices(triangles.reshape((-1, 3)).ravel())
    elements = model.mb.create_elements(
        types.MBTRI, np.asarray(vertices).reshape((-1, 3))
    )
    model.mb.add_entities(surface.handle, elements)
    surface.senses = [volume, None]
    material = pydagmc.Group.create(model, name="mat:Vacuum")
    material.add_set(volume)
    return model


def test_post_assembly_graveyard_is_the_only_outer_boundary():
    model = _one_sided_box()
    report = close_with_graveyard(model, margin_cm=5.0)
    assert len(model.volumes) == 3
    assert len(model.surfaces) == 3
    assert report["interstitial_vacuum_volume_id"] == 2
    assert report["graveyard_volume_id"] == 3
    assert report["inner_surface_id"] == 2
    assert report["outer_surface_id"] == 3
    assert [
        surface.id for surface in model.surfaces if None in surface.senses
    ] == [3]
    assert model.volumes_by_id[2] in model.surfaces_by_id[1].senses
    assert model.surfaces_by_id[2].senses == [
        model.volumes_by_id[2],
        model.volumes_by_id[3],
    ]
    assert "mat:Vacuum" in model.group_names
    assert "mat:Graveyard" in model.group_names


def test_post_assembly_graveyard_rejects_duplicate_closure():
    model = _one_sided_box()
    close_with_graveyard(model)
    with pytest.raises(ValueError, match="already contains"):
        close_with_graveyard(model)
