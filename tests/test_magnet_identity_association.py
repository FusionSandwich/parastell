from types import SimpleNamespace

import numpy as np
import pytest

from parastell.magnet_radiation_field import cad_solid_boundary_signature
from parastell.magnet_radiation_field import filament_associations


class Solid:
    def __init__(self, volume, lower, upper):
        self._volume = volume
        self._bounds = SimpleNamespace(
            xmin=lower[0],
            ymin=lower[1],
            zmin=lower[2],
            xmax=upper[0],
            ymax=upper[1],
            zmax=upper[2],
        )

    def Volume(self):
        return self._volume

    def BoundingBox(self):
        return self._bounds


class TessellatedSolid:
    def __init__(self, connectivity):
        self._connectivity = connectivity

    def Volume(self):
        # Deliberately disagree with the closed boundary.  The mass property is
        # retained as a diagnostic and cannot redefine transport geometry.
        return 1.0 / 3.0

    def tessellate(self, tolerance, angularTolerance):
        vertices = [
            SimpleNamespace(x=0.0, y=0.0, z=0.0),
            SimpleNamespace(x=1.0, y=0.0, z=0.0),
            SimpleNamespace(x=0.0, y=1.0, z=0.0),
            SimpleNamespace(x=0.0, y=0.0, z=1.0),
        ]
        return vertices, self._connectivity


def component(volume_id, volume, lower, upper):
    return SimpleNamespace(
        volume_id=volume_id,
        volume_cm3=volume,
        bounding_box_cm=(lower, upper),
    )


def test_cad_identity_signature_uses_closed_oriented_boundary_volume():
    signature = cad_solid_boundary_signature(
        TessellatedSolid([(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)])
    )

    assert signature["measurement_method"] == (
        "convergence_qualified_cadquery_closed_boundary_divergence"
    )
    assert signature["volume_cm3"] == pytest.approx(1.0 / 6.0)
    assert signature["occ_mass_volume_cm3"] == pytest.approx(1.0 / 3.0)
    assert signature["occ_mass_boundary_relative_difference"] == pytest.approx(
        1.0
    )
    assert signature["triangle_count"] == 4
    assert signature["vector_area_closure_relative"] < 1.0e-15


def test_cad_identity_signature_rejects_inconsistent_face_orientation():
    with pytest.raises(ValueError, match="consistently oriented"):
        cad_solid_boundary_signature(
            TessellatedSolid([(0, 1, 2), (0, 1, 3), (0, 3, 2), (1, 2, 3)])
        )


def test_filament_association_uses_clipped_solid_identity_not_pair_order(
    tmp_path,
):
    coils_path = tmp_path / "coils.dat"
    coils_path.write_text("coils", encoding="utf-8")
    small = [
        Solid(20.0, (0, 0, 0), (1, 1, 1)),
        Solid(80.0, (0.1, 0.1, 0.1), (0.9, 0.9, 0.9)),
    ]
    large = [
        Solid(200.0, (10, 0, 0), (14, 4, 4)),
        Solid(800.0, (10.5, 0.5, 0.5), (13.5, 3.5, 3.5)),
    ]
    magnet_set = SimpleNamespace(
        coil_solids=[small, large],
        magnet_coils=[
            SimpleNamespace(coords=np.zeros((5, 3))),
            SimpleNamespace(coords=np.ones((7, 3))),
        ],
    )
    # Deliberately reverse physical pair order relative to the CAD groups.
    inventory = SimpleNamespace(
        pairs=(
            SimpleNamespace(
                winding_pack=component(
                    2, 800.0, (10.5, 0.5, 0.5), (13.5, 3.5, 3.5)
                ),
                casing=component(1, 200.0, (10, 0, 0), (14, 4, 4)),
            ),
            SimpleNamespace(
                winding_pack=component(
                    4, 80.0, (0.1, 0.1, 0.1), (0.9, 0.9, 0.9)
                ),
                casing=component(3, 20.0, (0, 0, 0), (1, 1, 1)),
            ),
        )
    )
    result = filament_associations(
        inventory,
        magnet_set,
        coils_path=coils_path,
        machine_id="machine",
        sector_id="sector",
        bounding_box_tolerance_cm=0.01,
    )
    assert result[2]["coil_id"] == "coil-0001"
    assert result[4]["coil_id"] == "coil-0000"
    assert result[2]["source_coil_provenance"]["cad_solid_group_index"] == 1
    assert (
        result[4]["source_coil_provenance"]["cad_to_dagmc_identity"]["method"]
        == "global_role_closed_boundary_volume_and_bounding_box_assignment"
    )
