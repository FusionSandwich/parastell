"""Optional real CadQuery/Gmsh smoke for the source-CAD imprint operation."""

import pytest

from parastell.source_cad_refaceting import _fragment_ordered_volumes
from parastell.source_cad_refaceting import _cadquery_solid_signature
from parastell.source_cad_refaceting import _gmsh_volume_signature
from parastell.source_cad_refaceting import (
    _match_imported_volumes_to_source_solids,
)
from parastell.source_cad_refaceting import _volume_signature_match


def test_gmsh_fragment_imprints_twenty_four_adjacent_solids():
    cadquery = pytest.importorskip("cadquery")
    cad_to_dagmc = pytest.importorskip("cad_to_dagmc")
    solids = [
        cadquery.Workplane("XY")
        .transformed(offset=(float(index), 0.0, 0.0))
        .box(1.0, 1.0, 1.0)
        .val()
        for index in range(24)
    ]
    geometry = cadquery.Compound.makeCompound(solids)
    gmsh = cad_to_dagmc.init_gmsh()
    try:
        _, imported = cad_to_dagmc.get_volumes(
            gmsh, geometry, method="in memory"
        )
        imported = list(imported)
        source_signatures = [
            _cadquery_solid_signature(solid) for solid in solids
        ]
        imported, import_evidence = _match_imported_volumes_to_source_solids(
            gmsh, imported, source_signatures
        )
        assert len(import_evidence) == 24
        assert all(row["signature_pass"] for row in import_evidence)
        ordered = _fragment_ordered_volumes(gmsh, imported)
        assert all(
            _volume_signature_match(
                _cadquery_solid_signature(solid),
                _gmsh_volume_signature(gmsh, volume),
            )
            for solid, volume in zip(solids, ordered)
        )
        assert len(ordered) == 24
        # 24 boxes have 144 faces before imprinting; the 23 shared interfaces
        # become single faces after BooleanFragments.
        assert len(gmsh.model.getEntities(2)) == 121
        boundaries = [
            {
                tag
                for dimension, tag in gmsh.model.getBoundary(
                    [volume], combined=False, oriented=False, recursive=False
                )
                if dimension == 2
            }
            for volume in ordered
        ]
        assert all(
            len(boundaries[index] & boundaries[index + 1]) == 1
            for index in range(23)
        )
    finally:
        gmsh.finalize()
