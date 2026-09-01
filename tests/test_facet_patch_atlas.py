import json

import numpy as np
import pytest

from parastell.facet_patch_atlas import (
    build_conservative_mapping_metadata,
    build_facet_patch_atlas,
    localize_surface_crossings_cached,
    write_facet_patch_atlas_create_only,
)
from parastell.surface_source_localization import localize_surface_crossings


def _catalog():
    return {
        "facet_id": ["surface-7-facet-a", "surface-7-facet-b"],
        "surface_id": [7, 7],
        "vertex_id": np.asarray([[0, 1, 2], [3, 2, 1]]),
        "vertices_global_cm": np.asarray(
            [
                [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                [[1, 1, 0], [0, 1, 0], [1, 0, 0]],
            ],
            dtype=float,
        ),
        "outward_normal_global": np.asarray([[0, 0, 1], [0, 0, 1]]),
        "normal_source": "dagmc_forward_reverse_topology",
    }


def _phase():
    return {
        "position_global_cm": np.asarray([[0.2, 0.2, 0], [0.8, 0.8, 0]]),
        "direction_global": np.asarray([[0, 0, -1], [0, 0, 1]]),
        "surface_id": np.asarray([7, 7]),
        "energy_eV": np.asarray([14.1e6, 1.0e6]),
    }


def test_atlas_caches_right_handed_frames_and_shared_edge_neighbors():
    atlas = build_facet_patch_atlas(_catalog())

    assert atlas.facet_count == 2
    assert atlas.neighbor_basis == "shared_vertex_id"
    assert atlas.neighbors(0).tolist() == [1]
    assert atlas.neighbors(1).tolist() == [0]
    assert np.allclose(atlas.areas_cm2, 0.5)
    assert np.allclose(
        np.cross(
            atlas.frames_global_to_local[:, 0],
            atlas.frames_global_to_local[:, 1],
        ),
        atlas.frames_global_to_local[:, 2],
    )
    assert atlas.compact_metadata()["physical_qualification_claimed"] is False


def test_atlas_frame_and_identity_are_stable_under_cyclic_facet_order():
    baseline = build_facet_patch_atlas(_catalog())
    cycled_catalog = _catalog()
    cycled_catalog["vertices_global_cm"] = np.roll(
        cycled_catalog["vertices_global_cm"], -1, axis=1
    )
    cycled_catalog["vertex_id"] = np.roll(
        cycled_catalog["vertex_id"], -1, axis=1
    )
    cycled = build_facet_patch_atlas(cycled_catalog)

    assert np.allclose(
        baseline.frames_global_to_local, cycled.frames_global_to_local
    )
    assert baseline.atlas_sha256 == cycled.atlas_sha256


def test_cached_localization_matches_existing_uncached_contract():
    atlas = build_facet_patch_atlas(_catalog())
    cached, audit = localize_surface_crossings_cached(_phase(), atlas)
    uncached, _ = localize_surface_crossings(_phase(), _catalog())

    for key in (
        "facet_id",
        "facet_index",
        "barycentric_coordinates",
        "mu",
    ):
        assert (
            np.allclose(cached[key], uncached[key])
            if cached[key].dtype.kind not in "US"
            else np.array_equal(cached[key], uncached[key])
        )
    reconstructed_directions = np.einsum(
        "nji,nj->ni",
        cached["local_frame_global"],
        cached["direction_local"],
    )
    assert np.allclose(reconstructed_directions, _phase()["direction_global"])
    assert audit["cached_frames_used"] is True
    assert (
        audit["candidate_facets_considered"]
        <= audit["full_surface_facets_available"]
    )
    assert audit["physical_qualification_claimed"] is False


def test_conservative_mapping_closes_integrals_and_declares_density_rule():
    atlas = build_facet_patch_atlas(_catalog())
    mapping = build_conservative_mapping_metadata(
        atlas, ["local-patch", "local-patch"]
    )

    assert mapping["conservative_integral_closure_pass"] is True
    assert mapping["patch_count"] == 1
    assert mapping["patch_area_cm2"] == pytest.approx([1.0])
    assert sum(
        mapping["patch_integral_to_facet_integral_weights"]
    ) == pytest.approx(1.0)
    facet_integrals = np.asarray([2.0, 3.0])
    rows = np.asarray(mapping["coo_row_patch_index"])
    patch_total = np.bincount(
        rows,
        weights=facet_integrals
        * np.asarray(mapping["facet_integral_to_patch_integral_weights"]),
    )
    assert patch_total.tolist() == pytest.approx([5.0])
    assert mapping["hidden_renormalization_used"] is False


def test_conservative_mapping_caches_family_and_tape_identity_per_patch():
    atlas = build_facet_patch_atlas(_catalog())
    mapping = build_conservative_mapping_metadata(
        atlas,
        ["patch-a", "patch-b"],
        family_id_by_facet=["family-a", "family-b"],
        tape_id_by_facet=["tape-a", "tape-b"],
    )

    binding = mapping["identity_binding"]
    assert binding["family_ids"] == ["family-a", "family-b"]
    assert binding["tape_ids"] == ["tape-a", "tape-b"]
    assert binding["complete_facet_identity_pass"] is True
    assert [row["patch_id"] for row in binding["patch_bindings"]] == [
        "patch-a",
        "patch-b",
    ]
    assert len(mapping["mapping_sha256"]) == 64


def test_conservative_mapping_rejects_mixed_family_identity_within_patch():
    atlas = build_facet_patch_atlas(_catalog())
    with pytest.raises(ValueError, match="exactly one REBCO family"):
        build_conservative_mapping_metadata(
            atlas,
            ["shared", "shared"],
            family_id_by_facet=["family-a", "family-b"],
            tape_id_by_facet=["tape-a", "tape-b"],
        )


def test_atlas_bundle_is_create_only_and_remains_diagnostic(tmp_path):
    atlas = build_facet_patch_atlas(_catalog())
    mapping = build_conservative_mapping_metadata(atlas)
    output = tmp_path / "atlas"
    identity = write_facet_patch_atlas_create_only(
        output,
        atlas,
        mapping,
        parent_geometry_sha256="a" * 64,
    )

    manifest = json.loads(
        (output / "FACET_PATCH_ATLAS_MANIFEST.json").read_text()
    )
    published_mapping = json.loads(
        (output / "CONSERVATIVE_FACET_PATCH_MAPPING.json").read_text()
    )
    assert len(identity["cache_sha256"]) == 64
    assert published_mapping["mapping_sha256"] == mapping["mapping_sha256"]
    assert (
        manifest["mapping_artifact"]["sha256"] == identity["mapping"]["sha256"]
    )
    assert manifest["transport_eligible"] is False
    assert manifest["physical_qualification_claimed"] is False
    with pytest.raises(FileExistsError):
        write_facet_patch_atlas_create_only(
            output,
            atlas,
            mapping,
            parent_geometry_sha256="a" * 64,
        )


def test_nonmanifold_edge_is_rejected():
    catalog = _catalog()
    catalog["facet_id"] = ["a", "b", "c"]
    catalog["surface_id"] = [7, 7, 7]
    catalog["vertex_id"] = np.asarray([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
    catalog["vertices_global_cm"] = np.asarray(
        [
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[1, 0, 0], [0, 0, 0], [0, -1, 0]],
            [[0, 0, 0], [1, 0, 0], [0, 2, 0]],
        ],
        dtype=float,
    )
    catalog["outward_normal_global"] = np.repeat(
        np.asarray([[0, 0, 1]]), 3, axis=0
    )

    with pytest.raises(ValueError, match="nonmanifold"):
        build_facet_patch_atlas(catalog)
