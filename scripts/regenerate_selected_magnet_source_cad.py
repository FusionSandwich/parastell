"""Regenerate one selected ParaStell filament magnet without approximation."""

from __future__ import annotations

import argparse
import json

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coils", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--target-geometry-fingerprint", required=True)
    parser.add_argument("--magnet-id", required=True)
    parser.add_argument("--source-filament-index", type=int, required=True)
    parser.add_argument("--casing-volume-id", type=int, required=True)
    parser.add_argument("--winding-pack-volume-id", type=int, required=True)
    parser.add_argument("--width-cm", type=float, required=True)
    parser.add_argument("--thickness-cm", type=float, required=True)
    parser.add_argument("--case-thickness-cm", type=float, required=True)
    parser.add_argument("--toroidal-extent-deg", type=float, required=True)
    parser.add_argument("--sample-mod", type=int, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--config-sha256", required=True)
    return parser.parse_args()


def select_original_filament(magnet_set, source_filament_index):
    """Retain one original filament after ParaStell's native sector filter."""
    magnet_set._instantiate_filaments()
    if not 0 <= source_filament_index < len(magnet_set.filaments):
        raise IndexError("source filament index is outside the coil file")
    selected = np.asarray(
        magnet_set.filaments[source_filament_index].coords, dtype=float
    ).copy()
    original_count = len(magnet_set.filaments)
    magnet_set.populate_magnet_coils()
    matches = [
        index
        for index, coil in enumerate(magnet_set.magnet_coils)
        if np.array_equal(np.asarray(coil.coords, dtype=float), selected)
    ]
    if len(matches) != 1:
        raise ValueError(
            "selected source filament is absent or ambiguous after the "
            f"native toroidal filter: matches={matches}"
        )
    filtered_index = matches[0]
    magnet_set.magnet_coils = [magnet_set.magnet_coils[filtered_index]]
    return original_count, filtered_index


def main():
    from parastell.magnet_coils import MagnetSetFromFilaments

    args = parse_args()
    magnet_set = MagnetSetFromFilaments(
        args.coils,
        args.width_cm,
        args.thickness_cm,
        args.toroidal_extent_deg,
        case_thickness=args.case_thickness_cm,
        sample_mod=args.sample_mod,
        scale=args.scale,
        mat_tag=["magnet_casing", "winding_pack"],
    )
    original_count, filtered_index = select_original_filament(
        magnet_set, args.source_filament_index
    )
    # ``populate_magnet_coils`` computed the native domain dimensions from the
    # complete source file before selection.  Building after selection mirrors
    # the production single-coil path while preserving that clipping domain.
    magnet_set.build_magnet_coils()
    result = magnet_set.export_selected_source_cad(
        coil_index=0,
        source_filament_index=args.source_filament_index,
        output_dir=args.output,
        magnet_id=args.magnet_id,
        casing_volume_id=args.casing_volume_id,
        winding_pack_volume_id=args.winding_pack_volume_id,
        coils_path=args.coils,
        source_revision=args.source_revision,
        target_geometry_fingerprint=args.target_geometry_fingerprint,
        global_transform=np.eye(4),
        source_parameters={
            "width_cm": args.width_cm,
            "thickness_cm": args.thickness_cm,
            "case_thickness_cm": args.case_thickness_cm,
            "toroidal_extent_deg": args.toroidal_extent_deg,
            "sample_mod": args.sample_mod,
            "scale_input_to_cm": args.scale,
            "configuration_sha256": args.config_sha256,
            "source_filament_count": original_count,
            "selected_filtered_coil_index": filtered_index,
            "selection_method": (
                "exact_coordinate_identity_after_native_toroidal_filter"
            ),
            "clipping_domain_dimensions_computed_from_complete_coil_file": True,
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
