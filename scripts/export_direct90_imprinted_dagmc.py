"""Create an imprinted DAGMC mesh from the accepted direct 90-degree CAD."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time


REFERENCE_MANIFEST_SHA256 = (
    "b6e723cdb9ac95d789a838abbf44590d210c4fdbe718c3b459777d38768e0499"
)
COMPONENT_ORDER = (
    "chamber",
    "first_wall",
    "breeder",
    "back_wall",
    "hts",
    "vacuum_vessel",
    "lts",
    "gap",
    "magnets",
)
MATERIAL_TAGS = (
    "Vacuum",
    "first_wall",
    "breeder",
    "back_wall",
    "high_temperature_shield",
    "vacuum_vessel",
    "low_temperature_shield",
    "Vacuum",
    "magnet_envelope",
)
STEP_SHA256 = {
    "chamber": "5694bab78ae9ae193af6586fc217f13805c44e317795a1516640c81afc464b5c",
    "first_wall": "b071c8dae65251600bf1966811c5382ebe7b488c4bfd79c8ae5a57da9be8077b",
    "breeder": "9ea4d72b09b7073ad19a62378cbb5aeb51045ae3a0baddd037cdf5602c65b879",
    "back_wall": "b0b18bfe1248b8bb7e756d962d954f98f649ce93fd638197b7717d1a020ffe2c",
    "hts": "51156b53e75777c83feb2fb95244d2e618c39e0437aabea29c9e887141bb39e3",
    "vacuum_vessel": "c6c0d4f9457e37878922b93d478980cdf842b96557e840bb21a8fbf14895a180",
    "lts": "31435e66333c7d3496a5ca19442f73ace1b1bcae6e9f5e44582421b20384f6c4",
    "gap": "ce3ed34deece5730f155116777f7d29117816b54c7ce167def9c0068ea76ceb0",
    "magnets": "d570d8c38b3f68b4b3b097a85f7f07a9b21c5fa4e3db7dbc11816dc9bdca380a",
}
COARSE_MIN_MESH_SIZE_CM = 5.0
COARSE_MAX_MESH_SIZE_CM = 20.0
COARSE_MESH_ALGORITHM = 1
COARSE_THREADS = 32
REQUIRED_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "32",
    "OMP_THREAD_LIMIT": "32",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}
VOLUME_RELATIVE_TOLERANCE = 1.0e-7
VOLUME_ABSOLUTE_TOLERANCE_CM3 = 1.0e-3


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_volumes(manifest: dict) -> dict[str, float]:
    rows = manifest.get("cad_validation", {}).get("components", [])
    result = {}
    for row in rows:
        name = row.get("name")
        volume = float(row.get("volume_cm3", math.nan))
        if (
            name in result
            or name not in COMPONENT_ORDER
            or not math.isfinite(volume)
            or volume <= 0.0
            or row.get("valid") is not True
            or int(row.get("solid_count", 0)) != 1
        ):
            raise ValueError("invalid reference component-volume inventory")
        result[name] = volume
    if tuple(result) != COMPONENT_ORDER:
        raise ValueError(
            "reference component order is not the direct-90 order"
        )
    return result


def _validated_step_artifacts(steps: dict[str, Path]) -> dict[str, dict]:
    """Bind every selected STEP file to the accepted direct-90 byte stream."""
    if tuple(steps) != COMPONENT_ORDER or set(STEP_SHA256) != set(
        COMPONENT_ORDER
    ):
        raise ValueError("STEP component inventory/order is incomplete")
    result = {}
    for name, path in steps.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != STEP_SHA256[name]:
            raise ValueError(f"STEP hash mismatch for {name}")
        result[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
    return result


def _validate_frozen_mesh_controls(
    minimum: float, maximum: float, algorithm: int, threads: int
) -> None:
    if not all(
        math.isfinite(value) and value > 0.0 for value in (minimum, maximum)
    ):
        raise ValueError("mesh sizes must be finite and positive")
    if (
        minimum != COARSE_MIN_MESH_SIZE_CM
        or maximum != COARSE_MAX_MESH_SIZE_CM
        or algorithm != COARSE_MESH_ALGORITHM
        or threads != COARSE_THREADS
    ):
        raise ValueError("mesh controls differ from the frozen coarse attempt")


def _validate_thread_environment() -> None:
    mismatches = {
        name: os.environ.get(name)
        for name, expected in REQUIRED_THREAD_ENVIRONMENT.items()
        if os.environ.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"thread environment is not frozen: {mismatches}")


def _unique_volume_assignment(
    imported_masses: list[float], expected_volumes: dict[str, float]
) -> tuple[list[int], list[dict]]:
    """Return a bijective import ordering using accepted physical volumes."""
    if len(imported_masses) != len(COMPONENT_ORDER):
        raise RuntimeError("Gmsh import changed the component count")
    if not all(
        math.isfinite(value) and value > 0.0 for value in imported_masses
    ):
        raise RuntimeError("Gmsh returned a non-finite or non-positive volume")
    available = set(range(len(imported_masses)))
    ordered_indices = []
    evidence = []
    for name in COMPONENT_ORDER:
        expected = expected_volumes[name]
        matches = sorted(
            index
            for index in available
            if math.isclose(
                imported_masses[index],
                expected,
                rel_tol=VOLUME_RELATIVE_TOLERANCE,
                abs_tol=VOLUME_ABSOLUTE_TOLERANCE_CM3,
            )
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"component {name} has {len(matches)} Gmsh volume matches"
            )
        index = matches[0]
        available.remove(index)
        ordered_indices.append(index)
        evidence.append(
            {
                "component": name,
                "material": MATERIAL_TAGS[COMPONENT_ORDER.index(name)],
                "original_import_index": index,
                "reference_volume_cm3": expected,
                "imported_volume_cm3": imported_masses[index],
                "pass": True,
            }
        )
    if available or len(set(ordered_indices)) != len(COMPONENT_ORDER):
        raise RuntimeError("Gmsh volume assignment is not bijective")
    return ordered_indices, evidence


def _fragment_one_to_one(gmsh, ordered_volumes):
    _, mapping = gmsh.model.occ.fragment(
        ordered_volumes[:1],
        ordered_volumes[1:],
        removeObject=True,
        removeTool=True,
    )
    gmsh.model.occ.synchronize()
    if len(mapping) != len(ordered_volumes):
        raise RuntimeError("Gmsh fragment mapping changed the input count")
    fragmented = []
    for index, row in enumerate(mapping):
        mapped_volumes = [
            (int(dimension), int(tag))
            for dimension, tag in row
            if int(dimension) == 3
        ]
        if len(mapped_volumes) != 1:
            raise RuntimeError(
                f"component {COMPONENT_ORDER[index]} maps to "
                f"{len(mapped_volumes)} fragmented volumes"
            )
        fragmented.append(mapped_volumes[0])
    if len(set(fragmented)) != len(fragmented) or set(fragmented) != {
        (int(dim), int(tag)) for dim, tag in gmsh.model.getEntities(3)
    }:
        raise RuntimeError("fragmented volume inventory is not bijective")
    return fragmented


def export(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"create-only output root exists: {output_root}")
    manifest_path = source_root / "manifest.json"
    if sha256_file(manifest_path) != REFERENCE_MANIFEST_SHA256:
        raise ValueError(
            "source manifest is not the accepted direct-90 reference"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "geometry_validated"
        or manifest.get("model", {}).get("toroidal_extent_degrees") != 90.0
        or manifest.get("model", {}).get("control_grid") != [80, 90]
        or "continuous 30 cm"
        not in manifest.get("model", {}).get("magnet_representation", "")
    ):
        raise ValueError(
            "source manifest is not the accepted direct 90-degree model"
        )
    expected_volumes = _expected_volumes(manifest)
    _validate_frozen_mesh_controls(
        args.min_mesh_size_cm,
        args.max_mesh_size_cm,
        args.algorithm,
        args.threads,
    )
    _validate_thread_environment()

    import cad_to_dagmc
    import cadquery as cq

    steps = {name: source_root / f"{name}.step" for name in COMPONENT_ORDER}
    source_artifacts = _validated_step_artifacts(steps)
    solids = []
    for name, path in steps.items():
        shape = cq.importers.importStep(str(path)).val()
        shape_solids = shape.Solids()
        if len(shape_solids) != 1:
            raise RuntimeError(
                f"{name} STEP contains {len(shape_solids)} solids"
            )
        solids.append(shape_solids[0])
    geometry = cq.Compound.makeCompound(solids)

    output_root.mkdir(parents=True)
    h5m_path = output_root / "dagmc.h5m"
    started = time.time()
    gmsh = cad_to_dagmc.init_gmsh()
    try:
        for option in (
            "General.NumThreads",
            "Mesh.MaxNumThreads1D",
            "Mesh.MaxNumThreads2D",
            "Mesh.MaxNumThreads3D",
        ):
            gmsh.option.setNumber(option, args.threads)
            if int(gmsh.option.getNumber(option)) != args.threads:
                raise RuntimeError(f"Gmsh did not retain {option}")
        _, imported = cad_to_dagmc.get_volumes(
            gmsh, geometry, method="in memory"
        )
        imported = [(int(dim), int(tag)) for dim, tag in imported]
        masses = [
            float(gmsh.model.occ.getMass(dim, tag)) for dim, tag in imported
        ]
        ordered_indices, import_evidence = _unique_volume_assignment(
            masses, expected_volumes
        )
        ordered = [imported[index] for index in ordered_indices]
        fragmented = _fragment_one_to_one(gmsh, ordered)
        fragment_evidence = []
        for index, (dimension, tag) in enumerate(fragmented):
            mass = float(gmsh.model.occ.getMass(dimension, tag))
            expected = expected_volumes[COMPONENT_ORDER[index]]
            passed = math.isclose(
                mass,
                expected,
                rel_tol=VOLUME_RELATIVE_TOLERANCE,
                abs_tol=VOLUME_ABSOLUTE_TOLERANCE_CM3,
            )
            fragment_evidence.append(
                {
                    "component": COMPONENT_ORDER[index],
                    "fragmented_dimtag": [dimension, tag],
                    "reference_volume_cm3": expected,
                    "fragmented_volume_cm3": mass,
                    "pass": passed,
                }
            )
        if not all(row["pass"] for row in fragment_evidence):
            raise RuntimeError(
                "imprinting changed a component physical volume"
            )
        cad_to_dagmc.set_sizes_for_mesh(
            gmsh,
            min_mesh_size=args.min_mesh_size_cm,
            max_mesh_size=args.max_mesh_size_cm,
            mesh_algorithm=args.algorithm,
        )
        gmsh.model.mesh.generate(dim=2)
        vertices, triangles = cad_to_dagmc.mesh_to_vertices_and_triangles(
            fragmented
        )
    finally:
        gmsh.finalize()
    cad_to_dagmc.vertices_to_h5m(
        vertices,
        triangles,
        list(MATERIAL_TAGS),
        h5m_filename=h5m_path,
    )
    receipt = {
        "schema": "wistell_d.direct90_imprinted_dagmc/v1.0.0",
        "status": "IMPRINTED_DAGMC_EXPORTED_NATIVE_GATES_PENDING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "transport_eligible": False,
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": REFERENCE_MANIFEST_SHA256,
        },
        "producer": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "geometry": {
            "construction": "direct ParaStell 0_to_90_degree full field period",
            "combined_from_45_degree_models": False,
            "extent_degrees": 90.0,
            "n_field_periods": 4,
            "component_order": list(COMPONENT_ORDER),
            "material_tags": list(MATERIAL_TAGS),
            "magnet_representation": "continuous_30_cm_magnet_envelope",
        },
        "meshing": {
            "backend": "cad_to_dagmc",
            "imprint": True,
            "min_mesh_size_cm": args.min_mesh_size_cm,
            "max_mesh_size_cm": args.max_mesh_size_cm,
            "algorithm": args.algorithm,
            "threads": args.threads,
            "thread_environment": dict(REQUIRED_THREAD_ENVIRONMENT),
        },
        "source_artifacts": source_artifacts,
        "import_assignment": import_evidence,
        "fragment_assignment": fragment_evidence,
        "h5m": {
            "path": str(h5m_path),
            "bytes": h5m_path.stat().st_size,
            "sha256": sha256_file(h5m_path),
        },
        "elapsed_seconds": time.time() - started,
        "required_successor_gate": "NATIVE_WATERTIGHT_OVERLAP_AND_TOPOLOGY",
    }
    receipt_path = output_root / "DAGMC_EXPORT_RECEIPT.json"
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(receipt, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--min-mesh-size-cm", type=float, default=COARSE_MIN_MESH_SIZE_CM
    )
    parser.add_argument(
        "--max-mesh-size-cm", type=float, default=COARSE_MAX_MESH_SIZE_CM
    )
    parser.add_argument("--algorithm", type=int, default=COARSE_MESH_ALGORITHM)
    parser.add_argument("--threads", type=int, default=COARSE_THREADS)
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
