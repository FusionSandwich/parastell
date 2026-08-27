import hashlib
import json
from pathlib import Path

from parastell.reference_geometry import validate_figure_manifest
from parastell.reference_geometry import validate_reference_manifest


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "geometry_parity"
SELECTED = ROOT / "figures" / "geometry_parity" / "selected"
R1_SHA = "8741dd48fded42e8411816e56e3e5e10a29db26ddb785b4a389f8a38b09707a0"
R1_FINGERPRINT = (
    "1c4a6c1fdb37f7bb9d7ef59ab99913884bde6df7b55977a53a68f2a037552bd1"
)


def load(name):
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_live_main_reference_manifest_is_frozen_and_geometry_neutral():
    manifest = load("REFERENCE_GEOMETRY_MANIFEST.json")
    validate_reference_manifest(manifest)

    assert manifest["fork_main_sha"] == manifest["upstream_main_sha"]
    assert (
        manifest["fork_main_sha"] == "de7d2978ff314b060ca2e6b10745a034e8b2a3c4"
    )
    assert manifest["raw_h5m_sha256"] == R1_SHA
    assert manifest["canonical_geometry_fingerprint"] == R1_FINGERPRINT
    assert manifest["ports_present"] is False
    assert manifest["case_thickness_cm"] == 0.0
    assert manifest["magnet_transform"].startswith("none")
    assert manifest["magnet_representation"] == "one_homogenized_solid_each"
    assert manifest["volume_inventory"]["count"] == 24
    assert manifest["surface_inventory"]["count"] == 142
    assert len(manifest["magnet_ids"]) == 18


def test_primary_instrumented_mode_is_byte_identical_without_extra_solids():
    parity = load("REFERENCE_VS_INSTRUMENTED_PARITY.json")

    assert parity["classification"] == "BYTE_IDENTICAL_REUSE"
    assert parity["pass"] is True
    assert parity["same_path"] is True
    assert parity["extra_physical_h5m_volumes"] == 0
    assert parity["h5m_mutation_performed"] is False
    assert parity["reference"]["raw_h5m_sha256"] == R1_SHA
    assert parity["instrumented"]["canonical_fingerprint"] == R1_FINGERPRINT


def test_all_original_homogenized_magnet_envelopes_close():
    inventory = load("ORIGINAL_MAGNET_ENVELOPE_INVENTORY.json")

    assert inventory["magnet_count"] == 18
    assert inventory["all_envelopes_close"] is True
    assert [row["magnet_id"] for row in inventory["magnets"]] == [
        f"magnet-{index:04d}" for index in range(18)
    ]
    assert all(row["closed"] for row in inventory["magnets"])
    assert all(
        row["casing_winding_split"] is False for row in inventory["magnets"]
    )
    assert all(
        row["edge_multiplicity_error_count"] == 0
        for row in inventory["magnets"]
    )


def test_deterministic_reference_ray_audit_passes():
    audit = load("REFERENCE_RAY_AUDIT.json")

    assert audit["magnet_count"] == 18
    assert audit["ray_count"] == 36
    assert audit["all_first_surface_ids_pass"] is True
    assert audit["all_point_in_volume_samples_pass"] is True


def test_native_overlaps_remain_an_explicit_blocking_gate():
    report = load("GEOMETRY_DEBUG_REPORT.json")

    assert report["native_check_watertight"]["pass"] is True
    assert report["native_overlap_check"]["unintended_overlap_count"] == 4
    assert all(
        row["classification"] == "TRUE_UNINTENDED_NONADJACENT_VOLUME_OVERLAP"
        for row in report["native_overlap_check"]["locations"]
    )
    assert report["geometry_gate_pass"] is False
    assert report["openmc_geometry_debug"]["pass"] is False
    assert report["openmc_geometry_debug"]["status"] == (
        "BLOCKED_OPENMC_GEOMETRY_DEBUG_OVERLAPPING_CELLS"
    )
    assert report["openmc_geometry_debug"]["h5m_sha256_after"] == R1_SHA


def test_failed_split_branch_remains_non_production():
    report = load("FAILED_FEATURE_ROOT_CAUSE.json")

    assert report["production_status"] == "NON_PRODUCTION"
    assert len(report["overlap_pairs"]) == 7
    assert report["casing_topology_conclusion"]["classification"] == (
        "E_COMBINATION_B_C_D"
    )
    assert (
        report["casing_topology_conclusion"][
            "direct_winding_to_vacuum_face_count"
        ]
        == 28
    )
    assert "102 and 103" in report["casing_topology_conclusion"]["C"]


def test_matched_camera_figure_manifest_and_selected_hashes():
    manifest = load("FIGURE_MANIFEST.json")
    validate_figure_manifest(manifest)

    rows = {row["filename"]: row for row in manifest["figures"]}
    assert manifest["figure_count"] == 70
    assert manifest["gate_status"] == "BLOCKED_GEOMETRY_PARITY"
    assert len([name for name in rows if "_magnet-" in name]) == 18
    for index in range(1, 13):
        reference = next(
            row
            for name, row in rows.items()
            if name.startswith(f"{index:02d}_")
        )
        instrumented = next(
            row
            for name, row in rows.items()
            if name.startswith(f"{index + 12:02d}_")
        )
        assert reference["camera"] == instrumented["camera"]
        assert reference["color_map"] == instrumented["color_map"]

    selected = sorted(SELECTED.glob("*.png"))
    assert len(selected) >= 12
    for path in selected:
        assert path.name in rows
        assert digest(path) == rows[path.name]["sha256"]


def test_final_gate_fails_closed_before_prompt_7b():
    status = load("FINAL_GATE_STATUS.json")

    assert status["decision"] == "BLOCKED_GEOMETRY_PARITY"
    assert status["gates"]["C_R1_GEOMETRY"].startswith("BLOCKED_")
    assert status["gates"]["G_OPENMC_GEOMETRY_DEBUG"].startswith("BLOCKED_")
    assert status["prompt_7b_authorized"] is False
    assert status["production_transport_authorized"] is False
