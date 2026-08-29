import pytest

from scripts.prepare_accepted_magnet_inventory import _native_gate_accepts
from scripts.prepare_accepted_magnet_inventory import _validated_magnets


def _gate():
    return {
        "schema": "parastell.dagmc_native_qualification/v1.0.0",
        "native_dagmc_gate_pass": True,
        "raw_h5m_sha256_before": "abc",
        "raw_h5m_sha256_after": "abc",
        "h5m_unchanged": True,
        "native_id_inventory": {"native_id_gate_pass": True},
        "check_watertight": {
            "pass": True,
            "unmatched_edge_count": 0,
            "unsealed_surface_count": 0,
            "unsealed_volume_count": 0,
        },
        "overlap_checks": [
            {
                "points_per_edge": value,
                "pass": True,
                "terminal_overlap_location_count": 0,
                "parsed_overlap_location_count": 0,
            }
            for value in (1, 2, 4)
        ],
    }


def test_accepts_exact_new_native_geometry_gate():
    assert _native_gate_accepts(_gate(), "abc") is True


def test_rejects_new_native_gate_overlap_or_hash_mismatch():
    gate = _gate()
    gate["overlap_checks"][1]["terminal_overlap_location_count"] = 1
    assert _native_gate_accepts(gate, "abc") is False
    gate = _gate()
    assert _native_gate_accepts(gate, "different") is False


def test_rejects_new_native_gate_watertight_failure():
    gate = _gate()
    gate["check_watertight"]["unmatched_edge_count"] = 1
    assert _native_gate_accepts(gate, "abc") is False


def test_rejects_unknown_or_legacy_gate_schema():
    gate = {
        "geometry_gate_pass": True,
        "dagmc": {
            "unchanged": True,
            "sha256_before": "abc",
            "sha256_after": "abc",
        },
    }
    assert _native_gate_accepts(gate, "abc") is False


def _magnets():
    return [
        {
            "magnet": f"magnet-{index:04d}",
            "dagmc_volume_id": index + 9,
            "closed": True,
            "coupling_interface": "homogenized_magnet_outer_boundary",
            "dagmc_surface_ids": [100 + index],
            "centroid_cm": [100.0 + index, 0.0, 0.0],
        }
        for index in range(18)
    ]


def test_accepts_exact_canonical_magnet_mapping():
    assert _validated_magnets({"magnets": _magnets()}) == _magnets()


@pytest.mark.parametrize("defect", ["count", "identity", "surface"])
def test_rejects_spoofed_or_duplicate_magnet_mapping(defect):
    magnets = _magnets()
    if defect == "count":
        magnets.pop()
    elif defect == "identity":
        magnets[1]["magnet"] = "magnet-0000"
    else:
        magnets[1]["dagmc_surface_ids"] = [100]
    with pytest.raises(ValueError, match="exactly 18|mapping is invalid"):
        _validated_magnets({"magnets": magnets})
