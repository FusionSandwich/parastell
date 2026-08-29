import json
from pathlib import Path

from parastell.transport_response_plan import build_response_plan


def test_direct90_smoke_config_declares_all_homogenized_magnets():
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "wistell_d_parametric_direct90_full_response_smoke.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["magnet_ids"] == [
        f"magnet-{index:04d}" for index in range(18)
    ]
    assert value["nuclide_mt_requests"] == {"Cu63": [2, 16, 102, 103, 107]}
    plan = build_response_plan(
        case_id=value["case_id"],
        magnet_ids=value["magnet_ids"],
        neutron_energy_edges_eV=value["energy_axes_eV"]["neutron"],
        photon_energy_edges_eV=value["energy_axes_eV"]["photon"],
        nuclide_mt_requests=value["nuclide_mt_requests"],
    )
    assert plan["proof_level"] == "DECLARED"
    assert plan["normalization"] == "per_source_history"
