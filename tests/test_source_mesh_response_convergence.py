import hashlib
import json
from pathlib import Path

from parastell.magnet_field_workflow import MagnetRadiationWorkflow
from parastell.source_convergence import (
    CONVERGED_STATUS,
    REQUIRED_CANDIDATE_SHAPES,
    SOURCE_METRICS,
    evaluate_source_mesh_convergence,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MEDIUM_CONFIG = (
    REPOSITORY_ROOT / "examples" / "magnet_radiation_medium_qualification.json"
)


def _hash(label):
    return hashlib.sha256(label.encode()).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _manifest(path, shape, delta):
    scale = 1.0 + delta
    return _write_json(
        path,
        {
            "schema": "parastell.magnet_source_stage/v1.0.0",
            "source_mesh": {
                "path": f"source-{'x'.join(map(str, shape))}.h5m",
                "sha256": _hash(str(shape)),
                "size_bytes": 1,
                "shape": list(shape),
                "tetrahedra": 1,
                "physical_source_rate_per_s": 2.4e20 * scale,
            },
            "convergence_observables": {
                "spatial_moments": {
                    "mean_x_cm": 100.0 * scale,
                    "mean_y_cm": 200.0 * scale,
                    "mean_z_cm": 50.0 * scale,
                    "mean_x2_cm2": (100.0 * scale) ** 2 + 100.0,
                    "mean_y2_cm2": (200.0 * scale) ** 2 + 100.0,
                    "mean_z2_cm2": (50.0 * scale) ** 2 + 100.0,
                },
                "dt_energy_moments": {
                    "mean_eV": 14.1e6 * scale,
                    "mean_squared_eV2": (14.1e6 * scale) ** 2 + 1.0e10,
                },
            },
        },
    )


def _response(path, response_ids, delta, cost):
    scale = 1.0 + delta
    rows = {}
    for index, response_id in enumerate(response_ids):
        if response_id == "whole_magnet.hotspot_location":
            rows[response_id] = {
                "value_xyz_cm": [10.0 * scale, 20.0, 30.0],
                "units": "cm",
                "coordinate_frame": "global_cartesian_cm",
                "magnet_id": "example-stellarator-sector-00-90deg-coil-0005",
                "bin_id": "hotspot-17",
            }
        elif response_id == "computational_cost":
            rows[response_id] = {"value": cost, "units": "core-s"}
        else:
            row = {"value": (index + 1.0) * scale, "units": "arb"}
            if response_id in {
                "whole_magnet.incoming_neutron_current",
                "whole_magnet.incoming_photon_current",
            }:
                row.update(
                    {
                        "direction": "incoming",
                        "crossing_selection": "inward_only",
                        "estimator": "direction_resolved_surface_current",
                    }
                )
            rows[response_id] = row
    return _write_json(
        path,
        {
            "schema": "test.medium_source_responses/v1",
            "whole_magnet_responses": rows,
        },
    )


def test_medium_configuration_reuses_prompt1_contracts_and_exact_identity(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PARASTELL_ARTIFACT_ROOT", str(tmp_path))
    workflow = MagnetRadiationWorkflow.from_json(MEDIUM_CONFIG)
    assert workflow.validate_configuration()["status"] == "PASS"
    config = workflow.config
    assert config["template_binding"]["path"].endswith(
        "magnet_radiation_smoke.json"
    )
    assert config["schema_reuse"] == {
        "magnet_geometry_interchange": (
            "parastell.magnet_geometry_interchange/v1.0.0"
        ),
        "activation_ready_metadata": (
            "parastell.activation_ready_metadata/v1.0.0"
        ),
        "boundary_handoff": "parastell.magnet_boundary_source/v2.1.0",
        "policy": "reuse_without_competing_contracts",
    }
    identity = config["representative_magnet_identity_lock"]
    assert identity["magnet_id"].endswith("coil-0005")
    assert identity["coils_sha256"] == (
        "69f508b216f0b674368ca8731d390c9d514736ff092f0ebecd854e8772ae04ab"
    )
    assert identity["casing_volume_id"] == 17
    assert identity["winding_pack_volume_id"] == 18
    assert identity["centreline_point_count"] == 129
    calibration = config["photon_calibration"]
    assert calibration["replica_count"] == 3
    assert calibration["histories_per_replica"] == 100_000
    assert len(set(calibration["seeds"])) == 3
    assert calibration["source_mesh_candidate"]["shape"] == [7, 41, 31]
    assert config["stages"]["prepare_unbiased_model"][
        "source_mesh_path"
    ].endswith("source_mesh_7x41x31.h5m")
    assert config["weight_window_policy"]["enabled"] is False
    assert not config["medium_history_policy"][
        "final_large_campaign_authorized"
    ]


def test_all_four_shapes_and_full_response_set_drive_selection(tmp_path):
    config = json.loads(MEDIUM_CONFIG.read_text(encoding="utf-8"))
    campaign = config["source_mesh_response_convergence"]
    assert campaign["required_shapes"] == [
        list(shape) for shape in REQUIRED_CANDIDATE_SHAPES
    ]
    stage = config["stages"]["qualify_source_convergence"]
    response_ids = stage["required_response_metric_ids"]
    assert set(response_ids) == set(stage["response_metric_tolerances"])
    assert {
        "whole_magnet.winding_pack_neutron_flux",
        "whole_magnet.winding_pack_photon_flux",
        "whole_magnet.casing_neutron_flux",
        "whole_magnet.casing_photon_flux",
        "whole_magnet.incoming_neutron_current",
        "whole_magnet.incoming_photon_current",
        "whole_magnet.neutron_heating",
        "whole_magnet.photon_heating",
        "whole_magnet.photon_production",
        "whole_magnet.hotspot_location",
    }.issubset(response_ids)

    candidates = []
    for index, (shape, delta) in enumerate(
        zip(REQUIRED_CANDIDATE_SHAPES, (0.2, 0.08, 0.004, 0.0))
    ):
        candidates.append(
            {
                "source_manifest_path": _manifest(
                    tmp_path / f"source-{index}.json", shape, delta
                ),
                "response_report_path": _response(
                    tmp_path / f"response-{index}.json",
                    response_ids,
                    delta,
                    cost=100.0 * (index + 1),
                ),
            }
        )
    result = evaluate_source_mesh_convergence(
        candidates,
        source_metric_tolerances={
            name: {"maximum_relative_change": 0.01} for name in SOURCE_METRICS
        },
        response_metric_tolerances=stage["response_metric_tolerances"],
        required_response_metric_ids=response_ids,
    )
    assert result["overall_status"] == CONVERGED_STATUS
    assert result["missing_evidence"] == []
    assert result["qualification_metric_count"] == (
        len(SOURCE_METRICS) + len(response_ids)
    )
    # The two finest candidates pass; cost selects 7x41x31 instead of
    # automatically choosing the largest source mesh.
    assert result["default_selection"]["selected_shape"] == [7, 41, 31]
