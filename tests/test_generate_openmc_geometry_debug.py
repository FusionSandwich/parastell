import json
from pathlib import Path

import pytest

from parastell.reference_geometry import sha256_file
from scripts.generate_openmc_geometry_debug import model_xml_contract
from scripts.generate_openmc_geometry_debug import nuclear_data_manifest


TEST_FILES = Path(__file__).parent / "files_for_tests"


def _write_cross_sections(tmp_path):
    rows = []
    for nuclide in ("H1", "Fe56", "Li6", "Cu63"):
        library = tmp_path / f"{nuclide}.h5"
        library.write_bytes(f"test-{nuclide}".encode())
        rows.append(
            f'<library materials="{nuclide}" path="{library.name}" '
            'type="neutron" />'
        )
    path = tmp_path / "cross_sections.xml"
    path.write_text(
        "<cross_sections>\n" + "\n".join(rows) + "\n</cross_sections>\n",
        encoding="utf-8",
    )
    return path


def test_nuclear_data_manifest_binds_each_required_library(tmp_path):
    cross_sections = _write_cross_sections(tmp_path)
    first = nuclear_data_manifest(cross_sections)
    assert [row["nuclide"] for row in first["libraries"]] == [
        "H1",
        "Fe56",
        "Li6",
        "Cu63",
    ]

    (tmp_path / "Li6.h5").write_bytes(b"mutated")
    second = nuclear_data_manifest(cross_sections)
    assert first["manifest_sha256"] != second["manifest_sha256"]


def test_generator_requires_preregistered_nuclear_data_manifest(tmp_path):
    from scripts.generate_openmc_geometry_debug import generate

    cross_sections = _write_cross_sections(tmp_path)
    dagmc = TEST_FILES / "one_cube.h5m"
    source = TEST_FILES / "source_mesh.h5m"
    criteria = tmp_path / "criteria.json"
    criteria.write_text("{}", encoding="utf-8")
    domain = tmp_path / "domain.json"
    domain.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="nuclear-data manifest hash"):
        generate(
            dagmc,
            source,
            tmp_path / "unused",
            str(cross_sections),
            expected_dagmc_sha256=sha256_file(dagmc),
            expected_source_mesh_sha256=sha256_file(source),
            acceptance_criteria=criteria,
            expected_acceptance_criteria_sha256=sha256_file(criteria),
            source_domain_audit=domain,
            expected_source_domain_audit_sha256=sha256_file(domain),
            expected_reference_source_mesh_sha256="d" * 64,
            expected_reference_source_fingerprint="e" * 64,
            expected_nuclear_data_manifest_sha256="0" * 64,
            seed=20260827,
        )


def test_model_xml_contract_treats_omitted_auto_ids_as_false(tmp_path):
    model = tmp_path / "model.xml"
    model.write_text(
        """<model>
<materials><cross_sections>/data/cross_sections.xml</cross_sections>
<material id="1" name="Vacuum"><density units="g/cm3" value="1e-20"/>
<nuclide name="H1" ao="1.0"/></material></materials>
<geometry><cell id="12" fill="14" region="-11" universe="13"/>
<dagmc_universe id="14" filename="/geometry/candidate.h5m"/>
<surface id="11" type="sphere" boundary="vacuum" coeffs="0 0 0 2500"/>
</geometry>
<settings><run_mode>fixed source</run_mode><particles>2000</particles>
<batches>2</batches><seed>20260827</seed>
<source particle="neutron"><space type="point"><parameters>1 2 3</parameters></space>
<angle type="isotropic"/><energy type="discrete"><parameters>14100000 1</parameters></energy>
</source></settings></model>""",
        encoding="utf-8",
    )

    contract = model_xml_contract(model)
    assert contract["dagmc_auto_geom_ids"] == "false"
    assert contract["wrapper_surface"]["coefficients_cm"] == [
        0.0,
        0.0,
        0.0,
        2500.0,
    ]
    assert contract["wrapper_cell"] == {
        "id": 12,
        "fill": 14,
        "region": "-11",
        "universe": 13,
    }


def test_generator_preserves_native_ids_and_requires_registered_seed(tmp_path):
    pytest.importorskip("pymoab")
    pytest.importorskip("openmc")
    from scripts.generate_openmc_geometry_debug import generate

    dagmc = TEST_FILES / "one_cube.h5m"
    source = TEST_FILES / "source_mesh.h5m"
    criteria = tmp_path / "criteria.json"
    criteria.write_text(
        json.dumps(
            {
                "openmc_geometry_debug": {
                    "histories": 4000,
                    "seeds": [20260827, 20260828],
                }
            }
        ),
        encoding="utf-8",
    )
    source_domain = tmp_path / "source_domain.json"
    cross_sections = _write_cross_sections(tmp_path)
    source_domain.write_text(
        json.dumps(
            {
                "schema": "parastell.source_domain_audit/v1.0.0",
                "source_domain_gate_pass": True,
                "raw_h5m_sha256": sha256_file(dagmc),
                "source_mesh_sha256": sha256_file(source),
                "acceptance_criteria_sha256": sha256_file(criteria),
                "reference_source_mesh_sha256": "d" * 64,
                "expected_reference_source_mesh_sha256": "d" * 64,
                "reference_source_mesh_identity": {
                    "canonical_fingerprint": "e" * 64
                },
                "expected_reference_source_fingerprint": "e" * 64,
                "reference_source_fingerprint_matches_expected": True,
                "input_immutability_pass": True,
                "source_mesh_semantic_identity_pass": True,
                "invalid_tetrahedron_count": 0,
                "negative_source_strength_count": 0,
                "invalid_vertex_sample_count": 0,
                "invalid_quadrature_point_count": 0,
                "invalid_sample_point_count": 0,
                "selected_source_point_cm": [0.5, 0.5, 0.5],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "replica"

    receipt = generate(
        dagmc,
        source,
        output,
        str(cross_sections),
        expected_dagmc_sha256=sha256_file(dagmc),
        expected_source_mesh_sha256=sha256_file(source),
        acceptance_criteria=criteria,
        expected_acceptance_criteria_sha256=sha256_file(criteria),
        source_domain_audit=source_domain,
        expected_source_domain_audit_sha256=sha256_file(source_domain),
        expected_reference_source_mesh_sha256="d" * 64,
        expected_reference_source_fingerprint="e" * 64,
        expected_nuclear_data_manifest_sha256=nuclear_data_manifest(
            cross_sections
        )["manifest_sha256"],
        seed=20260827,
    )

    assert receipt["seed"] == 20260827
    assert receipt["histories"] == 4000
    assert receipt["h5m_mutated"] is False
    wrapper = receipt["external_csg_vacuum_boundary"]
    assert (
        wrapper["surface_id"]
        > receipt["native_geometry_ids"]["maximum_native_id"]
    )
    assert (
        wrapper["wrapper_cell_id"]
        > receipt["native_geometry_ids"]["maximum_native_id"]
    )
    assert (
        'auto_geom_ids="true"'
        not in (output / "model.xml").read_text(encoding="utf-8").lower()
    )
    assert sha256_file(dagmc) == receipt["raw_h5m_sha256_before"]

    with pytest.raises(FileExistsError):
        generate(
            dagmc,
            source,
            output,
            str(cross_sections),
            expected_dagmc_sha256=sha256_file(dagmc),
            expected_source_mesh_sha256=sha256_file(source),
            acceptance_criteria=criteria,
            expected_acceptance_criteria_sha256=sha256_file(criteria),
            source_domain_audit=source_domain,
            expected_source_domain_audit_sha256=sha256_file(source_domain),
            expected_reference_source_mesh_sha256="d" * 64,
            expected_reference_source_fingerprint="e" * 64,
            expected_nuclear_data_manifest_sha256=nuclear_data_manifest(
                cross_sections
            )["manifest_sha256"],
            seed=20260827,
        )


def test_generator_rejects_unregistered_seed_before_openmc_import(tmp_path):
    from scripts.generate_openmc_geometry_debug import generate

    criteria = tmp_path / "criteria.json"
    criteria.write_text(
        json.dumps(
            {
                "openmc_geometry_debug": {
                    "histories": 4000,
                    "seeds": [20260827, 20260828],
                }
            }
        ),
        encoding="utf-8",
    )
    domain = tmp_path / "domain.json"
    domain.write_text("{}", encoding="utf-8")
    cross_sections = _write_cross_sections(tmp_path)
    dagmc = TEST_FILES / "one_cube.h5m"
    source = TEST_FILES / "source_mesh.h5m"

    with pytest.raises(ValueError, match="seed is not preregistered"):
        generate(
            dagmc,
            source,
            tmp_path / "unused",
            str(cross_sections),
            expected_dagmc_sha256=sha256_file(dagmc),
            expected_source_mesh_sha256=sha256_file(source),
            acceptance_criteria=criteria,
            expected_acceptance_criteria_sha256=sha256_file(criteria),
            source_domain_audit=domain,
            expected_source_domain_audit_sha256=sha256_file(domain),
            expected_reference_source_mesh_sha256="d" * 64,
            expected_reference_source_fingerprint="e" * 64,
            expected_nuclear_data_manifest_sha256=nuclear_data_manifest(
                cross_sections
            )["manifest_sha256"],
            seed=1,
        )


def test_generator_rejects_incomplete_or_forged_source_domain_receipt(
    tmp_path,
):
    from scripts.generate_openmc_geometry_debug import generate

    dagmc = TEST_FILES / "one_cube.h5m"
    source = TEST_FILES / "source_mesh.h5m"
    criteria = tmp_path / "criteria.json"
    criteria.write_text(
        json.dumps(
            {
                "openmc_geometry_debug": {
                    "histories": 4000,
                    "seeds": [20260827, 20260828],
                }
            }
        ),
        encoding="utf-8",
    )
    forged = tmp_path / "forged.json"
    forged.write_text(
        json.dumps(
            {
                "source_domain_gate_pass": True,
                "raw_h5m_sha256": sha256_file(dagmc),
                "source_mesh_sha256": sha256_file(source),
                "selected_source_point_cm": [0.5, 0.5, 0.5],
            }
        ),
        encoding="utf-8",
    )
    cross_sections = _write_cross_sections(tmp_path)

    with pytest.raises(ValueError, match="schema mismatch"):
        generate(
            dagmc,
            source,
            tmp_path / "unused",
            str(cross_sections),
            expected_dagmc_sha256=sha256_file(dagmc),
            expected_source_mesh_sha256=sha256_file(source),
            acceptance_criteria=criteria,
            expected_acceptance_criteria_sha256=sha256_file(criteria),
            source_domain_audit=forged,
            expected_source_domain_audit_sha256=sha256_file(forged),
            expected_reference_source_mesh_sha256="d" * 64,
            expected_reference_source_fingerprint="e" * 64,
            expected_nuclear_data_manifest_sha256=nuclear_data_manifest(
                cross_sections
            )["manifest_sha256"],
            seed=20260827,
        )


def test_generator_rejects_nonboolean_or_rebound_source_domain_gate(
    tmp_path, monkeypatch
):
    from scripts.generate_openmc_geometry_debug import generate

    dagmc = TEST_FILES / "one_cube.h5m"
    source = TEST_FILES / "source_mesh.h5m"
    criteria = tmp_path / "criteria.json"
    criteria.write_text(
        json.dumps(
            {
                "openmc_geometry_debug": {
                    "histories": 4000,
                    "seeds": [20260827, 20260828],
                }
            }
        ),
        encoding="utf-8",
    )
    cross_sections = _write_cross_sections(tmp_path)
    base = {
        "schema": "parastell.source_domain_audit/v1.0.0",
        "source_domain_gate_pass": "true",
        "raw_h5m_sha256": sha256_file(dagmc),
        "source_mesh_sha256": sha256_file(source),
        "acceptance_criteria_sha256": sha256_file(criteria),
        "reference_source_mesh_sha256": "d" * 64,
        "expected_reference_source_mesh_sha256": "d" * 64,
        "reference_source_mesh_identity": {"canonical_fingerprint": "e" * 64},
        "expected_reference_source_fingerprint": "e" * 64,
    }
    domain = tmp_path / "domain.json"
    domain.write_text(json.dumps(base), encoding="utf-8")

    with pytest.raises(ValueError, match="gate has not passed"):
        generate(
            dagmc,
            source,
            tmp_path / "unused-a",
            str(cross_sections),
            expected_dagmc_sha256=sha256_file(dagmc),
            expected_source_mesh_sha256=sha256_file(source),
            acceptance_criteria=criteria,
            expected_acceptance_criteria_sha256=sha256_file(criteria),
            source_domain_audit=domain,
            expected_source_domain_audit_sha256=sha256_file(domain),
            expected_reference_source_mesh_sha256="d" * 64,
            expected_reference_source_fingerprint="e" * 64,
            expected_nuclear_data_manifest_sha256=nuclear_data_manifest(
                cross_sections
            )["manifest_sha256"],
            seed=20260827,
        )

    base["source_domain_gate_pass"] = True
    base["expected_reference_source_mesh_sha256"] = "0" * 64
    domain.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ValueError, match="expected reference-source"):
        generate(
            dagmc,
            source,
            tmp_path / "unused-b",
            str(cross_sections),
            expected_dagmc_sha256=sha256_file(dagmc),
            expected_source_mesh_sha256=sha256_file(source),
            acceptance_criteria=criteria,
            expected_acceptance_criteria_sha256=sha256_file(criteria),
            source_domain_audit=domain,
            expected_source_domain_audit_sha256=sha256_file(domain),
            expected_reference_source_mesh_sha256="d" * 64,
            expected_reference_source_fingerprint="e" * 64,
            expected_nuclear_data_manifest_sha256=nuclear_data_manifest(
                cross_sections
            )["manifest_sha256"],
            seed=20260827,
        )
