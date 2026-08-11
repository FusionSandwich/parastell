import json

import pytest

from parastell.comsol_fusion_magnets import (
    CentralSolenoidPFParameters,
    DShapedCoilParameters,
    FusionMagnetDesign,
    build_fusion_magnet_model,
    export_all_fusion_magnet_models,
    export_fusion_magnet_model,
    main,
)


pytest.importorskip("cadquery")


def test_default_models_have_nonempty_solid_parts():
    for design in FusionMagnetDesign:
        model = build_fusion_magnet_model(design)
        assert model.parts
        for shape in model.parts.values():
            solids = shape.solids().vals()
            assert solids
            assert sum(solid.Volume() for solid in solids) > 0.0


def test_d_shaped_parameters_reject_closed_aperture():
    with pytest.raises(ValueError, match="leaves no D-shaped aperture"):
        DShapedCoilParameters(
            inboard_radius_m=1.0,
            outboard_radius_m=2.0,
            half_height_m=1.0,
            radial_build_m=0.6,
            poloidal_build_m=0.2,
            toroidal_depth_m=0.2,
        )


def test_cs_parameters_reject_zero_modules():
    with pytest.raises(ValueError, match="cs_modules"):
        CentralSolenoidPFParameters(cs_modules=0)


def test_export_writes_step_manifest_and_java_importer(tmp_path):
    design = FusionMagnetDesign.SPHERICAL_TOKAMAK_TF
    files = export_fusion_magnet_model(
        build_fusion_magnet_model(design),
        tmp_path,
        formats=("step",),
    )
    assert files["combined_step"].is_file()
    assert files["comsol_java"].is_file()
    manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
    assert manifest["design"] == design.value
    assert set(manifest["parts"]) == {
        "tf_center_column",
        "tf_outer_return",
    }
    assert "ModelUtil.create" in files["comsol_java"].read_text(
        encoding="utf-8"
    )


def test_export_all_supports_stl_without_combined_step(tmp_path):
    outputs = export_all_fusion_magnet_models(
        tmp_path,
        formats=("stl",),
    )
    assert set(outputs) == set(FusionMagnetDesign)
    for files in outputs.values():
        assert "combined_step" not in files
        assert files["manifest"].is_file()
        assert "comsol_java" not in files
        assert any(name.endswith("_stl") for name in files)


def test_cli_exports_one_design(tmp_path, capsys):
    result = main(
        [
            "--design",
            FusionMagnetDesign.CENTRAL_SOLENOID_PF.value,
            "--output-dir",
            str(tmp_path),
            "--format",
            "step",
        ]
    )
    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert "combined_step" in output
