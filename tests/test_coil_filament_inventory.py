from __future__ import annotations

import copy

import pytest

from parastell.coil_filament_inventory import (
    build_coil_filament_inventory,
    validate_coil_filament_inventory,
)


def _coil(path):
    path.write_text(
        "title\nperiods 4\nbegin filament\n"
        "1 0 0 1\n0 1 0 1\n-1 0 0 1\n0 -1 0 0\n"
        "2 0 0 1\n0 2 0 1\n-2 0 0 1\n0 -2 0 0\nend\n",
        encoding="utf-8",
    )
    return path


def test_inventory_is_generic_stable_and_self_sealed(tmp_path):
    coil = _coil(tmp_path / "coils.test")
    inventory = build_coil_filament_inventory(
        coil,
        start_line=3,
        scale_to_cm=100.0,
        input_coordinate_units="m",
        sector_start_degrees=15.0,
        sector_extent_degrees=73.0,
    )
    validate_coil_filament_inventory(inventory)
    assert inventory["filament_count"] == 2
    assert len({row["filament_id"] for row in inventory["filaments"]}) == 2
    assert inventory["sector"]["extent_degrees"] == 73.0
    assert inventory["claims"]["physical_global_volumes"] is False


@pytest.mark.parametrize("field", ["filament_id", "coordinates_sha256"])
def test_inventory_rejects_resealed_identity_tamper(tmp_path, field):
    coil = _coil(tmp_path / "coils.test")
    inventory = build_coil_filament_inventory(coil, start_line=3)
    changed = copy.deepcopy(inventory)
    changed["filaments"][0][field] = "0" * 64
    changed.pop("inventory_content_sha256")
    from parastell.coil_filament_inventory import canonical_sha256

    changed["inventory_content_sha256"] = canonical_sha256(changed)
    with pytest.raises(ValueError, match="inventory"):
        validate_coil_filament_inventory(changed)


def test_inventory_rejects_mutated_source_even_with_old_manifest(tmp_path):
    coil = _coil(tmp_path / "coils.test")
    inventory = build_coil_filament_inventory(coil, start_line=3)
    coil.write_text(
        coil.read_text(encoding="utf-8") + "# drift\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="source binding"):
        validate_coil_filament_inventory(inventory)


def test_inventory_rejects_resealed_fabricated_coordinate_identity(tmp_path):
    coil = _coil(tmp_path / "coils.test")
    inventory = build_coil_filament_inventory(coil, start_line=3)
    changed = copy.deepcopy(inventory)
    digest = "0" * 64
    changed["filaments"][0]["coordinates_sha256"] = digest
    changed["filaments"][0]["filament_id"] = f"filament-{digest[:16]}"
    changed.pop("inventory_content_sha256")
    from parastell.coil_filament_inventory import canonical_sha256

    changed["inventory_content_sha256"] = canonical_sha256(changed)
    with pytest.raises(ValueError, match="reproduce"):
        validate_coil_filament_inventory(changed)


def test_inventory_rejects_unterminated_filament(tmp_path):
    coil = tmp_path / "bad.coils"
    coil.write_text("x\ny\nz\n1 0 0 1\nend\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unterminated"):
        build_coil_filament_inventory(coil, start_line=3)
