import json

import pytest

from parastell.schemas import SCHEMA_FILES
from parastell.schemas import schema_path


@pytest.mark.parametrize("name", SCHEMA_FILES)
def test_packaged_producer_schema_is_draft_2020_12_object(name):
    path = schema_path(name)
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert value["$id"].startswith("https://parastell.dev/schemas/")
    assert value["type"] == "object"
    assert value["required"]


def test_unknown_packaged_schema_is_rejected():
    with pytest.raises(KeyError):
        schema_path("consumer.schema.json")


def test_bundle_schema_matches_hardened_runtime_contract():
    schema = json.loads(
        schema_path("magnet_radiation_field_bundle.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert "bank_completeness" in schema["required"]
    provenance = schema["properties"]["provenance"]
    assert set(provenance["required"]) == {
        "parastell_commit",
        "openmc_version",
        "openmc_commit",
        "statepoint_sha256",
        "vmec_sha256",
        "coils_sha256",
        "source_mesh_sha256",
        "histories",
        "batches",
        "seeds",
    }
    completeness = schema["$defs"]["bankCompleteness"]
    assert {
        "status",
        "bank_count",
        "classification_counts",
        "record_count_total",
        "empty_bank_count",
        "all_banks_valid",
        "all_banks_complete",
        "empty_banks_are_physical_zero",
    } == set(completeness["required"])
    product = schema["$defs"]["product"]
    assert {
        "kind",
        "magnet_id",
        "path",
        "sha256",
        "size_bytes",
        "units",
        "normalization",
        "quantity",
    } == set(product["required"])
    conditions = product["allOf"]
    routed_kinds = {
        condition["if"]["properties"]["kind"]["const"]
        for condition in conditions
    }
    assert {
        "volume_scalar_flux",
        "boundary_phase_space",
        "damage_and_gas_production",
    } <= routed_kinds
