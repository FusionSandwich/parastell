import json

from parastell.activation_ready_metadata import SCHEMA as ACTIVATION_SCHEMA
from parastell.magnet_geometry_interchange import SCHEMA_URI as GEOMETRY_SCHEMA
from parastell.material_manifest import NUCLEAR_DATA_SCHEMA
from parastell.material_manifest import SCHEMA as MATERIAL_SCHEMA
from parastell.schemas import SCHEMA_FILES
from parastell.schemas import schema_path


def test_medium_qualification_reuses_prompt1_contracts():
    assert GEOMETRY_SCHEMA == ("parastell.magnet_geometry_interchange/v1.0.0")
    assert ACTIVATION_SCHEMA == "parastell.activation_ready_metadata/v1.0.0"
    assert MATERIAL_SCHEMA == "parastell.magnet_material_manifest/v1.0.0"
    assert NUCLEAR_DATA_SCHEMA == "parastell.nuclear_data_audit/v1.0.0"

    geometry = json.loads(
        schema_path("magnet_geometry_interchange.schema.json").read_text(
            encoding="utf-8"
        )
    )
    activation = json.loads(
        schema_path("magnet_activation_ready_metadata.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert geometry["properties"]["schema"]["const"] == GEOMETRY_SCHEMA
    assert activation["properties"]["schema"]["const"] == ACTIVATION_SCHEMA


def test_parastell_does_not_create_a_competing_test_spectrum_contract():
    # The Prompt-1 test-spectrum package remains owned by DPA_workflow.
    assert not any("test_spectrum" in name for name in SCHEMA_FILES)
