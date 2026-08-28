import json

import pytest

from scripts.validate_geometry_runtime_receipt import SCHEMA
from scripts.validate_geometry_runtime_receipt import _publish_create_only


def test_runtime_validation_schema_is_frozen():
    assert SCHEMA == "parastell.geometry_runtime_validation/v1.0.0"


def test_runtime_validation_publication_is_create_only(tmp_path):
    destination = tmp_path / "validation.json"
    destination.write_text('{"owner":"existing"}\n', encoding="utf-8")

    with pytest.raises(FileExistsError):
        _publish_create_only(destination, {"owner": "validator"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "owner": "existing"
    }
    assert list(tmp_path.glob("*.pending")) == []
