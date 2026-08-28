import pytest

from parastell.surface_source_accounting import (
    validate_surface_source_accounting,
)


def test_assertion_only_accounting_cannot_emit_complete():
    with pytest.raises(RuntimeError, match="caller-asserted"):
        validate_surface_source_accounting(
            {
                "classification": "COMPLETE_CROSSING_BANK",
                "source_files": [
                    {
                        "path": "arbitrary.bin",
                        "sha256": "a" * 64,
                        "record_count": 1,
                    }
                ],
            }
        )
