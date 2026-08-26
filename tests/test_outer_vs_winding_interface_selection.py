import pytest

from parastell.magnet_surface_audit import validate_coupling_interface


@pytest.mark.parametrize("name", ("outer_casing_external", "winding_pack"))
def test_exactly_one_supported_coupling_interface_is_accepted(name):
    assert validate_coupling_interface(name) == name


def test_ambiguous_or_absent_coupling_interface_is_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        validate_coupling_interface(
            {"outer_casing_external": True, "winding_pack": True}
        )
    with pytest.raises(ValueError, match="exactly one"):
        validate_coupling_interface([])
