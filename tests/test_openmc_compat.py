import pytest

from parastell.openmc_compat import openmc_version_supported
from parastell.openmc_compat import parse_openmc_version
from parastell.openmc_compat import require_supported_openmc_version
from parastell.openmc_compat import statepoint_version_supported


@pytest.mark.parametrize(
    "value, expected",
    [
        ("0.16.0", True),
        ("0.16.1.dev46+g1d75981db", True),
        ("0.16.1-dev46", True),
        ("0.16.1rc1", True),
        ("0.16.0rc1", False),
        ("0.16.0.dev1", False),
        ("0.15.3", False),
        ("0.17.0", False),
        ("develop", False),
    ],
)
def test_openmc_version_supported(value, expected):
    assert openmc_version_supported(value) is expected


def test_parse_openmc_version_rejects_ambiguous_value():
    with pytest.raises(ValueError, match="invalid OpenMC version"):
        parse_openmc_version("0.16")


def test_require_supported_openmc_version_reports_range():
    with pytest.raises(RuntimeError, match=r">=0.16.0,<0.17.0"):
        require_supported_openmc_version("0.17.0", context="test")


@pytest.mark.parametrize(
    "value, expected",
    [
        ((0, 16, 0), True),
        ((0, 16, 1), True),
        ((0, 15, 3), False),
        ((0, 17, 0), False),
        ((0, 16), False),
        ((0, "bad", 1), False),
    ],
)
def test_statepoint_version_supported(value, expected):
    assert statepoint_version_supported(value) is expected
