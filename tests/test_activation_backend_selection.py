from pathlib import Path

import pytest

from parastell.activation.backends import ActivationBackend
from parastell.activation.backends import BackendCapability
from parastell.activation.backends import select_activation_backend


def _cap(backend, complete):
    return BackendCapability(
        backend,
        complete,
        complete,
        str(Path("tool")) if complete else None,
        None,
        (),
        () if complete else ("missing",),
    )


def test_openmc_is_primary_and_fispact_is_benchmark():
    caps = {
        ActivationBackend.OPENMC_R2S: _cap(ActivationBackend.OPENMC_R2S, True),
        ActivationBackend.ALARA: _cap(ActivationBackend.ALARA, True),
        ActivationBackend.FISPACT_II: _cap(ActivationBackend.FISPACT_II, True),
    }
    selected = select_activation_backend(caps)
    assert selected["selected"] == "openmc-r2s"
    assert selected["fispact_independent_benchmark"] is True


def test_no_backend_fails_loudly():
    caps = {backend: _cap(backend, False) for backend in ActivationBackend}
    with pytest.raises(RuntimeError, match="no complete activation backend"):
        select_activation_backend(caps)
