import json

from parastell.activation.fispact import (
    FispactActivationBridge,
    FispactConfiguration,
)
from parastell.activation.model import (
    ActivationRegion,
    ActivationSchedule,
    ActivationStep,
)
from parastell.activation.spectrum_export import ActivationSpectrum


def test_fispact_bundle_has_independent_provenance(tmp_path):
    files = tmp_path / "files.template"
    files.write_text("test files mapping\n", encoding="ascii")
    config = FispactConfiguration(None, files, target_group_count=709)
    region = ActivationRegion(
        "m1", "cell", 1, 2, "cu", 1.0, 8.9, 300.0, (("Cu63", 1.0),), "0" * 64
    )
    schedule = ActivationSchedule(
        "duty",
        (ActivationStep(10.0, 2.0, "on"), ActivationStep(20.0, 0.0, "off")),
    )
    spectrum = ActivationSpectrum.create(
        "s", "neutron", [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0], "m1", 2.0
    )
    bridge = FispactActivationBridge(config)
    bundle = bridge.prepare(
        tmp_path / "fispact",
        region,
        schedule,
        spectrum,
        atoms={"Cu63": 1.0e22},
    )
    manifest = json.loads(
        (bundle / "manifest.json").read_text(encoding="ascii")
    )
    assert manifest["backend"] == "fispact-ii"
    assert manifest["execution_validated"] is False
    assert (bundle / "arb_flux").is_file()
