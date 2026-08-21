import json

from parastell.activation.alara import (
    AlaraActivationBridge,
    AlaraConfiguration,
)
from parastell.activation.model import (
    ActivationRegion,
    ActivationSchedule,
    ActivationStep,
)
from parastell.activation.spectrum_export import ActivationSpectrum


def _inputs():
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
    return region, schedule, spectrum


def test_alara_bundle_preserves_region_schedule_and_flux(tmp_path):
    bridge = AlaraActivationBridge(
        AlaraConfiguration(
            None, 3, "descending", ("data_library test 1 alara.lib",)
        )
    )
    region, schedule, spectrum = _inputs()
    bundle = bridge.prepare(
        tmp_path / "alara",
        region,
        schedule,
        spectrum,
        mixture_lines=("element Cu 1.0 1.0",),
    )
    manifest = json.loads(
        (bundle / "manifest.json").read_text(encoding="ascii")
    )
    assert manifest["region"]["region_id"] == "m1"
    assert manifest["spectrum"]["total_flux_cm2_s"] == 6.0
    assert "parastell_flux_0 flux.txt 1.00000000000000000e+00" in (
        bundle / "alara.inp"
    ).read_text(encoding="ascii")
    assert manifest["execution_validated"] is False
