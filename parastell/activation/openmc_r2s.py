"""OpenMC 0.16 rigorous two-step activation workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .model import ActivationSchedule, base_manifest, sha256_file, write_json


def openmc_r2s_capability_report() -> dict[str, Any]:
    """Probe the installed OpenMC API rather than trusting a version string."""

    try:
        import openmc
        import openmc.deplete
    except ImportError as error:
        return {"passes": False, "error": str(error)}
    report = {
        "version": openmc.__version__,
        "R2SManager": hasattr(openmc.deplete, "R2SManager"),
        "DecaySpectrum": hasattr(openmc.stats, "DecaySpectrum"),
        "ParentNuclideFilter": hasattr(openmc, "ParentNuclideFilter"),
    }
    report["passes"] = all(
        report[name]
        for name in ("R2SManager", "DecaySpectrum", "ParentNuclideFilter")
    ) and tuple(int(value) for value in openmc.__version__.split(".")[:2]) >= (
        0,
        16,
    )
    return report


@dataclass(frozen=True)
class OpenMCR2SConfiguration:
    """Reproducible controls for an OpenMC R2S calculation."""

    chain_file: Path
    schedule: ActivationSchedule
    output_dir: Path
    photon_time_indices: tuple[int, ...] | None = None
    reaction_rate_mode: str = "direct"
    by_parent_nuclide: bool = True

    def __post_init__(self):
        if self.reaction_rate_mode not in {"direct", "flux"}:
            raise ValueError("reaction_rate_mode must be direct or flux")
        if not self.chain_file.is_file():
            raise FileNotFoundError(
                f"OpenMC depletion chain not found: {self.chain_file}"
            )


class OpenMCR2SActivationWorkflow:
    """Thin, audited driver over the official OpenMC 0.16 R2S manager."""

    def __init__(
        self,
        neutron_model,
        domains: Sequence[Any] | Any,
        configuration: OpenMCR2SConfiguration,
        *,
        photon_model=None,
    ):
        report = openmc_r2s_capability_report()
        if not report.get("passes"):
            raise RuntimeError(f"OpenMC R2S capability gate failed: {report}")
        import openmc.deplete

        self.configuration = configuration
        self.capabilities = report
        self.manager = openmc.deplete.R2SManager(
            neutron_model, domains, photon_model=photon_model
        )

    def run(
        self,
        *,
        bounding_boxes=None,
        mat_vol_kwargs: dict[str, Any] | None = None,
        run_kwargs: dict[str, Any] | None = None,
        operator_kwargs: dict[str, Any] | None = None,
    ) -> Path:
        output = self.configuration.output_dir.resolve()
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(
                f"R2S output directory is not empty: {output}"
            )
        operator = dict(operator_kwargs or {})
        requested_mode = operator.get("normalization_mode", "source-rate")
        if requested_mode != "source-rate":
            raise ValueError("OpenMC R2S normalization must be source-rate")
        operator["normalization_mode"] = "source-rate"
        micro = {
            "reaction_rate_mode": self.configuration.reaction_rate_mode,
            "include_model_tallies": True,
        }
        result = self.manager.run(
            timesteps=self.configuration.schedule.timesteps_s,
            source_rates=self.configuration.schedule.source_rates_n_s,
            timestep_units="s",
            photon_time_indices=self.configuration.photon_time_indices,
            output_dir=output,
            bounding_boxes=bounding_boxes,
            chain_file=self.configuration.chain_file,
            micro_kwargs=micro,
            mat_vol_kwargs=mat_vol_kwargs,
            run_kwargs=run_kwargs,
            operator_kwargs=operator,
            by_parent_nuclide=self.configuration.by_parent_nuclide,
        )
        self.write_manifest(
            Path(result) / "parastell_activation_manifest.json"
        )
        return Path(result)

    def load_results(self, path: str | Path) -> None:
        self.manager.load_results(path)

    def manifest(self) -> dict[str, Any]:
        return {
            **base_manifest(),
            "kind": "openmc_r2s",
            "backend": "openmc-r2s",
            "openmc_capabilities": self.capabilities,
            "chain_file": str(self.configuration.chain_file.resolve()),
            "chain_sha256": sha256_file(self.configuration.chain_file),
            "schedule": self.configuration.schedule.as_dict(),
            "domain_method": self.manager.method,
            "domain_ids": [
                getattr(domain, "id", None) for domain in self.manager.domains
            ],
            "reaction_rate_mode": self.configuration.reaction_rate_mode,
            "normalization_mode": "source-rate",
            "by_parent_nuclide": self.configuration.by_parent_nuclide,
            "photon_time_indices": self.configuration.photon_time_indices,
        }

    def write_manifest(self, path: str | Path) -> Path:
        return write_json(path, self.manifest())
