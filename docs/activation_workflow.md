# ParaStell activation workflow

ParaStell uses one physical activation problem definition for three backends:

1. OpenMC 0.16 rigorous two-step (R2S), the primary implementation.
2. ALARA, when its activation libraries cover a capability missing from the
   OpenMC chain.
3. FISPACT-II, as a fallback and an independent benchmark when its executable
   and licensed data are available.

The transport spectrum remains a group-integrated scalar flux in
`particles/cm^2/s`. It carries the physical neutron source rate at which that
flux was calculated. Irradiation steps are scaled only by
`step_source_rate/reference_source_rate`; a zero source rate is cooling. This
prevents per-source and per-second normalizations from being mixed.

## OpenMC R2S

`OpenMCR2SActivationWorkflow` wraps the official OpenMC 0.16 `R2SManager` and
retains its four independent stages:

1. neutron transport and microscopic cross sections;
2. source-rate-normalized activation;
3. radionuclide-resolved decay-photon source generation;
4. photon transport, optionally tallied by parent nuclide.

The workflow refuses a nonempty output directory and rejects any requested
normalization other than `source-rate`. A depletion chain must pass
`audit_activation_chain` against the transport cross-section index before a
scientific run. In particular, the local ENDF/B-VII.1 transport index must not
be silently combined with the currently discovered ENDF/B-VIII.0-fast chain.

## ALARA

`AlaraActivationBridge` writes a point-geometry activation bundle for each
transport region. The caller must supply the exact ALARA mixture lines and
library directives; ParaStell does not guess library syntax, isotopic
abundances, or data provenance. Intermediate shutdown periods are represented
as post-pulse delays and final cooling outputs use cumulative times.

## FISPACT-II

`FispactActivationBridge` writes the documented descending-energy `arb_flux`
input for `GRPCONVERT`, the physical irradiation history, the atom inventory,
and an immutable manifest. The user supplies a licensed FISPACT-II `files`
mapping. Preparation is not evidence that FISPACT-II ran: the manifest remains
`execution_validated: false` until the configured executable returns
successfully.

## Commands

```text
parastell activation audit-environment
parastell activation audit-chain --chain CHAIN.xml \
  --cross-sections cross_sections.xml --nuclide Cu63
parastell activation export-spectrum --input spectrum.json \
  --format fispact-arb --output arb_flux
parastell activation inspect-spectrum spectrum.json
```

The OpenMC, ALARA, and FISPACT Python adapters are public under
`parastell.activation`. Backend discovery never installs software and backend
selection fails if no complete backend is available.

## Scientific limits

- Activation requires volume-integrated scalar flux and reaction data, not the
  magnet boundary current alone. The directional surface bank remains the
  authoritative deterministic HTS boundary condition, while activation uses
  dedicated cell or mesh tallies inside materials.
- ALARA and FISPACT-II bundles are not validated calculations until run against
  explicit compatible libraries.
- Shutdown photon transport is not a charged-particle transport calculation.
- Activity, decay heat, shutdown dose, PKA production, and material-property
  degradation are distinct responses and are reported separately.
