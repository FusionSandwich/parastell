# Blanket reference integration audit — 2026-08-31

## Read-only sources

The four requested `blanket`/`BlanketNeutronics` checkouts are mirrors of the
same tree at commit `bc4ab3d0f27369d4eda908a3fc187a10b6c7fedb`. The review used
`D:\Scratch\BlanketNeutronics` read-only. No file in any Blanket checkout was
modified or copied wholesale.

The reusable material inputs were cross-checked against:

- `FusionSandwich/fusion-material-db` commit
  `9d8f84ad7b3ac2c587edfbe8ec1ed74891484498`;
- `FusionSandwich/radial_build_tools` commit
  `d195d5a9f777c3ac42a9033646937558e07748e0`;
- the model-specific WISTELL-D DCLL material builder in BlanketNeutronics.

## Integrated capabilities

`parastell.blanket_materials` now provides:

- complete DCLL and HCPB material presets for all seven non-void radial roles;
- controlled breeder-only and shield-only substitutions for a non-Cartesian
  poster scan;
- isotope-expanded mixtures computed directly from a hash-bound
  fusion-material-db JSON file;
- deterministic, create-only OpenMC `materials.xml` output;
- mixture density, isotope fractions, citations, source revisions, and hashes;
- an explicit firewall keeping the global magnet homogenized and local REBCO
  tape geometry downstream.

The Blanket tally utilities require TBR, heating, damage energy, gas
production, fast flux/fluence, and uncertainty propagation. The current
ParaStell response plan already contains these responses and additionally
provides neutron/photon spectra, surface current and complete phase space,
local meshes, isotope/MT events, photon production, activation fields, and
downstream SPECTRA-PKA/Beyond-DPA handoffs. The legacy fixed cell IDs and
incomplete tally script were therefore not copied.

## Resolved discrepancies

Two reference generations disagree on parts of the DCLL recipe. The
model-specific WISTELL-D builder uses a WC/RAFM/helium high-temperature shield,
whereas one later radial-build example substitutes borated RAFM. The baseline
`dcll` preset follows the model-specific WISTELL-D recipe. The borated mixture
is retained as the independently selectable HCPB shield case; the two are not
silently blended.

The qualified direct-90 geometry and its pointwise thickness fields are not
changed by these presets. The older Blanket constant-thickness geometry is
reference evidence only and does not supersede the accepted nine-volume H5M.
The vacuum-vessel and low-temperature-shield recipes follow the newer portable
radial-build definitions; the exact selected bundle records those recipe IDs
so they cannot be confused with the older SS316L-only variants.

## Bounded generation evidence

Two local, zero-transport bundles were generated from the exact
fusion-material-db JSON SHA-256
`bb15bbad56395269b2e07df183d35fa37894dd5f752ebc793166566d40d61577`.

| Preset | Bundle SHA-256 | OpenMC XML SHA-256 | Result |
|---|---|---|---|
| DCLL | `26efec77e4579e8fbba207eaa983072c464133a19fc743ed45b88c7e05821d50` | `09c075117e4a9150a79551928d4ad581b68ba48730e0c6cd3cbe25ca5199f179` | 7 roles, finite positive densities, expanded nuclides, structural XML pass |
| HCPB | `cd93eccd62cc80796c37e5de420084303910a0af2c5433289e746bb1cc4ca01a` | `a2131875e624aed091cb69f1d51f166f22117b2904ea8bf02343aecaab7cffc8` | 7 roles, finite positive densities, expanded nuclides, structural XML pass |

This Windows Python does not contain OpenMC, so no local OpenMC parser or
transport was claimed. The XML must next be bound and parsed in the already
qualified OpenMC 0.16 Docker/Bateman environment. No dependency was installed.
The 38 focused material, campaign, OpenMC-model, tally, response-plan, and
material-identity tests pass. Full-suite collection on this Windows Python is
environment-blocked by the already-known absent OpenMC, Gmsh, and PyMOAB
optional runtimes; this is not classified as a source failure.

## Decision

The material/case-definition infrastructure needed for DCLL, HCPB,
breeder-only, shield-only, and location-controlled poster studies is
implemented. A scientific case still must be selected and its exact material
bundle hash-bound before the authorized medium/poster transport campaign.
