# Blanket material presets

ParaStell can resolve the continuous WISTELL-D radial roles into portable,
isotope-expanded OpenMC materials without importing private BlanketNeutronics
code or requiring PyNE at runtime.

The two full presets are `dcll` and `hcpb`. Controlled comparisons can replace
only one role, for example the breeder or high-temperature shield, while the
geometry and every other material remain fixed. The CLI is create-only:

```text
python -m scripts.build_blanket_material_bundle PURE_MATERIALS.json OUTPUT_DIR --preset dcll
```

Optional `--role-recipe-overrides` accepts a JSON object such as the examples
in `configs/`. Every output contains the pure-material database hash, frozen
reference repository revisions, recipe identities, isotope mass fractions,
density, citations, and a canonical bundle hash.

The generated material names exactly match the qualified nine-volume DAGMC
model: `first_wall`, `breeder`, `back_wall`,
`high_temperature_shield`, `vacuum_vessel`,
`low_temperature_shield`, and `homogenized_magnet`. The chamber and vacuum gap
remain void DAGMC domains and therefore do not receive OpenMC materials.

The global magnet is intentionally homogenized. The BlanketNeutronics coil
recipe is a transport proxy and does not claim to resolve REBCO. Explicit tape,
cable, casing, coolant-channel, and micrometre layers belong to the downstream
local-magnet models.

Material selection does not authorize transport and never modifies STEP or
H5M geometry. A selected bundle and `materials.xml` must be hash-bound into the
OpenMC build control before a campaign is launched.
