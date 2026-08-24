# Magnet radiation-field bundle

`parastell.magnet_radiation_field_bundle/v1.0.0` is the solver-neutral handoff
between ParaStell/OpenMC reactor transport and downstream HTS transport and
damage workflows. Reading the bundle does not require ParaStell, OpenMC,
DAGMC, MOAB, or a CAD kernel.

The bundle keeps two different transport quantities separate:

- `boundary_phase_space` contains correlated surface-crossing position,
  direction, continuous energy, particle, weight, surface, and crossing sense.
- `volume_scalar_flux` contains OpenMC track-length flux divided by an explicit
  magnet-cell volume. This is the valid input for spectrum-only PKA folding.

A surface current is never relabeled or divided by area to manufacture scalar
flux. Every product is copied into the bundle and bound by SHA-256. The
manifest also binds the raw H5M hash, canonical DAGMC topology/geometry
fingerprint, source-definition hash, physical source rate, nuclear-data
provenance, material inventory, and magnet IDs.

## CLI

Create a JSON specification containing `provenance`, `source`, `nuclear_data`,
`materials`, `magnet_inventory`, `products`, and `verification`. Product paths
are resolved relative to the specification file.

```bash
parastell-magnet-handoff bundle \
  --spec radiation_bundle.json \
  --dagmc combined_reactor_magnet.h5m \
  --output-dir magnet_radiation_bundle
```

The CLI computes the canonical geometry fingerprint itself. It rejects missing
metadata, unknown product types, invalid normalization, a scalar-flux product
with surface-current semantics, missing files, and incompatible units.

## Downstream use

For deterministic tape replay, consume `boundary_phase_space` and preserve its
surface-local coordinates and directions. For SPECTRA-PKA or other
spectrum-only response folding, consume `volume_scalar_flux`; CCFE-709 is the
interoperability projection, while continuous boundary energies remain
authoritative for directional replay.
