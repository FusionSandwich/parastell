# Energy-group registry

The magnet boundary bank stores continuous energy and correlated particle state.
Energy groups are named, purpose-specific projections and never replace that bank.

ParaStell vendors the exact small boundary definitions so production files do not
depend on network access or whichever OpenMC version happens to be installed.
Every entry includes an edge checksum, source revision, intended purpose, particle
type, and redistribution note. The loader additionally calculates the checksum of
the installed JSON file.

## Built-in structures

| Name | Particle | Groups | Intended use |
| --- | --- | ---: | --- |
| `smoke-7` | neutron | 7 | fast software tests only |
| `regression-182` | neutron | 182 | legacy scaling regression only |
| `CCFE-709` | neutron | 709 | SPECTRA-PKA and FISPACT compatibility |
| `UKAEA-1102` | neutron | 1102 | activation and transport audit reference |
| `smoke-42` | photon | 42 | fast software tests only |
| `CCFE-162` | photon | 162 | FISPACT gamma-induced activation interoperability |
| `photon-master-v1` | photon | 240 | candidate transport/condensation master |
| `continuous-photon` | photon | n/a | authoritative continuous-energy representation |

`CCFE-162` is not asserted to be optimal for photon heating or thin-layer
transport. A response-selected photon grid must be derived separately from the
continuous bank and protected interaction/heating responses.

## Sources

- `CCFE-709` and `UKAEA-1102` are copied from OpenMC `v0.16.0`, commit
  `617d35a5063c57796b43428bc401e627d2011046`.
- `CCFE-162` is copied from UKAEA FISPACT-II `Ebins_162.txt`, accessed
  2026-08-20. Its source SHA-256 is recorded in the JSON metadata.
- Smoke and regression structures are ParaStell test definitions and are marked
  non-production.

## CLI

```bash
parastell energy-groups list
parastell energy-groups inspect UKAEA-1102
parastell energy-groups inspect CCFE-162 --descending
parastell energy-groups validate custom.csv --particle neutron --units eV
parastell energy-groups compare CCFE-709 UKAEA-1102
```

Custom definitions are validated but not silently added to the global registry.
External formats requiring high-to-low boundaries can request descending output;
ParaStell stores all finite structures low-to-high in eV.
