# Geometry and surface audit

## R1 result

The exact live-main public CAD-to-DAGMC example was reproduced at `de7d2978ff314b060ca2e6b10745a034e8b2a3c4`. The resulting 90-degree H5M contains 24 physical volumes, 142 DAGMC surfaces, and 18 original homogenized magnet solids. It contains no ports, no casing/winding split, and no magnet transform.

The mesh is topologically closed: PyMOAB and PyDAGMC reload, all 20,076 faceted edges match, all 142 surfaces seal, and all 24 physical volumes seal. Every original magnet's complete topology-sensed outer boundary passes edge-multiplicity and vector-area closure. The 36 deterministic magnet-entry rays hit their expected first surface, and bounded point-in-volume probes pass.

R1 nevertheless fails the physical overlap gate. Native DAGMC `overlap_check -p 2 -t 4` reports four nonadjacent contacts with no shared topological surface:

| Vacuum-vessel volume | Magnet | Magnet volume | Witness point [cm] |
|---:|---|---:|---|
| 6 | magnet-0005 | 12 | (1160.390, 380.476, 128.459) |
| 6 | magnet-0006 | 13 | (1095.150, 498.581, 92.590) |
| 6 | magnet-0011 | 18 | (476.445, 1095.280, -86.266) |
| 6 | magnet-0012 | 19 | (364.230, 1158.370, -121.329) |

These are classified as `TRUE_UNINTENDED_NONADJACENT_VOLUME_OVERLAP`, not shared-boundary contacts. The check's point sampling is non-exhaustive, so four is a lower-bound witness count rather than proof that no further intersections exist.

## OpenMC geometry debug

The qualified OpenMC 0.16.0 environment and existing nuclear-data library were used for a bounded 4,000-history geometry-debug attempt. After correcting an OpenMC-only CSG/DAGMC cell-ID collision with `auto_geom_ids=True`, both four-thread and one-thread runs entered batch 1 and reported overlapping cells 2 and 3 in DAGMC universe 1. One run then terminated with exit 139 and the confirming one-thread run terminated with exit 255. Zero lost particles and zero DAGMC navigation errors were therefore not proved.

The H5M SHA-256 remained unchanged before model export and after all debug attempts.

## Decision impact

The clean adapter passes byte parity and the original magnet envelopes close, but acceptance Gates C and G fail. No tally instrumentation or Prompt 7B work is authorized by this result.
